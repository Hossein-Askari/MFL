import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import make_grid

import glf.options.options as option
from glf.data import create_dataloader, create_dataset
from glf.data.data_sampler import DistIterSampler
from glf.metrics import InceptionPredictor, frechet_distance, compute_prd_from_embedding, prd_to_max_f_beta_pair
from glf.models import create_model
from glf.utils import util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_dist(backend='nccl', **kwargs):
	"""initialization for distributed training"""
	if mp.get_start_method(allow_none=True) != 'spawn':
		mp.set_start_method('spawn')
	rank = int(os.environ['RANK'])
	num_gpus = torch.cuda.device_count()
	torch.cuda.set_device(rank % num_gpus)
	dist.init_process_group(backend=backend, **kwargs)


def main():
	#### options
	parser = argparse.ArgumentParser()
	parser.add_argument('-opt', type=str, help='Path to option YAML file.')
	parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
	parser.add_argument('--local_rank', type=int, default=0)
	args = parser.parse_args()
	opt = option.parse(args.opt, is_train=True)

	#### distributed training settings
	if args.launcher == 'none':  # disabled distributed training
		opt['dist'] = False
		rank = -1
		print('Disabled distributed training.')
	else:
		opt['dist'] = True
		init_dist()
		world_size = torch.distributed.get_world_size()
		rank = torch.distributed.get_rank()

	#### loading resume state if exists
	if opt['path'].get('resume_state', None):
		# distributed resuming: all load into default GPU
		device_id = torch.cuda.current_device()
		resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
		option.check_resume(opt, resume_state['iter'])  # check resume options
	else:
		resume_state = None

	#### mkdir and loggers
	if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
		if resume_state is None:
			util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
			util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and 'pretrained_' not in key and 'resume' not in key))

		# config loggers. Before it, the log will not work
		util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
						  screen=True, tofile=True)
		logger = logging.getLogger('base')
		logger.info(option.dict2str(opt))
		# tensorboard logger
		if opt['use_tb_logger'] and 'debug' not in opt['name']:
			from tensorboardX import SummaryWriter
			tb_logger = SummaryWriter(log_dir='tb_logger/' + opt['name'])
	else:
		util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
		logger = logging.getLogger('base')

	# convert to NoneDict, which returns None for missing keys
	opt = option.dict_to_nonedict(opt)

	#### random seed
	seed = opt['train']['manual_seed']
	if seed is None:
		seed = random.randint(1, 10000)
	if rank <= 0:
		logger.info('Random seed: {}'.format(seed))
	util.set_random_seed(seed)
	torch.backends.cudnn.benchmark = True
	# torch.backends.cudnn.deterministic = True
	

	#### create train and val dataloader
	dataset_ratio = 200  # enlarge the size of each epoch
	for phase, dataset_opt in opt['datasets'].items():
		if phase == 'train':
			train_set = create_dataset(dataset_opt, is_train=True)
			train_size = int(len(train_set) / dataset_opt['batch_size'])
			total_iters = int(opt['train']['niter'])
			total_epochs = int(math.ceil(total_iters / train_size))

			if opt['dist']:
				train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
				total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
			else:
				train_sampler = None

			train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
			
			if rank <= 0:
				logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
				logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
		
		elif phase == 'val':
			val_set = create_dataset(dataset_opt, is_train=False)
			val_loader = create_dataloader(val_set, dataset_opt, opt)
			if rank <= 0:
				logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
		else:
			raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
	assert train_loader is not None


	#### create model
	model = create_model(opt)
	if opt['datasets'].get('val', None) and opt['train']['val_calculate_fid_prd'] and \
			opt['datasets']['val']['name'] in ['MNIST', 'SVHN', 'CIFAR-10', 'CelebA', 'dots']:
		predictor_device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
		predictor_dim = 2048
		predictor = InceptionPredictor(output_dim=predictor_dim).to(predictor_device)
	else:
		predictor = None

	#### resume training
	if resume_state:
		logger.info('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))
		start_epoch = resume_state['epoch']
		current_step = resume_state['iter']
		model.resume_training(resume_state)  # handle optimizers and schedulers
	else:
		start_epoch = 1
		current_step = 0

	#### training
	logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
	for epoch in range(start_epoch, total_epochs+1):

		if opt['dist']:
			train_sampler.set_epoch(epoch)
		for _, train_data in enumerate(train_loader):
			current_step += 1
			if current_step > total_iters:
				break

			#### training
			model.feed_data(train_data)
			model.optimize_parameters(current_step)

			#### update learning rate
			model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

			#### log
			if current_step % opt['logger']['print_freq'] == 0:
				logs = model.get_current_log()
				message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
				for v in model.get_current_learning_rate():
					message += '{:.3e},'.format(v)
				message += ')] '
				for k, v in logs.items():
					message += '{:s}: {:.4e} '.format(k, v)
					
					# tensorboard logger
					if opt['use_tb_logger'] and 'debug' not in opt['name']:
						if rank <= 0:
							tb_logger.add_scalar(k, v, current_step)
				if rank <= 0:
					logger.info(message)

		#### validation
		if opt['datasets'].get('val', None) and (epoch % opt['train']['val_freq'] == 0 or epoch == total_epochs):
			if rank <= 0:
				# save 60 samples generated from noise
				samples = model.sample_images(60)
				grid = make_grid(samples, nrow=6)
				grid = util.tensor2img(grid)
				sample_name = 'latest.png' if epoch == total_epochs else '{:d}.png'.format(epoch)
				util.save_img(grid, os.path.join(opt['path']['samples'], sample_name))
				del samples, grid

				# calculate FID
				if predictor is not None:
					art_samples = []
					true_samples = []
					logger.info('Calculating FID and PRD:')

					if opt['train']['val_num_batches'] is None or epoch == total_epochs:
						num_val_batches = len(val_loader)
					else:
						num_val_batches = int(opt['train']['val_num_batches'])
					pbar = util.ProgressBar(num_val_batches)
					
					preds_val=[]
					gts_val=[]
					lt_cls_val = 0
					
					for k, val_data in enumerate(val_loader):
						if k >= num_val_batches:
							break

						model.feed_data(val_data)
						l_cls_eval, pred_val, gt_val = model.loss_class_eval()
						lt_cls_val += l_cls_eval.item()
						preds_val.extend(pred_val)
						gts_val.extend(gt_val)

						samples = model.sample_images(val_data[0].shape[0]).to(predictor_device)
						val_data[0] = val_data[0].to(predictor_device)
						art_samples.append(predictor(samples).detach().cpu().numpy())
						true_samples.append(predictor(val_data[0]).detach().cpu().numpy())

						pbar.update('batch #{}'.format(k))

						del samples, val_data[0]

					art_samples = np.concatenate(art_samples, axis=0)
					true_samples = np.concatenate(true_samples, axis=0)

					art_samples = art_samples.reshape(-1, predictor_dim)
					true_samples = true_samples.reshape(-1, predictor_dim)
					FID = frechet_distance(true_samples, art_samples)
					precision, recall = compute_prd_from_embedding(true_samples, art_samples)
					f_8, f_1_8 = prd_to_max_f_beta_pair(precision, recall, beta=8)
					score_cls_val = (np.mean(np.asarray(preds_val) == np.asarray(gts_val))).astype(np.float)

					# log
					if epoch == total_iters:
						logger.info('# Final Validation # FID: {:.4e}'.format(FID))
						logger.info('# Final Validation # F8: {:.4e} F1/8: {:.4e}'.format(f_8, f_1_8))
						logger.info('# Final Validation # Acc: {:.4e}'.format(score_cls_val))
						logger.info('# Final Validation # LVAL: {:.4e}'.format(lt_cls_val/len(val_loader)))
					else:
						logger.info('# Validation # FID: {:.4e}'.format(FID))
						logger.info('# Validation # F8: {:.4e} F1/8: {:.4e}'.format(f_8, f_1_8))
						logger.info('# Final Validation # Acc: {:.4e}'.format(score_cls_val))
						logger.info('# Final Validation # LVAL: {:.4e}'.format(lt_cls_val/len(val_loader)))

					# tensorboard logger
					if opt['use_tb_logger'] and 'debug' not in opt['name']:
						tb_logger.add_scalar('metrics/fid', FID, epoch)
						tb_logger.add_scalar('metrics/F8', f_8, epoch)
						tb_logger.add_scalar('metrics/F1_8', f_8, f_1_8)
						tb_logger.add_scalar('metrics/Acc', score_cls_val, epoch)
						tb_logger.add_scalar('metrics/LVAL', lt_cls_val/len(val_loader), epoch)

					del art_samples, true_samples, FID



		#### save models and training states
		if opt['logger']['save_checkpoint_freq'] and epoch % opt['logger']['save_checkpoint_freq'] == 0:
			if rank <= 0:
				logger.info('Saving models and training states.')
				model.save(epoch)
				model.save_training_state(epoch)



	if rank <= 0:
		logger.info('Saving the final model.')
		model.save('latest')
		logger.info('End of training.')
		tb_logger.close()


if __name__ == '__main__':
	main()
