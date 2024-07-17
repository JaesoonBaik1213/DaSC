#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import json
import torch
import random
import argparse
import numpy as np

def parse_arguments():
	# parse arguments
	parser = argparse.ArgumentParser(description='Imbalanced Example')
	parser.add_argument('--dataset', default='cifar10', type=str,
	                    help='dataset (cifar10 [default] or cifar100)')
	parser.add_argument('--corruption_prob', type=float, default=0.4,
	                    help='label noise')
	parser.add_argument('--corruption_type', '-ctype', type=str, default='flip',
	                    help='Type of corruption ("unif" or "flip" or "flip2").')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--num_classes', type=int, default=10)
	parser.add_argument('--num_meta', type=int, default=10,
	                    help='The number of meta data for each class.')
	parser.add_argument('--imb_factor', type=float, default=0.1)
	parser.add_argument('--meta-batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for metaing (default: 100)')
	parser.add_argument('--classifier-batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for classifier train (default: 100)')
	parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for testing (default: 100)')
	parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
	                    help='initial learning rate')
	parser.add_argument('--lr_decay', action='store_true')
	parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
	parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
	                    help='weight decay (default: 2e-4)') # from 5e-4 to 2e-4
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	parser.add_argument('--split', type=int, default=1000)
	parser.add_argument('--seed', type=int, default=25, metavar='S',
	                    help='random seed (default: 42)')
	parser.add_argument('--print-freq', '-p', default=50, type=int,
	                    help='print frequency (default: 10)')
	parser.add_argument('--model_name', default='baseline', type=str,
	                    help='use model version')
	parser.add_argument('--exp-str', default='', type=str)
	parser.add_argument('--inverse-imbalance', action='store_true',
	                    help='if use inverse-imbalance sampler')
	parser.add_argument('--tfrecord', action='store_true',
	                    help='if load tfrecord dataset')
	parser.add_argument('--embedding_dim', default=20, type=int)
	parser.add_argument('--other_param_index', default=1, type=int)
	parser.add_argument('--adjust_lr',action='store_true',
	                    help="if adjusts model lr")
	parser.add_argument('--with_index',action='store_true',
	                    help="if dataset get with index")
	parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
	                    help='start epoch to train')
	parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
	parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
	parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
	parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
	parser.add_argument('--num_epochs', default=100, type=int)
	parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
	parser.add_argument('-w', '--warm_up', default=30, type=int)
	parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
	parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
	parser.add_argument('--strong_augment', default='randaugment', choices=['default', 'randaugment', 'simaugment'])
	
	# DaCC
	parser.add_argument('--phi', default=1.005, type=float, help='parameter for dynamic threshold')
	parser.add_argument('--is_prototypical_selection', action='store_true')
	parser.add_argument('--feature_type', default='project', choices=['project', 'default'])
	parser.add_argument('--eval_temp', default=0.1, type=float)
	parser.add_argument('--proto_assign_type', default='DaCC', type=str, choices=['DaCC'])

	# Confidence-aware Contrastive Loss
	parser.add_argument('--conf_mask_threshold', default=0.9, type=float)
	parser.add_argument('--conf_mask_type', default='targets', choices=['targets'])
	
	# SBCL	
	parser.add_argument('--hidden_dim', default=128, type=int)
	parser.add_argument('--warmup_SBCL', action='store_true')
	parser.add_argument('--train_SBCL', action='store_true')
	parser.add_argument('--SBCL_lambda', default=0.5, type=float, help='coefficient for SBCL')
	parser.add_argument('--SBCL_temp', default=0.1, type=float, help='temperature parameter for SBCL')
	parser.add_argument('--SBCL_epochs', default=10, type=int)
	
	# MIDL
	parser.add_argument('--train_MIDL', action='store_true')
	parser.add_argument('--MIDL_lambda', default=0.3, type=float)
	parser.add_argument('--MIDL_temp', default=0.5, type=float)
	parser.add_argument('--MIDL_memory_size', default=1024, type=int)
	parser.add_argument('--MIDL_memory_type', default='mix', choices=['default', 'low_conf', 'mix', 'mix_low_conf'])

	args = parser.parse_args()
	print(args)
	return args


def set_seed(seed=None):
	# control randomness
	if seed:
		# random #
		random.seed(seed)  
		# hashseed #
		os.environ['PYTHONHASHSEED'] = str(seed)  
		# numpy #
		np.random.seed(seed)  
		# torch #
		torch.manual_seed(seed)  # torch cpu
		torch.cuda.manual_seed(seed)  # torch current gpu
		torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
		torch.backends.cudnn.benchmark = False  # ensure CUDA selects the same algorithm each time an application is run
		torch.backends.cudnn.deterministic = True  # Avoiding nondeterministic algorithms

def set_device(args):
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	return device

def load_config():
	with open("./settings/config.json") as json_file:
		config = json.load(json_file)
	return config
