#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from itertools import count
import os
from numpy import random
import torch
import platform
import numpy as np
from torch.autograd import Variable
import pdb
import copy

systype = platform.system()

def make_dir(root):
    from torch.utils.tensorboard import SummaryWriter
    try:
        original_umask = os.umask(0)
        os.makedirs(root, exist_ok=True)
        writer = SummaryWriter(root)
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(root):
            pass
        else:
            raise
    finally:
        os.umask(original_umask)

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def set_tensor(tensor_var, boolen, device):
	# print(tensor_var)
	tensor_var = tensor_var.to(device)
	# tensor_var = tensor_var.to(device,non_blocking=True)  
	#return Variable(tensor_var, requires_grad=boolen)
	tensor_var.requires_grad = boolen
	return tensor_var

def plot_tsne(X, y, legends, save_dir, file_name, epoch, num_classes=10):
	import matplotlib.pyplot as plt
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, init='pca', random_state=0)
	X_tsne = tsne.fit_transform(X)
	plt.figure(figsize=(10, 10))
	colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown'
	for label, c, in zip(range(num_classes), colors):
		idx = np.where(y == label)
		plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=c, label=legends[label], s=10)
	plt.legend(fontsize=17)
	plt.savefig(os.path.join(save_dir, file_name + '_epoch{}.png'.format(epoch+1)))
	plt.close()

def plot_tsne_prototypes(X, y, clean_y, prototypes, target_mask, total_outputs, current_mask, legends, legends_proto, save_dir, file_name, epoch, num_classes=10):
	# ['$D_{k}$', '$D_{k}$+AM', 'Ours($D$+AM)']
	import matplotlib.pyplot as plt
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, init='pca', random_state=0)
	clean_y, current_mask = clean_y.cpu().numpy(), current_mask.cpu().numpy()
	if len(prototypes[2]) == 2:
		prototypes_ours = prototypes[2][0]
		prototypes_ours2 = prototypes[2][1]
		X_tsne_temp = tsne.fit_transform(torch.cat([torch.tensor(X), prototypes[0], prototypes[1], prototypes_ours, prototypes_ours2], dim=0))
		X_tsne, prototypes_subset, prototypes_subsetAM, prototypes_ours, prototypes_ours2 = X_tsne_temp[:X.shape[0]], X_tsne_temp[-40:-30], X_tsne_temp[-30:-20], X_tsne_temp[-20:-10], X_tsne_temp[-10:]
	else:
		prototypes_ours = prototypes[2][0]
		prototypes_ours2 = None
		X_tsne_temp = tsne.fit_transform(torch.cat([torch.tensor(X), prototypes[0], prototypes[1], prototypes_ours], dim=0))
		X_tsne, prototypes_subset, prototypes_subsetAM, prototypes_ours = X_tsne_temp[:X.shape[0]], X_tsne_temp[-30:-20], X_tsne_temp[-20:-10], X_tsne_temp[-10:]
	
	
	plt.figure(figsize=(10, 10))
	# colors = [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9), (0.23, 0.79, 0.73), 'pink', 'lightyellow', 'peachpuff', 'violet', 'lightcoral', 'lightgray']
	colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'gray']
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y == label)
		plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=c, label=legends[label], s=10, alpha=0.6)

	for label, c in zip(range(num_classes-1), colors[:-1]):
		plt.scatter(prototypes_ours[label, 0], prototypes_ours[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
		if prototypes_ours2 is not None:
			plt.scatter(prototypes_ours2[label, 0], prototypes_ours2[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
		plt.scatter(prototypes_subset[label, 0], prototypes_subset[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
		plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=None, s=110, marker='^', edgecolors='k', linewidths=1.5)
	
	label = num_classes-1
	idx = np.where(y == label)
	c = colors[-1]
	plt.scatter(prototypes_subset[label, 0], prototypes_subset[label, 1], c=c, label=legends_proto[0], s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=legends_proto[1], s=110, marker='^', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_ours[label, 0], prototypes_ours[label, 1], c=c, label=legends_proto[2], s=90, marker='s', edgecolors='k', linewidths=1.5)
	if prototypes_ours2 is not None:
		plt.scatter(prototypes_ours2[label, 0], prototypes_ours2[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, file_name + '.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	# colors = [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9), (0.23, 0.79, 0.73), 'pink', 'lightyellow', 'peachpuff', 'violet', 'lightcoral', 'lightgray']
	colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'gray']
	for label, c in zip(range(num_classes), colors):
		idx = np.where(clean_y == label)
		plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=c, label=legends[label], s=10, alpha=0.6)

	for label, c in zip(range(num_classes-1), colors[:-1]):
		plt.scatter(prototypes_ours[label, 0], prototypes_ours[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
		if prototypes_ours2 is not None:
			plt.scatter(prototypes_ours2[label, 0], prototypes_ours2[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
		plt.scatter(prototypes_subset[label, 0], prototypes_subset[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
		plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=None, s=110, marker='^', edgecolors='k', linewidths=1.5)
	
	label = num_classes-1
	idx = np.where(clean_y == label)
	c = colors[-1]
	plt.scatter(prototypes_subset[label, 0], prototypes_subset[label, 1], c=c, label=legends_proto[0], s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=legends_proto[1], s=110, marker='^', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_ours[label, 0], prototypes_ours[label, 1], c=c, label=legends_proto[2], s=90, marker='s', edgecolors='k', linewidths=1.5)
	if prototypes_ours2 is not None:
		plt.scatter(prototypes_ours2[label, 0], prototypes_ours2[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, file_name + '_true_label.png'))
	plt.close()


	# plot activate samples for generating each prototype
	X_tsne_conf = X_tsne[current_mask]
	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(clean_y[current_mask] == label)
		plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c=c, label=legends[label], s=10, alpha=0.7)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subset_true_label.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y[current_mask] == label)
		plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c=c, label=legends[label], s=10, alpha=0.7)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subset_noise_label.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(clean_y[current_mask] == label)
		plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c=c, label=legends[label], s=10, alpha=0.7)
	for label, c in zip(range(num_classes-1), colors):
		plt.scatter(prototypes_subset[label, 0], prototypes_subset[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_subset[num_classes-1, 0], prototypes_subset[num_classes-1, 1], c=colors[-1], label='Prototype', s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subset_true_label_prototype.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y[current_mask] == label)
		plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c=c, label=legends[label], s=10, alpha=0.7)
	for label, c in zip(range(num_classes-1), colors):
		plt.scatter(prototypes_subset[label, 0], prototypes_subset[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_subset[num_classes-1, 0], prototypes_subset[num_classes-1, 1], c=colors[-1], label='Prototype', s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subset_noise_label_prototype.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(clean_y[current_mask] == label)
		# plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[current_mask][idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne_conf[idx][idx_conf, 0], X_tsne_conf[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subsetAM_true_label.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y[current_mask] == label)
		# plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[current_mask][idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne_conf[idx][idx_conf, 0], X_tsne_conf[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subsetAM_noise_label.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(clean_y[current_mask] == label)
		# plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[current_mask][idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne_conf[idx][idx_conf, 0], X_tsne_conf[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
		# plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	for label, c in zip(range(num_classes-1), colors):
		plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_subsetAM[num_classes-1, 0], prototypes_subsetAM[num_classes-1, 1], c=colors[-1], label='Prototype', s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subsetAM_true_label_prototype.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y[current_mask] == label)
		# plt.scatter(X_tsne_conf[idx, 0], X_tsne_conf[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[current_mask][idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne_conf[idx][idx_conf, 0], X_tsne_conf[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
		# plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	for label, c in zip(range(num_classes-1), colors):
		plt.scatter(prototypes_subsetAM[label, 0], prototypes_subsetAM[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_subsetAM[num_classes-1, 0], prototypes_subsetAM[num_classes-1, 1], c=colors[-1], label='Prototype', s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_subsetAM_noise_label_prototype.png'))
	plt.close()
	
	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(clean_y == label)
		# plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne[idx][idx_conf, 0], X_tsne[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_AM_true_label.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y == label)
		# plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne[idx][idx_conf, 0], X_tsne[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_AM_noise_label.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(clean_y == label)
		# plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne[idx][idx_conf, 0], X_tsne[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
	for label, c in zip(range(num_classes-1), colors):
		plt.scatter(prototypes_ours[label, 0], prototypes_ours[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_ours[num_classes-1, 0], prototypes_ours[num_classes-1, 1], c=colors[-1], label='Prototype', s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_AM_true_label_prototype.png'))
	plt.close()

	plt.figure(figsize=(10, 10))
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y == label)
		# plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c='k', label=None, s=10, alpha=0.90)
		idx_conf = torch.max(total_outputs[idx], dim=1)[0]>0.5
		idx_conf = idx_conf.cpu().numpy()
		plt.scatter(X_tsne[idx][idx_conf, 0], X_tsne[idx][idx_conf, 1], c=c, label=legends[label], s=10, alpha=0.7)
	for label, c in zip(range(num_classes-1), colors):
		plt.scatter(prototypes_ours[label, 0], prototypes_ours[label, 1], c=c, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.scatter(prototypes_ours[num_classes-1, 0], prototypes_ours[num_classes-1, 1], c=colors[-1], label='Prototype', s=200, marker='*', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, 'activate_samples_AM_noise_label_prototype.png'))
	plt.close()


def plot_tsne_nt_prototypes(X, y, prototypes, legends, legends_proto, save_dir, file_name, epoch, num_classes=10):
	import matplotlib.pyplot as plt
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, init='pca', random_state=0)
	X_tsne_temp = tsne.fit_transform(torch.cat([torch.tensor(X), prototypes[0], prototypes[1], prototypes[2], prototypes[3]], dim=0))
	# X_tsne, prototype_soft, prototype_hard, prototype, prototype_1 = X_tsne_temp[:X.shape[0]], X_tsne_temp[-40:-30], X_tsne_temp[-30:-20], X_tsne_temp[-20:-10], X_tsne_temp[-10:]
	X_tsne, prototype_soft, prototype_hard, prototype, prototype_1, prototype_2 = X_tsne_temp[:X.shape[0]], X_tsne_temp[-50:-40], X_tsne_temp[-40:-30], X_tsne_temp[-30:-20], X_tsne_temp[-20:-10], X_tsne_temp[-10:]
	plt.figure(figsize=(10, 10))
	# colors = [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9), (0.23, 0.79, 0.73), 'pink', 'lightyellow', 'peachpuff', 'violet', 'lightcoral', 'lightgray']
	colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'gray']
	for label, c in zip(range(num_classes), colors):
		idx = np.where(y == label)
		plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=c, label=legends[label], s=10, alpha=0.6)

	for label, c_p in zip(range(num_classes-1), colors[:-1]):
		plt.scatter(prototype[label, 0], prototype[label, 1], c=c_p, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
		plt.scatter(prototype_1[label, 0], prototype_1[label, 1], c=c_p, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
		plt.scatter(prototype_2[label, 0], prototype_2[label, 1], c=c_p, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
		# plt.scatter(prototype_soft[label, 0], prototype_soft[label, 1], c=c_p, label=None, s=200, marker='*', edgecolors='k', linewidths=1.5)
		# plt.scatter(prototype_hard[label, 0], prototype_hard[label, 1], c=c_p, label=None, s=110, marker='^', edgecolors='k', linewidths=1.5)
		
		
	
	label = num_classes-1
	idx = np.where(y == label)
	c = colors[-1]
	plt.scatter(prototype[label, 0], prototype[label, 1], c=c, label=legends_proto[2], s=90, marker='s', edgecolors='k', linewidths=1.5)
	plt.scatter(prototype_1[label, 0], prototype_1[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
	plt.scatter(prototype_2[label, 0], prototype_2[label, 1], c=c, label=None, s=90, marker='s', edgecolors='k', linewidths=1.5)
	# plt.scatter(prototype_soft[label, 0], prototype_soft[label, 1], c=c, label=legends_proto[0], s=200, marker='*', edgecolors='k', linewidths=1.5)
	# plt.scatter(prototype_hard[label, 0], prototype_hard[label, 1], c=c, label=legends_proto[1], s=110, marker='^', edgecolors='k', linewidths=1.5)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(save_dir, file_name + '.png'))
	plt.close()


def plot_confidence_avg_matrix(labels, outputs, save_dir, file_name, epoch, num_classes=10):
	import matplotlib.pyplot as plt
	conf_avg = torch.zeros(num_classes, num_classes)
	for i in range(num_classes):
		matching_idx = np.where(labels == i)[0]
		avg_confidence = torch.mean(outputs[matching_idx, :], dim=0)
		conf_avg[i, :] = avg_confidence
	plt.figure(figsize=(10, 10))
	plt.imshow(conf_avg, cmap='coolwarm', interpolation='nearest')
	plt.ylabel('True label', fontsize=15)
	plt.xlabel('Predicted label', fontsize=15)
	plt.colorbar()
	plt.savefig(os.path.join(save_dir, file_name + '.png'))
	plt.close()

def plot_confusion_matrix(labels, outputs, save_dir, file_name, epoch, num_classes=10):
	import matplotlib.pyplot as plt
	from sklearn.metrics import confusion_matrix
	conf = confusion_matrix(labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), labels=range(num_classes))
	plt.figure(figsize=(10, 10))
	plt.imshow(conf, cmap='coolwarm', interpolation='nearest')
	plt.ylabel('True label', fontsize=15)
	plt.xlabel('Predicted label', fontsize=15)
	plt.colorbar()
	plt.savefig(os.path.join(save_dir, file_name + '.png'))
	plt.close()