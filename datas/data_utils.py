from select import select
from matplotlib.pyplot import axis
import os
import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageFilter
from sklearn.preprocessing import LabelEncoder
from typing import Any, Callable, Optional, Tuple
from randaugment import randSVHNPolicy, randCIFARPolicy, randImageNetPolicy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import random
import numpy as np
import copy
import re
import requests
import io

from torchnet.meter import AUCMeter
import pdb
            
def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class noPolicy(object):
    def __call__(self, img):
        return img

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class simCIFARPolicy(object):
    def __init__(self, size=32):
        s = 2
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.policy = transforms.Compose([transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size))])
    
    def __call__(self, img):
        return self.policy(img)

def get_augment(dataset, aug_method):
    if dataset == 'svhn':
        # if aug_method == 'autoaugment':
        #     return autoSVHNPolicy
        if aug_method == 'randaugment':
            return randSVHNPolicy
        elif aug_method == 'default':
            return noPolicy
        else:
            raise ValueError
    if dataset in ['cifar', 'cifar10', 'cifar100']:
        # if aug_method == 'autoaugment':
        #     return autoCIFARPolicy
        if aug_method == 'randaugment':
            return randCIFARPolicy
        elif aug_method == 'simaugment':
            return simCIFARPolicy
        elif aug_method == 'default':
            return noPolicy
        else:
            raise ValueError
    if dataset in ['imagenetLT', 'iNaturalist18', 'placesLT', 'mini-imagenet', 'animal10-NLT', 'food101-NLT', 'webvision']:
        # if aug_method == 'augtoaugment':
        #     return autoImageNetPolicy
        if aug_method == 'randaugment':
            return randImageNetPolicy
        elif aug_method == 'default':
            return noPolicy
        else:
            raise ValueError

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, train_imgs, train_labels, val_labels, transform, mode, val_imgs=[], num_class=100, transform_w=None, select_num=0, select_index=[], pred=[], probability=[], log='', args=None, device=None):
        self.root = root_dir
        self.transform = transform
        self.transform_w = transform_w
        self.mode = mode
        self.probability = np.array(probability)
        # self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.real_img_num_list = [0] * num_class
        self.args = args
        self.device = device
        self.noise_label = train_labels
        if mode == 'all':
            self.train_imgs = train_imgs
            self.noise_label = torch.zeros(len(train_imgs)).long()
            self.noisy_img_num_list = [0] * num_class
            for i in range(len(train_imgs)):
                train_img = train_imgs[i]
                self.noisy_img_num_list[train_labels[train_img]] += 1
                self.noise_label[i] = train_labels[train_img]
        elif self.mode == "labeled":
            self.train_imgs = np.array(train_imgs)[select_index].tolist()             
            print("select num :",len(select_index),end=" ")
        elif self.mode == "unlabeled":   
            unlabel_index = np.delete(np.arange(len(train_imgs)), select_index)
            self.train_imgs = np.array(train_imgs)[unlabel_index].tolist()           
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs))) 
        # elif self.mode == "select":
        #     self.train_imgs = [i for i in select_index]   
        elif mode=='test':
            self.val_imgs = val_imgs
        else:
            data_list = [i for i in train_imgs if self.train_labels[i] == select_num]
            if self.mode == "single" :
                self.train_imgs = [i for i in data_list] 

    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            clean_target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            img3 = self.transform_w(image)
            return img1, img2, img3, clean_target, target, prob
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            clean_target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            img3 = self.transform_w(image)
            return img1, img2, img3, clean_target
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            clean_target = self.train_labels[img_path]
            img = self.transform(image)
            img2 = self.transform_w(image)
            img3 = self.transform_w(image)
            return img, img2, img3, clean_target, target, index
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root+img_path).convert('RGB')
            img = self.transform(image)
            return img, target
    
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    
        
class cifar_dataset(Dataset): 
    def __init__(self, train_data,noise_label,data_index,dataset, root_dir, transform, mode, transform_w=None, class_num=0, select_index=[], probability=[],train_label=None,log=''):
        self.transform = transform
        self.transform_w = transform_w
        self.mode = mode  
        self.data_index = data_index
        self.probability = np.array(probability)
        self.train_label = train_label
        self.precision = []
        self.recall = []
        self.auc = []
        if self.mode=='test':
            if dataset=='cifar10':
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:        
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
  
            elif self.mode == "labeled":
                self.train_data = train_data[select_index]
                self.noise_label = noise_label[select_index]  
                try:
                    clean = (np.array(noise_label) == np.array(train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(self.probability, clean)
                    auc,_,_ = auc_meter.value()
                    cls_id_list, cls_num_list = np.unique(noise_label, return_counts=True)
                    if dataset == 'cifar10':
                        many_shot = cls_id_list < 2
                        few_shot = cls_id_list >= 7
                        medium_shot = ~(many_shot | few_shot)
                    elif dataset == 'cifar100':
                        many_shot = cls_num_list > 100
                        few_shot = cls_num_list < 20
                        medium_shot = ~(many_shot | few_shot)
                    if dataset in ['cifar10', 'cifar100']:
                        many_shot_idx = many_shot[noise_label]
                        medium_shot_idx = medium_shot[noise_label]
                        few_shot_idx = few_shot[noise_label]
                        auc_meter.reset()
                        auc_meter.add(probability[many_shot_idx], clean[many_shot_idx])
                        auc_many, _, _ = auc_meter.value()
                        auc_meter.reset()
                        auc_meter.add(probability[medium_shot_idx], clean[medium_shot_idx])
                        auc_medium, _, _ = auc_meter.value()
                        auc_meter.reset()
                        auc_meter.add(probability[few_shot_idx], clean[few_shot_idx])
                        auc_few, _, _ = auc_meter.value()
                        log.write('Current AUC (many,medium,few):{:0.4f}({:0.3f},{:0.3f},{:0.3f})  Numer of labeled samples:{:d}'.format(auc, auc_many, auc_medium, auc_few, len(select_index)))
                        print("AUC val (many,medium,few): {:0.4f}({:0.3f}/{:0.3f}/{:0.3f})".format(auc, auc_many, auc_medium, auc_few))
                        log.write(' | Shot P&R: ')
                        pred = self.probability > 0.5
                        for shot in (many_shot, medium_shot, few_shot):
                            p_idxs = shot[noise_label] & pred
                            r_idxs = shot[noise_label] & clean
                            p = (pred[p_idxs] == clean[p_idxs]).mean() if p_idxs.any() else 0.0
                            r = (pred[r_idxs] == clean[r_idxs]).mean()
                            log.write('(%.4f %.4f) ' % (p, r))
                        log.write('\n')
                        log.flush()
                except:
                    pass
                   
            elif self.mode == "unlabeled":                                         
                self.train_data = np.delete(train_data,select_index ,axis=0)
                self.noise_label = np.delete(noise_label,select_index ,axis=0) 
            elif self.mode == 'select':
                self.train_data = train_data[select_index]
                self.noise_label = noise_label[select_index]     
            else: 
                data_list = [i for i,label in enumerate(noise_label) if label == class_num ]  
                if self.mode == 'single':
                    self.train_data = train_data[data_list]
                    self.noise_label = noise_label[data_list]  
                    self.data_index = data_index[data_list]
                elif self.mode == 'other': 
                    self.train_data = np.delete(train_data,data_list ,axis=0)
                    self.noise_label = np.delete(noise_label,data_list ,axis=0)  
            # print("%s data has a size of %d"%(self.mode, len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            # print(self.probability)
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            clean_target = self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img)
            img3 = self.transform_w(img)
            return img1, img2, img3, clean_target, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            clean_target = self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img)
            img3 = self.transform_w(img)
            return img1, img2, img3, clean_target
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif self.mode=='all': # mode == 'all' is used for warmup, eval_train
            img, target = self.train_data[index], self.noise_label[index]
            clean_target = self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) # test augmentation
            img2 = self.transform_w(img) # weak augmentation
            img3 = self.transform_w(img) # weak augmentation
            return img1, img2, img3, clean_target, target, self.data_index[index]     
        else:
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, self.data_index[index]     
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         

class imagenet_dataloader():
    def __init__(self, dataset, corrupt_prob, imb_factor, noise_mode, batch_size, num_workers, root_dir, log, args=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.corrupt_prob = corrupt_prob
        self.imb_factor = imb_factor
        self.num_class = 100
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_mode + '_noise_nl_' + str(corrupt_prob)
        self.strong_augment = args.strong_augment
        self.clean_labels = {}
        self.noise_labels = {}
        self.train_labels = {}
        self.val_labels = {}
        self.train_imgs = []
        self.noisy_imgs = []
        self.val_imgs = []
        self.args = args
        self.transform_train = transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4,contrast = 0.4 ,saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    get_augment('cifar', self.strong_augment)(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        self.transform_warmup = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4,contrast = 0.4 ,saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        
        control_label_path = self.root_dir + 'split'
        with open('%s/blue_noise_nl_0.0'%control_label_path,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = self.root_dir + 'all_images'+ '/' +entry[0]
                self.train_imgs.append(img_path)
                self.clean_labels[img_path] = int(entry[1]) 
        with open('%s/red_noise_nl_%.1f'%(control_label_path,0.8),'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                if  re.match('^n.*',entry[0])  is None:   
                    img_path = self.root_dir + 'all_images'+ '/' +entry[0]
                    self.noisy_imgs.append(img_path)
                    self.noise_labels[img_path] = int(entry[1])
        random.shuffle(self.noisy_imgs)
        with open('%s/clean_validation'%control_label_path,'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                # entry = l.split() 
                # img_path = self.root_dir + '/val'+ '/' +entry[0]
                # self.val_imgs.append(img_path)
                # self.val_labels[img_path] = int(entry[1])
                img, target = line.split()
                target = int(target)
                val_path = 'val/' + str(target) + '/'
                self.val_imgs.append(val_path+img)
                self.val_labels[val_path+img]=target

        img_num_list = self.get_img_num_per_cls(len(self.train_imgs) / self.num_class, self.num_class, imb_factor, 0)

        self.train_imgs =self.sample_dataset(self.train_imgs ,self.clean_labels,img_num_list,self.num_class,'select')
        self.data_num = sum(img_num_list)
        # select_noisy_num = int(self.data_num * corrupt_prob/(0.8-corrupt_prob))
        select_noisy_num = int(self.data_num / (1 - corrupt_prob) - self.data_num)
        self.train_imgs.extend(self.noisy_imgs[:select_noisy_num])
        self.train_labels.update(self.clean_labels)
        self.train_labels.update(self.noise_labels)
        self.data_num = len(self.train_imgs)
        self.cal_label_distribution(self.train_imgs,self.train_labels)

    def sample_dataset(self, train_data, train_label, img_num_list, num_classes, kind):
        """
        Args:
            dataset
            img_num_list
            num_classes
            kind
        Returns:

        
        """
        data_list = {}
        for j in range(num_classes):
            data_list[j] = [i for i in train_data if train_label[i] == j]

        idx_to_del = []
        for cls_idx, img_id_list in data_list.items():
            '''
            cls_idx : class index
            img_id_list:sample global index list
            data_list:{'cls_idx':[img_id_list],}
            '''
            np.random.shuffle(img_id_list)
            # print(img_id_list)
            img_num = img_num_list[int(cls_idx)]
            # print(img_num)
            if kind=='delete':
                idx_to_del.extend(img_id_list[:img_num])
            else:
                idx_to_del.extend(img_id_list[img_num:])
        train_data_ = list(set(train_data).difference(set(idx_to_del))) 

        return train_data_

    def cal_label_distribution(self, train_data,train_label):
        class_num_list = []
        data_list= {}
        for j in range(100):
            data_list[j] = [i for i in train_data if train_label[i] == j]
            class_num_list.append(len(data_list[j]))
        print(class_num_list)

    def get_img_num_per_cls(self, img_num,cls_num,imb_factor=None,num_meta=None):
        """
        Get a list of image numbers for each class, given cifar version
        Num of imgs follows emponential distribution
        img max: 5000 / 500 * e^(-lambda * 0);
        img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
        exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
        args:
        cifar_version: str, '10', '100', '20'
        imb_factor: float, imbalance factor: img_min/img_max,
            None if geting default cifar data number
        output:
        img_num_per_cls: a list of number of images per class
        """
        img_max = img_num
        if imb_factor is None:
            return [img_max] * cls_num
        img_num_per_cls = []
        imbalance_ratios = get_imbalance_ratios(imb_factor, cls_num)
        for cls_idx in range(cls_num):
            ratio = imbalance_ratios[cls_idx]
            num = img_max * ratio
            img_num_per_cls.append(int(num))
        return img_num_per_cls
    
    def run(self,mode,class_num=0,select_index=[],prob=[]):
        if mode=='warmup':
            all_dataset = imagenet_dataset(root_dir=self.root_dir, train_imgs=self.train_imgs, train_labels=self.train_labels,
                                           val_labels=self.val_labels, transform=self.transform_warmup, transform_w=self.transform_warmup, mode="all")
            self.noisy_img_num_list = all_dataset.noisy_img_num_list
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader
        
        elif mode=='train':
            labeled_dataset = imagenet_dataset(root_dir=self.root_dir, train_imgs=self.train_imgs, train_labels=self.train_labels,
                                               val_labels=self.val_labels, transform=self.transform_train, transform_w=self.transform_warmup, mode="labeled", select_index=select_index, probability=prob,log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = imagenet_dataset(root_dir=self.root_dir,train_imgs=self.train_imgs, train_labels=self.train_labels,
                                                 val_labels=self.val_labels, transform=self.transform_train, transform_w=self.transform_warmup, mode="unlabeled", select_index=select_index)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='test':
            test_dataset = imagenet_dataset(root_dir=self.root_dir, train_imgs=self.train_imgs, train_labels=self.train_labels, val_imgs=self.val_imgs,
                                            val_labels=self.val_labels, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size*10,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode == 'eval_train':
            all_dataset = imagenet_dataset(root_dir=self.root_dir, train_imgs=self.train_imgs, train_labels=self.train_labels,
                                           val_labels=self.val_labels, transform=self.transform_test, transform_w=self.transform_warmup, mode="all")
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*10,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader


class cifar_dataloader():  
    def __init__(self, dataset,corrupt_prob,imb_factor,noise_mode, batch_size, num_workers, root_dir, log, noise_path=None, args=None):
        self.dataset = dataset
        self.corrupt_prob = corrupt_prob
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_path = noise_path
        self.root_dir = root_dir
        self.log = log
        self.strong_augment = args.strong_augment
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    get_augment('cifar', self.strong_augment)(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
            self.transform_warmup = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    get_augment('cifar', self.strong_augment)(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_warmup = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
        train_data=[]
        train_label=[]
        
        if dataset == 'cifar10':
            num_classes = 10
            for n in range(1,6):
                dpath = '%s/data_batch_%d'%(root_dir,n)
                data_dic = unpickle(dpath)
                train_data.append(data_dic['data'])
                train_label = train_label+data_dic['labels']
            train_data = np.concatenate(train_data)
        elif dataset == 'cifar100':
            num_classes = 100 
            train_dic = unpickle('%s/train'%root_dir)
            train_data = train_dic['data']
            train_label = train_dic['fine_labels']
        
        if  noise_mode == 'human':
            if dataset == 'cifar10':
                noise_label = torch.load(self.noise_path)['worse_label']
            elif dataset == 'cifar100':
                noise_label = torch.load(self.noise_path)['noisy_label']
        else:
            noise_label = None
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))
        
        # imbalance 
        img_num_list = get_img_num_per_cls_1(dataset, num_classes, imb_factor, 0)
        self.img_num_list = img_num_list
        print("real_img_num_list:", img_num_list)
        print("real_img_num_list-sum:", sum(img_num_list))
        train_data ,train_label, noise_label = sample_dataset_1(train_data, train_label, noise_label, img_num_list, num_classes, 'select')
        self.data_num = sum(img_num_list)
        
        if noise_label is None:
            # noisy
            if noise_mode == 'unif':
                noisy_transaction_matrix_real = uniform_mix_c_1(self.corrupt_prob, num_classes)
            elif noise_mode == 'flip':
                noisy_transaction_matrix_real = flip_labels_c_1(self.corrupt_prob, num_classes)
            noise_label = copy.deepcopy(train_label)
            
            for i in range(sum(img_num_list)):
                noise_label[i] = np.random.choice(num_classes, p=noisy_transaction_matrix_real[train_label[i]])
            print("noisy transation matrix:",noisy_transaction_matrix_real)
        else: # cifar10N, cifar100N
            print("Real-world noisy labels are used.")
        self.train_data = train_data
        self.train_label = train_label
        self.noise_label = noise_label
        self.noisy_img_num_list = [len([j for j in noise_label if j==i]) for i in range(num_classes)]
        self.data_index  = np.array([i for i in range(len(noise_label))])
    
    def plot_confusion(self, ):
        pdb.set_trace()

        print(1)

    def run(self,mode,class_num=0,select_index=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_warmup, transform_w=self.transform_warmup, train_label=self.train_label, mode="all")
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, transform_w=self.transform_warmup, mode="labeled", select_index=select_index, probability=prob, train_label=self.train_label,log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_train, transform_w=self.transform_warmup, train_label=self.train_label, mode="unlabeled", select_index=select_index)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        elif mode=='test':
            test_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        elif mode == 'eval_train':
            all_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test, transform_w=self.transform_warmup, train_label=self.train_label, mode="all")
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        else:
            eval_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test, mode=mode,class_num=class_num,select_index = select_index)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        


def sample_dataset_1(train_data, train_label, noise_label, img_num_list, num_classes, kind):
    """
    Args:
        dataset
        img_num_list
        num_classes
        kind
    Returns:

    
    """

    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(train_label) if label == j]

    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        '''
        cls_idx : class index
        img_id_list:sample global index list
        data_list:{'cls_idx':[img_id_list],}
        '''
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        if kind=='delete':
            idx_to_del.extend(img_id_list[:img_num])
        else:
            idx_to_del.extend(img_id_list[img_num:])

    # new_dataset = copy.deepcopy(dataset)
    train_label = np.delete(train_label, idx_to_del, axis=0)
    train_data = np.delete(train_data, idx_to_del, axis=0)
    if noise_label is not None:
        noise_label = np.delete(noise_label, idx_to_del, axis=0)
    # data_index = np.delete(data_index, idx_to_del, axis=0)
    return train_data, train_label, noise_label



def get_imbalance_ratios_1(imb_factor, cls_num):
    imbalance_ratios = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        imbalance_ratios.append(ratio)
    return imbalance_ratios

def get_img_num_per_cls_1(dataset,cls_num,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset in ['cifar10', 'cifar10n']:
        img_max = 5000-num_meta

    if dataset in ['cifar100', 'cifar100n']:
        img_max = 500-num_meta

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    imbalance_ratios = get_imbalance_ratios_1(imb_factor, cls_num)
    for cls_idx in range(cls_num):
        ratio = imbalance_ratios[cls_idx]
        num = img_max * ratio
        img_num_per_cls.append(int(num))
    return img_num_per_cls

def uniform_mix_c_1(mixing_ratio, num_classes):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_c_1(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    torch.save(C, 'noisy_transaction_matrix_real.pt')
    return C


def get_transform(args):
    if args.dataset == 'cifar10':
        # cifar 10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform_train = transforms.Compose([

    transforms.Pad(padding=4, fill=0, padding_mode="reflect"),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    transform_for_contrast = TwoCropTransform(transforms.Compose([

    transforms.Pad(padding=4, fill=0, padding_mode="reflect"),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
    transforms.RandomGrayscale(p=0.2),

    
    transforms.ToTensor(),
    normalize
    ]))

    return transform_train,transform_test,transform_for_contrast



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



class CIFAR10_With_Index(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        return super().__getitem__(index), index
    def __len__(self) -> int:
        return super().__len__()

class CIFAR100_With_Index(torchvision.datasets.CIFAR100):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index), index
    def __len__(self) -> int:
        return super().__len__()
        
def build_dataset(args,transform_type = 'normal'):
    transform_train,transform_test,transform_for_contrast = get_transform(args)
    if transform_type == 'normal':
        transforms = transform_train
    else:
        transforms = transform_for_contrast

    if args.dataset == 'cifar10':
        if args.with_index:
            train_dataset = CIFAR10_With_Index(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = CIFAR10_With_Index('./datas/data', train=False, transform=transform_test)
        else:
            train_dataset = torchvision.datasets.CIFAR10(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = torchvision.datasets.CIFAR10('./datas/data', train=False, transform=transform_test)
        img_num_list = [args.num_meta] * args.num_classes
        num_classes = 10
    elif args.dataset == 'cifar100':
        if args.with_index:
            train_dataset = CIFAR100_With_Index(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = CIFAR100_With_Index('./datas/data', train=False, transform=transform_test)
        else:
            train_dataset = torchvision.datasets.CIFAR100(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = torchvision.datasets.CIFAR100('./datas/data', train=False, transform=transform_test)
        img_num_list = [args.num_meta] * args.num_classes
        num_classes = 100
    # if num_classes > args.num_classes:
    #     class_list = np.random.randint(0,num_classes,size=args.num_classes)
    #     train_dataset = get_sub_class_dataset(train_dataset,class_list,num_classes,reset_index=True)
    #     test_dataset = get_sub_class_dataset(test_dataset,class_list,num_classes,reset_index=True)
    elif num_classes < args.num_classes:
        print("args.num_classes is larger than dataset class num")
        exit(1)

    if args.dataset in ['cifar10', 'cifar100', 'cifar10n', 'cifar100n']:
        meta_dataset = sample_dataset(train_dataset, img_num_list, args.num_classes, 'select')
        np.random.seed(args.seed)
        train_dataset=sample_dataset(train_dataset,img_num_list,args.num_classes,'delete')
    else:
        pass

    return meta_dataset,train_dataset,test_dataset

def get_imbalance_dataset(args,dataset):
    img_num_list = get_img_num_per_cls(args.dataset, args.num_classes, args.imb_factor, args.num_meta)
    print("img_num_list:", img_num_list)
    print("img_num_list-sum:", sum(img_num_list))
    imbalance_dataset=sample_dataset(dataset,img_num_list,args.num_classes,'select')
    return imbalance_dataset


def get_img_num_per_cls(dataset,cls_num,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = 5000-num_meta

    if dataset == 'cifar100':
        img_max = 500-num_meta

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    imbalance_ratios = get_imbalance_ratios(imb_factor, cls_num)
    for cls_idx in range(cls_num):
        ratio = imbalance_ratios[cls_idx]
        num = img_max * ratio
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def sample_dataset(dataset, img_num_list, num_classes, kind):
    """
    Args:
        dataset
        img_num_list
        num_classes
        kind
    Returns:

    
    """
    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(dataset.targets) if label == j]

    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        '''
        cls_idx : class index
        img_id_list:sample global index list
        data_list:{'cls_idx':[img_id_list],}
        '''
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        if kind=='delete':
            idx_to_del.extend(img_id_list[:img_num])
        else:
            idx_to_del.extend(img_id_list[img_num:])

    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = np.delete(dataset.targets, idx_to_del, axis=0)
    new_dataset.data = np.delete(dataset.data, idx_to_del, axis=0)

    return new_dataset


def sample_dataset_with_index(dataset, data_index):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = dataset.target[data_index]
    new_dataset.data = dataset.data[data_index]
    return new_dataset

def sample_dataset_with_indexs_trans(dataset, select_index,select_prob,transform_label,transform_unlabel):
    new_dataset_1 = copy.deepcopy(dataset)    
    new_dataset_2 = copy.deepcopy(dataset)
    new_dataset_1.targets = dataset.target[select_index]
    new_dataset_1.data = dataset.data[select_index]
    new_dataset_1.data_index = dataset.data_index[select_index]
    new_dataset_1.transform = transform_label
    new_dataset_1.mode = 'label'
    new_dataset_1.prob = select_prob
    
    new_dataset_2.targets = np.delete(dataset.targets, select_index, axis=0)
    new_dataset_2.data = np.delete(dataset.data, select_index, axis=0)
    new_dataset_2.data_index = np.delete(dataset.data_index, select_index, axis=0)
    new_dataset_2.transform = transform_unlabel
    new_dataset_2.mode = 'unlabel'
    
    return new_dataset_1,new_dataset_2

def get_sub_class_dataset(dataset, class_list, num_classes,reset_index=False):
    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(dataset.targets) if label == j]
    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        if cls_idx not in class_list:
            idx_to_del.extend(img_id_list)
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = np.delete(dataset.targets, idx_to_del, axis=0)
    new_dataset.data = np.delete(dataset.data, idx_to_del, axis=0)
    if reset_index:
        # convert discrete label to continuous label
        label_convertor = LabelEncoder()
        label_convertor.fit(class_list)
        new_dataset.targets = label_convertor.transform(new_dataset.targets)
    return new_dataset

def get_single_class_dataset(dataset,class_index):
    data_list = [i for i, label in enumerate(dataset.targets) if label == class_index]
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = dataset.targets[data_list]
    new_dataset.data = dataset.data[data_list]
    new_dataset.data_index = dataset.data_index[data_list]
    return new_dataset

def get_other_class_dataset(dataset,class_index):
    data_list = [i for i, label in enumerate(dataset.targets) if label == class_index]
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = np.delete(dataset.targets, data_list, axis=0)
    new_dataset.data = np.delete(dataset.data, data_list, axis=0)
    new_dataset.data_index = np.delete(dataset.data_index, data_list, axis=0)
    return new_dataset

def get_sub_dataset(dataset,sub_index):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = dataset.targets[sub_index]
    new_dataset.data = dataset.data[sub_index]
    return new_dataset


def get_only_noisy_dataset(clean_dataset, with_noisy_dataset):
    idx_to_del = []
    # print(clean_dataset.targets)
    for idx in range(0,len(clean_dataset)):
        if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx]:
            idx_to_del.append(idx)
    new_dataset = copy.deepcopy(with_noisy_dataset)
    new_dataset.targets = np.delete(with_noisy_dataset.targets, idx_to_del, axis=0)
    new_dataset.data = np.delete(with_noisy_dataset.data, idx_to_del, axis=0)
    return new_dataset

def get_only_clean_dataset(clean_dataset, with_noisy_dataset):
    idx_to_del = []
    # print(clean_dataset.targets)
    for idx in range(0,len(clean_dataset)):
        if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx]:
            idx_to_del.append(idx)
    new_dataset = copy.deepcopy(with_noisy_dataset)
    new_dataset.targets = with_noisy_dataset.targets[idx_to_del]
    new_dataset.data = with_noisy_dataset.data[idx_to_del]
    return new_dataset

def get_sub_clean_or_noisy_dataset(clean_dataset, with_noisy_dataset,class_index):

    idx_to_del_1 = []
    idx_to_del_2 = []

    for idx in range(0,len(clean_dataset)):
        if clean_dataset.targets[idx]==class_index:
            idx_to_del_1.append(idx)
            if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx] :
                idx_to_del_2.append(idx)
                idx_to_del_1.remove(idx)
    clean_new_dataset = copy.deepcopy(clean_dataset)
    noisy_new_dataset = copy.deepcopy(with_noisy_dataset)
    clean_new_dataset.targets = clean_dataset.targets[idx_to_del_2]
    clean_new_dataset.data = clean_dataset.data[idx_to_del_2]
    noisy_new_dataset.targets = with_noisy_dataset.targets[idx_to_del_1]
    noisy_new_dataset.data = with_noisy_dataset.data[idx_to_del_1]
    return clean_new_dataset,noisy_new_dataset

def get_sub_clean_or_noisy_dataset_2(clean_dataset, with_noisy_dataset,class_index):

    idx_to_del_1 = []
    idx_to_del_2 = []

    for idx in range(0,len(clean_dataset)):
        if with_noisy_dataset.targets[idx]==class_index:
            idx_to_del_1.append(idx)
            if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx] :
                idx_to_del_2.append(idx)
                idx_to_del_1.remove(idx)
    clean_new_dataset = copy.deepcopy(clean_dataset)
    noisy_new_dataset = copy.deepcopy(with_noisy_dataset)
    clean_new_dataset.targets = clean_dataset.targets[idx_to_del_2]
    clean_new_dataset.data = clean_dataset.data[idx_to_del_2]
    # noisy_new_dataset.targets = clean_dataset.targets[idx_to_del_1]
    noisy_new_dataset.targets = with_noisy_dataset.targets[idx_to_del_1]
    noisy_new_dataset.data = with_noisy_dataset.data[idx_to_del_1]
    return clean_new_dataset,noisy_new_dataset

def get_sub_clean_or_noisy_dataset_3(clean_dataset, with_noisy_dataset,class_index):

    idx_to_del_1 = []

    for idx in range(0,len(clean_dataset)):
        if with_noisy_dataset.targets[idx]==class_index:
            idx_to_del_1.append(idx)
    new_dataset = copy.deepcopy(clean_dataset)
    new_dataset.targets = clean_dataset.targets[idx_to_del_1]
    new_dataset.data = clean_dataset.data[idx_to_del_1]
    return new_dataset

def get_imbalance_ratios(imb_factor, cls_num):
    imbalance_ratios = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        imbalance_ratios.append(ratio)
    return imbalance_ratios


def get_inverse_imbalance_sampler(args, data):
    # init weights list
    weights = torch.zeros(len(data), dtype=torch.long)

    # get imbalance ratio
    sample_probability = get_imbalance_ratios(imb_factor=args.imb_factor, cls_num=args.num_classes)
    # get reverse
    sample_probability.sort(reverse=False)
    torch_sample_probability = torch.tensor(sample_probability)
    lables = []
    # give sample weight
    for index, (data, target) in enumerate(data):
        lables.append(target)
    weights = torch_sample_probability[lables]
    # create  inverse_imbalance_sampler
    inverse_imbalance_sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(data), replacement=True)
    return inverse_imbalance_sampler

# noisy parse


def uniform_mix_c(mixing_ratio, num_classes):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_c(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    torch.save(C, 'noisy_transaction_matrix_real.pt')
    return C


def flip_labels_c_two(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

def circle_flip_label(corruption_prob, num_classes):
    C_1 = np.eye(num_classes)
    C_2 = np.fliplr(C_1)
    return C_1 * (1 - corruption_prob) + C_2 * corruption_prob

def manual_label(corruption_prob, num_classes,seed=1):
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    C[0][1] = corruption_prob
    C[1][0] = corruption_prob
    for i in range(2,num_classes):
        C[i][np.random.choice(row_indices[(row_indices != i ) & (row_indices > 1)])] = corruption_prob
    torch.save(C, 'noisy_transaction_matrix_real.pt')
    return C

def get_noisy_dataset(dataset, args):
    # avoid make influence on origin dataset
    new_dataset = copy.deepcopy(dataset)
    if args.corruption_type == 'unif':
        noisy_transaction_matrix_real = uniform_mix_c(args.corruption_prob, args.num_classes)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'flip':
        # noisy_transaction_matrix_real = flip_labels_c(args.corruption_prob, args.num_classes,seed=args.seed)
        noisy_transaction_matrix_real = flip_labels_c(args.corruption_prob, args.num_classes,seed=args.seed)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'flip2':
        noisy_transaction_matrix_real = flip_labels_c_two(args.corruption_prob, args.num_classes,seed=args.seed)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'cflip':
        noisy_transaction_matrix_real = circle_flip_label(args.corruption_prob, args.num_classes)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'manual':
        # noisy_transaction_matrix_real = flip_labels_c(args.corruption_prob, args.num_classes,seed=args.seed)
        noisy_transaction_matrix_real = manual_label(args.corruption_prob, args.num_classes)
        print(noisy_transaction_matrix_real)
    else:
        noisy_transaction_matrix_real = None
    for i in range(len(new_dataset.targets)):
        new_dataset.targets[i] = np.random.choice(args.num_classes, p=noisy_transaction_matrix_real[new_dataset.targets[i]])
    return new_dataset, noisy_transaction_matrix_real
