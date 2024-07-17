import sys
from matplotlib.pyplot import get
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from utils.util import make_dir, AverageMeter
from settings.configs import *
from datas.data_utils import *
from models.PreResNet import *
from validates.validation import *
from math import inf
import math
from sklearn import metrics
import torch.distributed as dist
from matplotlib import pyplot as plt

import pdb


# get shell arg
args = parse_arguments()

# fixed seed
set_seed(args.seed)
# device detect
device = set_device(args)
# load config
config = load_config()
if args.dataset == 'cifar10':
    args.data_path = '../data-local/cifar10/cifar-10-batches-py/'
    if args.corruption_type == 'human':
        args.noise_path = './datas/CIFAR-10_human.pt'
        args.log_path = './results/cifar10n/'
    else:
        args.noise_path = None
        args.log_path = './results/cifar10/'
elif args.dataset == 'cifar100':
    args.data_path = '../data-local/cifar100/cifar-100-python/'
    if args.corruption_type == 'human':
        args.noise_path = './datas/CIFAR-100_human.pt'
        args.log_path = './results/cifar100n/'
    else:
        args.noise_path = None
        args.log_path = './results/cifar100/'
elif args.dataset == 'imagenet':
    args.data_path = '../data-local/mini-imagenet/mini-imagenet/'
    args.log_path = './results/red_imagenet/'

# Dataloader set
kwargs = {'num_workers': config['Dataloader_set']['num_workers'],
          'pin_memory': config['Dataloader_set']['pin_memory']}

iters = 0

# # get dataset
# train_data_meta, train_data, test_dataset = build_dataset(args)

# # make imbalance dataset 
# imbalanced_train_dataset = get_imbalance_dataset(args,train_data)
# # get noisy dataset 
# noisy_train_dataset, noisy_transaction_matrix_real = get_noisy_dataset(train_data, args)
# # make imbalance and noisy dataset 
# imbalanced_and_noisy_train_dataset, noisy_transaction_matrix_real = get_noisy_dataset(imbalanced_train_dataset, args)
# # imbalanced_and_noisy_train_dataset.update()
# imbalanced_and_noisy_train_loader = torch.utils.data.DataLoader(imbalanced_and_noisy_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

# dist initialize
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("gloo", rank=args.rank, world_size=args.world_size)

class BalSCL_BAL(nn.Module):
    def __init__(self, temperature=0.1):
        super(BalSCL_BAL, self).__init__()
        self.temperature = temperature

    def forward(self, centers1, features, targets, epoch=None, cls_num_list=None):
        '''
        centers1: [num_classes, feature_dim]
        features: [B, 2, feature_dim] (one from the standard image, one from the augmented image)
        targets: [2*B] labels for loss function
        '''
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        batch_size = features.shape[0]//2
        targets = targets.contiguous().view(-1, 1)
        centers1 = centers1.reshape(-1, centers1.size(-1))
        repeat_size = centers1.size(0)//len(cls_num_list)
        targets_centers = torch.arange(len(cls_num_list), device=device).repeat(repeat_size, 1).view(-1, 1)
        if centers1 is not None:
            targets = torch.cat([targets, targets_centers], dim=0)
        else:
            targets = targets
        batch_cls_count = torch.eye(len(cls_num_list)).to(device)[targets].sum(dim=0).squeeze()
        
        # [2*B, targets+num_classes]: remove identity element 
        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device) 
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * 2).view(-1, 1).to(device), 0) 
        mask = mask * logits_mask
        
        # class-complement: [2*B, targets+num_classes]
        # features = torch.cat(torch.unbind(features, dim=1), dim=0)
        if centers1 is not None:
            features = torch.cat([features, centers1], dim=0)
        else:
            features = features
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        if centers1 is not None:
            per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(2 * batch_size, 2 * batch_size + repeat_size*len(cls_num_list)) - mask
        else:
            per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(2 * batch_size, 2 * batch_size) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss

class BalSCL_SSL_BAL(nn.Module):
    def __init__(self, temperature=0.1):
        super(BalSCL_SSL_BAL, self).__init__()
        self.temperature = temperature

    def forward(self, centers1, features, targets, logits=None, cls_num_list=None, conf_mask=None, epoch=None):
        # centers1: [num_classes, feature_dim], features: [2*2, feature_dim] (one from the standard image, one from the augmented image), targets: [2*B, num_classes]
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        batch_size = features.shape[0]//2
        targets = targets.contiguous()
        centers1 = centers1.reshape(-1, centers1.size(-1))
        repeat_size = centers1.size(0)//len(cls_num_list)
        cls_num_list = cls_num_list.to(device)
        targets_centers = torch.eye(len(cls_num_list), device=device).repeat(repeat_size, 1)
        if centers1 is not None:
            targets = torch.cat([targets, targets_centers], dim=0) # [2*B+num_classes, num_classes]
        else:
            targets = targets
        target_preds = torch.max(targets, dim=1)[1]
        base_count = torch.eye(len(cls_num_list)).to(device)[target_preds].sum(dim=0).squeeze()
        
        batch_cls_count = base_count
        mask = torch.mm(targets[:2 * batch_size], targets.T).float().to(device) # [B, targets+num_classes]
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * 2).view(-1, 1).to(device), 0) # remove identity element
        mask = mask * logits_mask
        
        # class-complement
        if centers1 is not None:
            features = torch.cat([features, centers1], dim=0)
        else:
            features = features
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        if centers1 is not None:
            per_ins_weight = torch.tensor([batch_cls_count[i] for i in target_preds], device=device).view(1, -1).expand(2 * batch_size, 2 * batch_size + repeat_size*len(cls_num_list)) - mask
        else:
            per_ins_weight = torch.tensor([batch_cls_count[i] for i in target_preds], device=device).view(1, -1).expand(2 * batch_size, 2 * batch_size) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = - mean_log_prob_pos * conf_mask
        loss = loss.sum()/conf_mask.sum()
        return loss


def balanced_softmax_loss(labels, logits, sample_per_class, args):
    batch_size, num_classes = logits.shape
    if labels.dim() == 1:
        labels = torch.eye(num_classes).to(logits.device)[labels]
    spc = torch.tensor(sample_per_class).type_as(logits)
    spc = spc/spc.sum()
    spc = spc.unsqueeze(0).repeat(logits.shape[0], 1)
    logits = logits + spc.log()
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1))
    return loss

def balanced_softmax_loss_semi(logits_x, targets_x, logits_u, targets_u, sample_per_class, args):
    spc = sample_per_class.type_as(logits_x)
    spc = spc/spc.sum()
    spc_x = spc.unsqueeze(0).expand(logits_x.shape[0], -1)
    logits_x = logits_x + spc_x.log()
    Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))

    spc_u = spc.unsqueeze(0).expand(logits_u.shape[0], -1)
    logits_u = logits_u + spc_u.log()
    probs_u = torch.softmax(logits_u, dim=1)
    Lu = torch.mean((probs_u - targets_u)**2)
    return Lx, Lu

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, tmp_img_num_list, MIDL_memory=None, args=None):
    net.train()
    net2.eval() #fix one network and train the other
    criterion_scl_ssl =  BalSCL_SSL_BAL(args.SBCL_temp)
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    
    global iters
    for batch_idx, (inputs_x, inputs_x2, inputs_x3, _, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            try:
                inputs_u, inputs_u2, inputs_u3, _ = unlabeled_train_iter.next()
            except:
                inputs_u, inputs_u2, inputs_u3, _ = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Pseudo-label generation used in DivideMix
        with torch.no_grad():
            # label co-guessing of unlabeled samples
            features_u13 = net(inputs_u3, return_features=True)
            features_u23 = net2(inputs_u3, return_features=True)

            pu = (torch.softmax(net.classify1(features_u13), dim=1) + torch.softmax(net2.classify1(features_u23), dim=1)) / 2
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x13, outputs_x13_bal = net(inputs_x3)
            outputs_x23, outputs_x23_bal = net2(inputs_x3)

            px = (torch.softmax(outputs_x13, dim=1) + torch.softmax(outputs_x23, dim=1) + torch.softmax(outputs_x13_bal, dim=1) + torch.softmax(outputs_x23_bal, dim=1)) / 4
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 

            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        
        mixed_feats = net(mixed_input, return_features=True)

        mixed_logits, mixed_logits_bal = net.classify1(mixed_feats), net.classify2(mixed_feats)
        logits_x = mixed_logits[:batch_size*2]
        logits_u = mixed_logits[batch_size*2:]
        logits_x_bal = mixed_logits_bal[:batch_size*2]
        logits_u_bal = mixed_logits_bal[batch_size*2:]
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        loss_BCE_x, loss_BCE_u = balanced_softmax_loss_semi(logits_x_bal, mixed_target[:batch_size*2], logits_u_bal, mixed_target[batch_size*2:], tmp_img_num_list, args)

        # Regularization
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()        
        pred_mean = torch.softmax(mixed_logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + loss_BCE_x + lamb * loss_BCE_u + penalty

        # Semi-supervised balanced loss for high-confidence samples
        if args.train_SBCL:
            feats_input = net(all_inputs, return_features=True)
            target_a, target_b = all_targets, all_targets
            q1 = net.project1(feats_input)
            logits, logits_bal = net.classify1(feats_input), net.classify2(feats_input)
            proto_centers = net.get_class_prototypes1()
            
            
            if args.conf_mask_threshold > 0.0:
                if args.conf_mask_type == 'targets':
                    conf_mask = torch.max(target_a, dim=1)[0] > args.conf_mask_threshold
                    conf_mask_bal = torch.max(target_a, dim=1)[0] > args.conf_mask_threshold
                elif args.conf_mask_type == 'logits':
                    conf_mask = torch.max(torch.softmax(logits, dim=1), dim=1)[0] > args.conf_mask_threshold
                    conf_mask_bal = torch.max(torch.softmax(logits_bal, dim=1), dim=1)[0] > args.conf_mask_threshold
                    conf_mask, conf_mask_bal = torch.logical_and(conf_mask, conf_mask_bal), torch.logical_and(conf_mask, conf_mask_bal)
                conf_labels = torch.argmax(target_a, dim=1)[conf_mask]
                # index, counts = torch.unique(conf_labels, return_counts=True)
            else:
                conf_mask = torch.ones(q1.size(0)).to(device)
            
            SBCL_loss = criterion_scl_ssl(proto_centers, q1, target_a, logits.detach(), tmp_img_num_list, conf_mask, epoch)

            loss += args.SBCL_lambda * SBCL_loss

        # Mixup-enhanced instance discrimination contrastive loss for low-confidence samples
        if args.train_MIDL:
            if args.conf_mask_threshold > 0.0:
                MIDL_mask = torch.max(target_a, dim=1)[0] < args.conf_mask_threshold
                MIDL_mask = torch.cat([MIDL_mask[:inputs_x.size(0)], MIDL_mask[inputs_x.size(0)*2:inputs_x.size(0)*2+inputs_u.size(0)]], dim=0)
            else:
                MIDL_mask = torch.ones(q1.size(0)//2).to(device)
            MIDL_q1 = torch.cat([q1[:inputs_x.size(0)], q1[inputs_x.size(0)*2:inputs_x.size(0)*2+inputs_u.size(0)]], dim=0)
            MIDL_k1 = torch.cat([q1[inputs_x.size(0):inputs_x.size(0)*2], q1[inputs_x.size(0)*2+inputs_u.size(0):]], dim=0)
            MIDL_queue = MIDL_memory
            if MIDL_queue.size(0) == 0:
                MIDL_loss = 0
            else:
                pos_l = torch.einsum('nc,nc->n', [MIDL_q1, MIDL_k1.detach()]).unsqueeze(-1) # [B, 1]
                neg_l = torch.einsum('nc,kc->nk', [MIDL_q1, MIDL_queue]) # [B, N_queue]
                MIDL_logits = torch.cat([pos_l, neg_l], dim=1)
                MIDL_labels = torch.arange(len(MIDL_logits), device=device)
                MIDL_loss = F.cross_entropy(MIDL_logits/args.MIDL_temp, MIDL_labels, reduction='none')
                if MIDL_mask.sum() == 0:
                    MIDL_loss = 0
                else:
                    MIDL_loss = MIDL_loss * MIDL_mask
                    MIDL_loss = MIDL_loss.sum()/MIDL_mask.sum()
            
            # update Instance Discrimination Contrastive Loss memory
            q1_mix= net.project1(mixed_feats)
            k1_mix = torch.cat([q1_mix[inputs_x.size(0):inputs_x.size(0)*2], q1_mix[inputs_x.size(0)*2+inputs_u.size(0):]], dim=0)
            MIDL_memory = torch.cat([MIDL_memory, k1_mix.detach()], dim=0)
            del q1_mix, k1_mix
            
            if MIDL_memory.size(0) > args.MIDL_memory_size:
                MIDL_memory = MIDL_memory[-args.MIDL_memory_size:]
            loss += args.MIDL_lambda * MIDL_loss
        else:
            MIDL_memory = None

        iters += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                    %(args.dataset, args.imb_factor, args.corruption_type,args.corruption_prob, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
            sys.stdout.flush()
        
    return MIDL_memory

def warmup_net(epoch, net, optimizer, dataloader, img_num_list, MIDL_memory=None, args=None):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    criterion_scl = BalSCL_BAL(args.SBCL_temp)

    global iters
    for batch_idx, (inputs, inputs2, _, _, labels, path) in enumerate(dataloader):
        inputs, inputs2, labels = inputs.cuda(), inputs2.cuda(), labels.cuda() 
        optimizer.zero_grad()
        feats1 = net(inputs, return_features=True)
        feats2 = net(inputs2, return_features=True)

        outputs, outputs_bal = net.classify1(feats1), net.classify2(feats1)
        outputs2, outputs2_bal = net.classify1(feats2), net.classify2(feats2)
        loss = CEloss(outputs, labels)
        loss2 = CEloss(outputs2, labels)
        loss_bal = balanced_softmax_loss(labels, outputs_bal, img_num_list, args)
        loss2_bal = balanced_softmax_loss(labels, outputs2_bal, img_num_list, args)
        L = (loss + loss_bal + loss2 + loss2_bal)/2

        # SBCL
        if args.warmup_SBCL and epoch >= args.SBCL_epochs:
            q1 = net.project1(feats1)
            q2 = net.project1(feats2)
            proto_centers = net.get_class_prototypes1()
            SBCL_loss = criterion_scl(proto_centers, torch.cat([q1, q2], dim=0), torch.cat([labels, labels], dim=0), epoch, img_num_list)
            L += args.SBCL_lambda * SBCL_loss
        
        # Prepare MIDL memory
        if args.train_MIDL and epoch >= args.SBCL_epochs:
            MIDL_memory = torch.cat([MIDL_memory, q1.detach().clone()], dim=0)
            if MIDL_memory.size(0) > args.MIDL_memory_size:
                MIDL_memory = MIDL_memory[-args.MIDL_memory_size:]

        iters += 1
        L.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.2f-%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f\t BCE-loss: %.4f'
                    %(args.dataset, args.imb_factor, args.corruption_type,args.corruption_prob, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item(), loss_bal.item()))
            sys.stdout.flush()  
        
    return MIDL_memory

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct, correct1, correct2 = 0, 0, 0
    correct_bal, correct1_bal, correct2_bal = 0, 0, 0
    correct_avg, correct1_avg, correct2_avg = 0, 0, 0
    total = 0
    acc_category = [0, 0, 0, 0]
    correct_cls = [0 for i in range(args.num_classes)]
    total_cls = [0 for i in range(args.num_classes)]
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            features1 = net1(inputs, return_features=True)
            features2 = net2(inputs, return_features=True)
            outputs1, outputs1_bal = net1.classify1(features1), net1.classify2(features1)
            outputs2, outputs2_bal = net2.classify1(features2), net2.classify2(features2)
            
            outputs = outputs1+outputs2
            outputs_bal = outputs1_bal+outputs2_bal
            outputs_avg = outputs1 + outputs2 + outputs1_bal + outputs2_bal
            outputs1_avg = outputs1 + outputs1_bal
            outputs2_avg = outputs2 + outputs2_bal
            _, predicted = torch.max(outputs, 1)
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)
            _, predicted_bal = torch.max(outputs_bal, 1)
            _, predicted1_bal = torch.max(outputs1_bal, 1)
            _, predicted2_bal = torch.max(outputs2_bal, 1)
            _, predicted_avg = torch.max(outputs_avg, 1)
            _, predicted1_avg = torch.max(outputs1_avg, 1)
            _, predicted2_avg = torch.max(outputs2_avg, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
            correct1 += predicted1.eq(targets).cpu().sum().item()                 
            correct2 += predicted2.eq(targets).cpu().sum().item()                 
            correct_bal += predicted_bal.eq(targets).cpu().sum().item()
            correct1_bal += predicted1_bal.eq(targets).cpu().sum().item()
            correct2_bal += predicted2_bal.eq(targets).cpu().sum().item()
            correct_avg += predicted_avg.eq(targets).cpu().sum().item()
            correct1_avg += predicted1_avg.eq(targets).cpu().sum().item()
            correct2_avg += predicted2_avg.eq(targets).cpu().sum().item()
            for i in range(len(targets)):
                total_cls[targets[i]] += 1
                if predicted_avg[i] == targets[i]:
                    correct_cls[targets[i]] += 1
    acc = 100.*correct/total
    acc1 = 100.*correct1/total
    acc2 = 100.*correct2/total
    acc_avg = 100.*correct_avg/total
    acc_cls = [100.*correct_cls[i]/total_cls[i] for i in range(args.num_classes)]
    acc_category[0], acc_category[1], acc_category[2], acc_category[3] = sum(acc_cls)/args.num_classes, sum(acc_cls[0:2])/2, sum(acc_cls[2:7])/5, sum(acc_cls[7:])/3
    
    print("\n| Test Epoch #%d\t Accuracy(Conv): %.2f" %(epoch, acc))
    print("| Model1: %.2f  | Model 2: %.2f\n"%(acc1, acc2))
    test_log.write('Epoch:%d  Acc(Conv):%.2f (Model1:%.2f Model2:%.2f)\n'%(epoch, acc, acc1, acc2))
    test_log.flush()
    test_cls_log.write('Epoch:%d  Acc_cls:%s  Acc_category:%s\n'%(epoch, acc_cls, acc_category))
    test_cls_log.flush()
    return acc_avg


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
    
def create_model(args=None):
    if args.dataset == 'imagenet':
        model = ResNet18(num_classes=args.num_classes, args=args)
    else:
        model = Preact_ResNet18(num_classes=args.num_classes, args=args)
    model = model.cuda()
    return model

def get_knn_center(model,dataloader_,device):
	print('===> Calculating KNN centroids.')
	feats_all, labels_all = [], []

	# Calculate initial centroids only on training data.
	with torch.no_grad():
		for idxs,(inputs, labels)  in enumerate(dataloader_):

			inputs = set_tensor(inputs,False,device)
			labels = set_tensor(labels,False,device)

			# Calculate Features of each training data
			features = forward_2(model,inputs)

			feats_all.append(features.cpu().numpy())
			labels_all.append(labels.cpu().numpy())
	
	feats = np.concatenate(feats_all)
	labels = np.concatenate(labels_all)

	featmean = feats.mean(axis=0)

	def get_centroids(feats_, labels_):
		centroids = []        
		for i in np.unique(labels_):
			centroids.append(np.mean(feats_[labels_==i], axis=0))
		return np.stack(centroids)
	# Get unnormalized centorids
	un_centers = get_centroids(feats, labels)

	# Get l2n centorids
	l2n_feats = torch.Tensor(feats.copy())
	norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
	l2n_feats = l2n_feats / norm_l2n
	l2n_centers = get_centroids(l2n_feats.numpy(), labels)

	# Get cl2n centorids
	cl2n_feats = torch.Tensor(feats.copy())
	cl2n_feats = cl2n_feats - torch.Tensor(featmean)
	norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
	cl2n_feats = cl2n_feats / norm_cl2n
	cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

	return {'mean': featmean,
			'uncs': un_centers,
			'l2ncs': l2n_centers,   
			'cl2ncs': cl2n_centers}

def prototypical_eval_train(loader, model, epoch, prototypes_prev, prototypes_bal_prev, prev_tmp_img_num_list, args):
    if args.train_SBCL == False and args.train_MIDL == False:
        args.feature_type == 'default'

    model.eval()
    eval_loader = loader.run('eval_train')
    total_outputs = torch.zeros((num_all_img, args.num_classes))  # save outputs of conventional branch
    if args.feature_type == 'project':
        total_features = torch.zeros((num_all_img, model.hidden_dim))  # save sample features
    elif args.feature_type == 'default':
        total_features = torch.zeros((num_all_img, model.hidden_dim*4))  # save sample features
    else:
        pdb.set_trace()
    total_labels = torch.zeros(num_all_img).long()  # save sample labels
    total_clean_labels = torch.zeros(num_all_img).long()  # save sample labels
    tmp_img_num_list = torch.zeros(args.num_classes)  # compute N_k from clean sample set
    pred = np.zeros(num_all_img, dtype=bool)  # clean probability

    conf_threshold = args.phi ** epoch / args.num_classes
    confs = torch.zeros(num_all_img)  # confidence from ABC
    confs_bal = torch.zeros(num_all_img)  # confidence from ABC
    max_confs = torch.zeros(num_all_img)  # save sample max confidence
    max_confs_bal = torch.zeros(num_all_img)  # save sample max confidence

    with torch.no_grad():
        for batch_idx, (inputs, inputs2, inputs3, clean_targets, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats = model(inputs, return_features=True)
            logits, logits_bal = model.classify1(feats), model.classify2(feats)
            q = model.project1(feats)
            prob = F.softmax(logits, dim=1)
            prob_bal = F.softmax(logits_bal, dim=1)
            confs[index] = prob[range(prob.size(0)), targets].cpu()
            confs_bal[index] = prob_bal[range(prob_bal.size(0)), targets].cpu()
            max_confs[index] = torch.max(prob, dim=1)[0].cpu()
            max_confs_bal[index] = torch.max(prob_bal, dim=1)[0].cpu()
            
            for b in range(inputs.size(0)):
                total_outputs[index[b]] = logits[b]
                if args.feature_type == 'project':
                    total_features[index[b]] = F.normalize(q[b].cpu(), dim=-1)
                elif args.feature_type == 'default':
                    total_features[index[b]] = F.normalize(feats[b].cpu(), dim=-1)

                total_labels[index[b]] = targets[b]
                total_clean_labels[index[b]] = clean_targets[b]
    
    confs, total_labels, total_clean_labels = confs.cuda(), total_labels.cuda(), total_clean_labels.cuda()
    total_outputs, total_features = total_outputs.cuda(), total_features.cuda()
    max_confs = max_confs.cuda()
    confs_bal, max_confs_bal = confs_bal.cuda(), max_confs_bal.cuda()

    mask = max_confs > conf_threshold
    mask_bal = max_confs_bal > conf_threshold
    total_outputs = F.softmax(total_outputs/args.eval_temp, dim=1)
    
    if args.proto_assign_type == 'DaCC':
        current_mask = torch.logical_and(mask, mask_bal)
        outputs_conf = total_outputs[current_mask]
        features_conf = total_features[current_mask]
        prototypes = F.normalize(outputs_conf.T.mm(features_conf), dim=1)
    else:
        pdb.set_trace()

    assign_label =  F.softmax(total_features.mm(prototypes.T), dim=1)
    losses = -assign_label[torch.arange(total_labels.size(0)), total_labels]
    losses = losses.cpu().numpy()[:, np.newaxis]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    total_labels = total_labels.cpu().numpy()
    prob = np.zeros((losses.shape[0],))
    
    from sklearn.mixture import GaussianMixture
    for i in range(args.num_clusters):
        matching_idx = total_labels == i
        gm = GaussianMixture(n_components=2, random_state=0).fit(losses[matching_idx, :])
        pdf = gm.predict_proba(losses[matching_idx, :])
        prob[matching_idx] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
    
    for i in range(args.num_classes):
        pred[idx_class[i]] = (prob[idx_class[i]] > args.p_threshold)
        tmp_img_num_list[i] = np.sum(pred[idx_class[i]])
        
    select_index = pred.nonzero()[0]

    return select_index, prob, tmp_img_num_list, prototypes

def update_batchnorm(model, train_loader, args, verbose=False):
    if verbose: print("Updating Batchnorm")

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    if not momenta:
        return
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    model.train()
    i = 0
    for batch_idx, (inputs, _, _, _, labels, path) in enumerate(train_loader):
        if i > 100:
            return
        inputs, labels = inputs.cuda(), labels.cuda()
        feats1, feats1_bal = model(inputs, return_features=True)
        outputs, outputs_bal = model.classify1(feats1), model.classify2(feats1_bal)

        if verbose and i % 100 == 0:
            print('Updating BN. i = {}'.format(i))

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]


def bi_dimensional_sample_selection_2(args,model,loader,epoch,devices):

    model.eval()
    
    select_sample_index_list = []
    select_sample_prob_list = []
    
    select_sample_index_list_1 = []
    select_sample_prob_list_1 = []
    select_sample_prototype_1 = []
    cluster_num_list_1 = [] 

    select_sample_index_list_2 = []
    select_sample_prob_list_2 = []
    select_sample_prototype_2 = []
    cluster_num_list_2 = [] 

    select_sample_dimensionnal_status = []
    tmp_img_num_list = torch.zeros(args.num_classes)  # compute N_k from clean sample set
    avg_pred_list = get_avg_pred_list(args, loader, model, devices) # compute average prediction [num_class, num_class]
    avg_pred_list_2 = get_avg_pred_list_2(args,loader,avg_pred_list,model,devices) # compute weighted avg prediction [num_class, num_class]
    cfeat = get_knn_center(model,imbalanced_and_noisy_train_loader,devices)
    mean_feat = torch.tensor(cfeat['mean']).to(devices)

    centriod_list,sample_num_list = get_centriod_list(args,avg_pred_list_2,mean_feat,model,devices)
    centriod_distance = torch.softmax(torch.einsum('ij,jk->ik',centriod_list,centriod_list.T),dim=1)

    for class_num in range(args.num_classes):
        class_dataloader = loader.run(mode='single', class_num = class_num)

        wjsd_info,index_info = get_wjsd_info(class_dataloader,avg_pred_list[class_num],model,devices) 
  
        acd_info, index_info = get_adaptive_centriod_distance_info(class_dataloader,centriod_list[class_num],model,mean_feat,devices) # model

        
        wjsd_info = np.array(wjsd_info)
        wjsd_info = (wjsd_info-wjsd_info.min())/(wjsd_info.max()-wjsd_info.min())
        acd_info = np.array(acd_info)
        acd_info = (acd_info-acd_info.min())/(acd_info.max()-acd_info.min())

        combine_wjsd = wjsd_info.reshape(-1,1)
        combine_acd = acd_info.reshape(-1,1)

        prob_wjsd, gmm_wjsd= gmm_fit_func(combine_wjsd)  
        prob_acd, gmm_acd= gmm_fit_func(combine_acd)        

        cluster_select_index_1_wjsd = (prob_wjsd[:,gmm_wjsd.means_.argmin()]>0.5)
        cluster_select_index_2_wjsd = ~cluster_select_index_1_wjsd
        cluster_index_1_wjsd = index_info[cluster_select_index_1_wjsd]
        cluster_index_2_wjsd = index_info[cluster_select_index_2_wjsd]
        
        cluster_select_index_1_acd = (prob_acd[:,gmm_acd.means_.argmin()]>0.5)
        cluster_select_index_2_acd = ~cluster_select_index_1_acd
        cluster_index_1_acd = index_info[cluster_select_index_1_acd]
        cluster_index_2_acd = index_info[cluster_select_index_2_acd]        

        if cluster_select_index_1_acd.size == 0:
            cluster_select_index_1_acd = cluster_select_index_2_acd 
        if cluster_select_index_2_acd.size == 0:
            cluster_select_index_2_acd = cluster_select_index_1_acd 

        acd_wjsd_mean_1 = wjsd_info[cluster_select_index_1_acd].mean(0)
        acd_wjsd_std_1 = wjsd_info[cluster_select_index_1_acd].std(0)
        acd_wjsd_pred_1 = gmm_wjsd.predict(acd_wjsd_mean_1.reshape(1, -1))[0]
        acd_wjsd_mean_2 = wjsd_info[cluster_select_index_2_acd].mean(0)
        acd_wjsd_std_2 = wjsd_info[cluster_select_index_2_acd].std(0)
        acd_wjsd_pred_2 = gmm_wjsd.predict(acd_wjsd_mean_2.reshape(1, -1))[0]

        std_list = [1] * 2
        if acd_wjsd_std_1:
            std_list[acd_wjsd_pred_1] = acd_wjsd_std_1
        else:
            std_list[acd_wjsd_pred_1] = 0.1 
        if acd_wjsd_std_2:
            std_list[acd_wjsd_pred_2] = acd_wjsd_std_2
        else:
            std_list[acd_wjsd_pred_2] = 0.1 

        if (acd_wjsd_pred_1 == acd_wjsd_pred_2 and acd_wjsd_pred_1 != gmm_wjsd.means_.argmin()) or (std_list[gmm_wjsd.means_.argmax()] / std_list[gmm_wjsd.means_.argmin()] < 0.65 ):
            select_sample_dimensionnal_status.append(1)
            select_sample_index_list_1.append(cluster_index_1_wjsd)
            select_sample_prob_list_1.append(prob_wjsd[:,gmm_wjsd.means_.argmin()][cluster_select_index_1_wjsd])

            select_sample_index_list_2.append(cluster_index_2_wjsd)
            select_sample_prob_list_2.append(prob_wjsd[:,gmm_wjsd.means_.argmax()][cluster_select_index_2_wjsd])

        else:
            select_sample_dimensionnal_status.append(0)         
            select_sample_index_list_1.append(cluster_index_1_acd)
            select_sample_prob_list_1.append(prob_acd[:,gmm_acd.means_.argmin()][cluster_select_index_1_acd])

            select_sample_index_list_2.append(cluster_index_2_acd)
            select_sample_prob_list_2.append(prob_acd[:,gmm_acd.means_.argmax()][cluster_select_index_2_acd])

    for class_num in range(args.num_classes):
        if select_sample_dimensionnal_status[class_num]:
            select_sample_index_list.extend(select_sample_index_list_1[class_num])
            select_sample_prob_list.extend(select_sample_prob_list_1[class_num])
            tmp_img_num_list[class_num] = len(select_sample_index_list_1[class_num])
        else:
            centriod_distance_copy = copy.deepcopy(centriod_distance[class_num])
            current_centriod_distance =copy.deepcopy(centriod_distance_copy[class_num]) 
            centriod_distance_copy[class_num] = 0
            max_centriod_distance,max_centriod_indice = centriod_distance_copy.topk(k=1, largest=True)
 
            if (abs((current_centriod_distance-max_centriod_distance[0]).item()) < 0.1 * current_centriod_distance.item()  and (sample_num_list[class_num] < sample_num_list[max_centriod_indice[0].item()])): 
                select_sample_index_list.extend(select_sample_index_list_1[class_num])
                select_sample_prob_list.extend(select_sample_prob_list_1[class_num])
                tmp_img_num_list[class_num] = len(select_sample_index_list_1[class_num])
            else:
                select_sample_index_list.extend(select_sample_index_list_2[class_num])
                select_sample_prob_list.extend(select_sample_prob_list_2[class_num])
                tmp_img_num_list[class_num] = len(select_sample_index_list_2[class_num])
    return select_sample_index_list, select_sample_prob_list, tmp_img_num_list



def get_normalization_info(info_1,info_2):
	info = np.array(info_1 + info_2)
	normal_info = (info-info.min())/(info.max()-info.min())
	normal_info_1 = normal_info[:len(info_1)].tolist()
	normal_info_2 = normal_info[len(info_1):].tolist()
	return normal_info_1,normal_info_2

def get_wjsd_info(data_loader,avg_pred,model,device):
    model.eval()
    jsd_info=[]
    index_info = [] 
    JS_dist = Jensen_Shannon()
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)

        y_f, _ = model(input_var)
        out = torch.softmax(y_f,dim=1)

        idx = torch.tensor([x for x in range(len(out))])
        weight = out[idx,torch.argmax(out,dim=1)] / out[:,target_var[0]]
        weight = torch.clamp(weight,min=1,max = (avg_pred[torch.argmax(avg_pred,dim=0)] / avg_pred[target_var[0]]).item())

        jsd =  weight * JS_dist(out,  F.one_hot(target_var, num_classes = args.num_classes))

        jsd_info.extend(jsd.tolist())
        index_info.extend(indexs.tolist())
    return jsd_info,np.array(index_info)

def get_jsd_info(data_loader,avg_pred,model,class_num,device):
    model.eval()
    jsd_info=[]
    index_info = [] 
    JS_dist = Jensen_Shannon()

    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)

        y_f, _ = model(input_var)
        out = torch.softmax(y_f,dim=1)
        
        idx = torch.tensor([x for x in range(len(out))])
        weight = out[idx,torch.argmax(out,dim=1)] / out[:,target_var[0]]

        weight = torch.clamp(weight,min=1,max = (avg_pred[torch.argmax(avg_pred,dim=0)] / avg_pred[target_var[0]]).item())

        jsd =  JS_dist(out,  F.one_hot(target_var, num_classes = args.num_classes))

        jsd_info.extend(jsd.tolist())
        index_info.extend(indexs.tolist())
    return jsd_info,np.array(index_info)

def get_adaptive_centriod_distance_info(data_loader,centriod,model,meat_feat,devices):
    model.eval()
    dist_info = []
    index_info = []
    for i, (input, target,indexs) in enumerate(data_loader):


        input_var = set_tensor(input, False, devices)

        features = forward_2(model, input_var) - meat_feat
        features = F.normalize(features,p=2,dim=1)

        dist = torch.einsum('ij,j->i',features,centriod.T)
        dist_info.extend(dist.tolist())
        index_info.extend(indexs.tolist())
    return dist_info,np.array(index_info)


def get_adaptive_centriod_distance_info_(data_loader,centriod,model,meat_feat,devices):
    model.eval()
    dist_info = []
    index_info = []
    feature_info = []
    for i, (input, target,indexs) in enumerate(data_loader):


        input_var = set_tensor(input, False, devices)

        features_ = forward_2(model, input_var) - meat_feat
        features = F.normalize(features_,p=2,dim=1)

        dist = torch.einsum('ij,j->i',features,centriod.T)
        dist_info.extend(dist.tolist())
        index_info.extend(indexs.tolist())
        feature_info.extend(features_.tolist())
    return dist_info,np.array(index_info),np.array(feature_info)
        
def get_adaptive_centriod(args,global_dataloader,local_dataloader,avg_pred,class_num,feat_mean,model,devices):

    high_confidence_samples,sample_num = get_high_confidence_samples(local_dataloader,avg_pred,class_num,model,devices)

    adptive_feat_c = high_confidence_samples - feat_mean
    adptive_feat_cl2 = F.normalize(adptive_feat_c,p=2,dim=1)
    adptive_centriod  = adptive_feat_cl2.mean(0)
    return adptive_centriod,sample_num

def get_high_confidence_samples(global_dataloader,avg_pred,class_num,model,devices):
    select_features_list = torch.tensor([]).to(devices)
    avg_pred = avg_pred[class_num]
    sample_num = 0
    for i,(input, target,indexs) in enumerate(global_dataloader):

        input_var = set_tensor(input, False, device)

        features = forward_2(model,input_var)
        y_f = model.linear(features)
        preds = torch.softmax(y_f,dim=1)
        arg_idx = torch.argmax(preds,dim=1)
        select_ = torch.eq(arg_idx, torch.argmax(avg_pred))
        get_high_confidence_criterion = avg_pred[torch.argmax(avg_pred)]
        select_index = torch.gt(preds[:,torch.argmax(avg_pred)],get_high_confidence_criterion)

        select_features = features[select_index*select_]
        sample_num += (select_index*select_).sum().item()
        select_features_list = torch.cat([select_features_list,select_features],dim=0)

    if sample_num == 0 :
        for i,(input, target,indexs) in enumerate(global_dataloader):

            input_var = set_tensor(input, False, device)

            features = forward_2(model,input_var)

            select_features = features
            sample_num += len(target)
            select_features_list = torch.cat([select_features_list,select_features],dim=0)
        
    return select_features_list,sample_num

def get_avg_pred(data_loader,model,devices):
    model.eval()

    avg_pred = torch.tensor([]).to(devices)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)

        y_f, _ = model(input_var)
        out = torch.softmax(y_f,dim=1).mean(0).unsqueeze(0)
        avg_pred = torch.cat([avg_pred,out],dim=0)

    return avg_pred.mean(0)

def get_avg_pred_2(data_loader,avg_pred_2,model,device):
	model.eval()
	avg_pred = torch.tensor([]).to(device)
	avg_argmax = torch.argmax(avg_pred_2,dim=0) # 특정 class의 conf가 가장 큰 위치를 찾는 것 (해당 class가 해당 class 예측할 때 true일 듯)
	for i, (input, target,indexs) in enumerate(data_loader):

		input_var = set_tensor(input, False, device)

		y_f, _ = model(input_var)
		out = torch.softmax(y_f,dim=1)
		idx = [i for i in range(target.shape[0])]
		weight = torch.clamp(out[idx,avg_argmax] / avg_pred_2[avg_argmax],min=1)
		out[idx,avg_argmax] = weight * out[idx,avg_argmax]
		avg_pred = torch.cat([avg_pred,out.mean(0).unsqueeze(0)],dim=0)

	return avg_pred.mean(0)

def get_avg_pred_list(args,loader,model,devices):

    avg_pred_list = torch.tensor([]).to(devices)
    for class_num in range(args.num_classes):

        class_dataloader = loader.run(mode='single',class_num=class_num)
        avg_pred = get_avg_pred(class_dataloader,model,devices).unsqueeze(0) 

        avg_pred_list = torch.cat([avg_pred_list,avg_pred],dim=0)
        del class_dataloader
    return avg_pred_list

def get_avg_pred_list_2(args,loader,avg_pred_list,model,devices):

    avg_pred_list_2 = torch.tensor([]).to(devices)
    for class_num in range(args.num_classes):

        class_dataloader = loader.run(mode='single',class_num=class_num)
        avg_pred = get_avg_pred_2(class_dataloader,avg_pred_list[class_num],model,devices).unsqueeze(0) 

        avg_pred_list_2 = torch.cat([avg_pred_list_2,avg_pred],dim=0)
        del class_dataloader
    return avg_pred_list_2

def get_centriod_list(args,avg_pred_list_2,mean_feat,model,devices):
    centriod_list = torch.tensor([]).to(devices)
    sample_num_list = []
    for class_num in range(args.num_classes):
        class_dataloader = loader.run(mode='single',class_num=class_num)
        centriod,sample_num = get_adaptive_centriod(args,'other_class_dataloader',class_dataloader,avg_pred_list_2,class_num,mean_feat,model,devices) 
        centriod_list = torch.cat([centriod_list,centriod.unsqueeze(0)],dim=0)
        sample_num_list.append(sample_num)
        del class_dataloader
    return centriod_list,sample_num_list
# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

def gmm_fit_func(input_loss):
    input_loss = np.array(input_loss)

    gmm = GaussianMixture(n_components=2,max_iter=30,tol=1e-2,reg_covar=5e-4) 
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 

    return prob,gmm
## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

def forward_1(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

def forward_2(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)

        return out
if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30
elif args.dataset == 'imagenet':
    warm_up = 30

if args.warm_up > 0 :
    warm_up = args.warm_up
args.file_name = '_'.join([args.dataset, 'IBexp'+str(args.imb_factor), 'N'+args.corruption_type, str(args.corruption_prob), args.exp_str])
args.file_root = args.log_path + args.file_name
args.log_name = args.file_name + f'_w{warm_up}'
args.num_clusters = args.num_classes
if not os.path.exists('./results'):
    make_dir('./results')
if not os.path.exists(args.log_path):
    make_dir(args.log_path)
if not os.path.exists(args.file_root):
    make_dir(args.file_root)

stats_log = open(args.file_root + '/' + args.log_name + '_stats.txt','w') 
test_log = open(args.file_root + '/' + args.log_name + '_acc.txt','w')
test_cls_log = open(args.file_root + '/' + args.log_name + '_acc_cls.txt','w')

if args.dataset == 'imagenet':
    loader = imagenet_dataloader(args.dataset, corrupt_prob=args.corruption_prob, imb_factor=args.imb_factor, noise_mode=args.corruption_type,
                            batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log, args=args)
else:
    loader = cifar_dataloader(args.dataset, corrupt_prob=args.corruption_prob, imb_factor=args.imb_factor, noise_mode=args.corruption_type,
                            batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log, noise_path=args.noise_path, args=args)

with open(os.path.join(args.file_root, 'args.txt'), 'w') as f:
    f.write(str(args))
with open(os.path.join(args.file_root, 'config.txt'), 'w') as f:
    f.write(str(config))

print('| Building net')
net1 = create_model(args=args)
net2 = create_model(args=args)
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
CEloss = nn.CrossEntropyLoss()

test_loader = loader.run('test')  
save_warmup_path = args.file_root + '/'
warm_up_flag = True
eval_flag = False
warmup_trainloader = loader.run('warmup')
num_all_img = len(warmup_trainloader.dataset)
idx_class = []  # index of sample in each class
for i in range(args.num_classes):
    idx_class.append((torch.tensor(warmup_trainloader.dataset.noise_label) == i).nonzero(as_tuple=True)[0])

def main(warm_up_flag):
    best_acc = 0
    prototypes1, prototypes2, prototypes1_bal, prototypes2_bal = None, None, None, None
    if args.MIDL_memory_size and args.train_MIDL> 0:
        MIDL_memory1 = torch.zeros((0, net1.hidden_dim)).cuda()
        MIDL_memory2 = torch.zeros((0, net2.hidden_dim)).cuda()
    else:
        MIDL_memory1 = None
        MIDL_memory2 = None

    if not warm_up_flag:
        net1.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net1.pt'))
        net2.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net2.pt'))

    for epoch in range(args.num_epochs+1):   
        lr=args.lr
        if args.lr_decay:
            if epoch >= 80:
                lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        if  warm_up_flag: 
            if  epoch < warm_up:     
                print('Warmup Net1')
                MIDL_memory1 = warmup_net(epoch, net1, optimizer1, warmup_trainloader, loader.noisy_img_num_list, MIDL_memory1, args)
                print('\nWarmup Net2')
                MIDL_memory2 = warmup_net(epoch, net2, optimizer2, warmup_trainloader, loader.noisy_img_num_list, MIDL_memory2, args)
                if epoch == warm_up -1:
                    torch.save(net1.state_dict(),save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net1.pt') # _1
                    torch.save(net2.state_dict(),save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net2.pt') # _2
                    warm_up_flag = False


                tmp_img_num_list_1 = None
                tmp_img_num_list_2 = None
        else:  
            if epoch < warm_up:
                continue
            else:
                with torch.no_grad():
                    if args.is_prototypical_selection:
                        select_index_1, prob_1, tmp_img_num_list_1, prototypes1 = prototypical_eval_train(loader, net1, epoch, prototypes1, prototypes1_bal, tmp_img_num_list_1, args)  # sample selection from model1
                        select_index_2, prob_2, tmp_img_num_list_2, prototypes2 = prototypical_eval_train(loader, net2, epoch, prototypes2, prototypes2_bal, tmp_img_num_list_2, args)  # sample selection from model2
                    else:
                        select_index_1, prob_1, tmp_img_num_list_1 = bi_dimensional_sample_selection_2(args,net1,loader,epoch,devices=device)    
                        select_index_2, prob_2, tmp_img_num_list_2 = bi_dimensional_sample_selection_2(args,net2,loader,epoch,devices=device)    
                
                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader = loader.run('train',select_index=select_index_2, prob=prob_2) # co-divide
                MIDL_memory2_temp = train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader,
                                                            tmp_img_num_list_2, MIDL_memory=MIDL_memory1, args=args) # train net1
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader = loader.run('train',select_index=select_index_1,prob=prob_1) # co-divide
                MIDL_memory1 = train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader,
                                                  tmp_img_num_list_1, MIDL_memory=MIDL_memory2, args=args) # train net2  
                MIDL_memory2 = MIDL_memory2_temp # without switching memory bank significantly degrade the performance
    
        test_acc = test(epoch, net1, net2)
        if test_acc > best_acc:
            best_acc = test_acc
    
    print('Best Acc:%.3f'%(best_acc))
    test_log.write('Best Acc:%.3f'%(best_acc))

if __name__ == '__main__':
    main(warm_up_flag)
    
