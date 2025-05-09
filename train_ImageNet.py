import os
import argparse
import torch
import torch.nn as nn 
from Model.SAT import * 
from DataLoader import *
from torch.autograd import Variable
#from asyncio import trsock
from utils.accuracy import *
from utils.lr import *
from utils.util import copy_dir, makedirs
from utils.optimizer import *
import os
import random
from skimage import measure
import cv2
from utils.func import *
from evaluator import val_loc_one_epoch 
import sys
import pprint
import shutil
from utils.optimizer import create_optimizerv2
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import clip
 
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
parser = argparse.ArgumentParser()
##  path
parser.add_argument('--root', type=str, help="[CUB_200_2011, ILSVRC, OpenImage, Fgvc_aircraft_2013b, Standford_Car, Stanford_Dogs]", 
                                        default='ILSVRC')
parser.add_argument('--num_classes', type=int, default=1000)
##  save
parser.add_argument('--save_path', type=str, default='logs')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--log_code_dir', type=str, default='save_code')
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256) 
##  dataloader
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--weight_decay', type=float, default=5e-2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--opt', type=str, default='adamw')
parser.add_argument('--seed', default=6, type=int) 
##  train
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')
parser.add_argument('--weight_decay_end', type=float, default=None)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--phase', type=str, default='train') 
parser.add_argument('--drop', type=float, default=0.0)  
parser.add_argument('--drop_path', type=float, default=0.1)  
parser.add_argument('--update_freq', default=1, type=int) 
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--area_thr', type=float, default=0.35)  
parser.add_argument('--lr', type=float, default=4.5e-6)#4.5e-6
parser.add_argument('--min_lr', type=float, default=1e-6)
##  model
parser.add_argument('--arch', type=str, default='deit_sat_small_patch16_224')  
##  evaluate 
parser.add_argument('--save_img_flag', type=bool, default=True)
parser.add_argument('--save_error_flag', type=bool, default=False)
parser.add_argument('--tencrop', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)  
parser.add_argument("--local_rank", type=int,default=-1)
##  GPU'
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lr = args.lr 
## save_log_txt
makedirs(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir)
sys.stdout = Logger(args.save_path + '/' + args.root + '/'  + args.arch +  '/' + args.log_code_dir + '/' + args.log_file) 
sys.stdout.log.flush()

##  save_code
save_file = ['train.py', 'evaluator.py', 'evaluator_ImageNet.py', 'train_ImageNet.py']
for file_name in save_file:
    shutil.copyfile(file_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + file_name)
save_dir = ['Model', 'utils', 'DataLoader']
for dir_name in save_dir:
    copy_dir(dir_name, args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + dir_name)

set_seed(args.seed)

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()

if __name__ == '__main__':
    ##  dataloader
    args.batch_size  = args.batch_size // torch.cuda.device_count()
    TrainData = eval(args.root).ImageDataset(args, phase='train')
    Train_Loader = torch.utils.data.DataLoader(dataset=TrainData, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)  #

    ##  model
    model = eval(args.arch)(num_classes=args.num_classes, drop_rate=args.drop, drop_path_rate=args.drop_path, pretrained=True)
    model = nn.DataParallel(model, device_ids=[int(ii) for ii in range(int(torch.cuda.device_count()))])
    model.cuda(device=0)

    total_batch_size = args.batch_size * int(torch.cuda.device_count()) * args.update_freq
    num_training_steps_per_epoch = len(TrainData) // total_batch_size
    ##  lr
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch) 
    ##  optimizer 
    optimizer = create_optimizerv2(args, model)
    loss_fnc = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    best_gt, best_top1, best_loc = 0, 0, 0

    clip_model = '/home/dell/data/code/whb/pretrained/ViT-B-16.pt'
    device = 'cuda'
    encoder, _ = clip.load(clip_model, device=device)

    classname_path = '/home/dell/data/dataset/ImageNet_2012/imagenet-classes.txt'
    classname_list = []
    with open(classname_path, 'r') as f:
        for line in f:
            cls_name = line.strip('\n')
            if ',' in cls_name:
                cls_name = cls_name.split(',')[0]
            classname_list.append(cls_name)

    for epoch in range(0, args.epochs):
        ##  accuracy
        cls_acc_1 = AverageMeter()
        loss_epoch_1 = AverageMeter()
        loss_epoch_2 = AverageMeter()
        loss_epoch_3 = AverageMeter() 
        model.train()

        for step, (path, imgs, label, pos_imgs) in enumerate(Train_Loader):
            imgs, label, pos_imgs = Variable(imgs).cuda(), label.cuda(), Variable(pos_imgs).cuda()

            new_class_names = []
            for i in range(len(label)):
                new_class_names.append(classname_list[label[i]])
            fg_text_features = zeroshot_classifier(new_class_names, ['a photo of a {}.'], encoder)

            label = torch.cat((label, label), dim=0)

            optimizer.zero_grad()

            ##  loss
            area_thr = args.area_thr

            output1, mask_all, norm_loss, x_loc, x_loc_aug, mask_aug_all, norm_loss_aug \
                 , x_patch, Fuse_map, mul_output \
                = model(imgs, pos_x=pos_imgs, phase='train', epoch=epoch, text=fg_text_features)#

            ba_loss = mask_all.view(label.shape[0], -1).mean(-1)
            ba_loss, norm_loss = ba_loss.mean(0), norm_loss.mean(0)

            ba_loss_aug = mask_aug_all.view(label.shape[0], -1).mean(-1)
            ba_loss_aug, norm_loss_aug = ba_loss_aug.mean(0), norm_loss_aug.mean(0)

            loss_OT = loss_mse(Fuse_map.unsqueeze(1).detach(), mask_all).cuda()

            loss_clip = loss_mse(mask_all.detach(), mask_aug_all).cuda()

            loss_token = loss_mse(x_loc.detach(), x_loc_aug).cuda()

            loss_same = loss_mse(mask_all.repeat(1, 1000, 1, 1).detach(), x_patch).cuda()

            loss_cls = loss_fnc(output1, label).cuda()
            loss_cls_mul1 = loss_fnc(mul_output[0], label).cuda()
            loss_cls_mul4 = loss_fnc(mul_output[1], label).cuda()
            loss_cls_mul9 = loss_fnc(mul_output[2], label).cuda()
            loss_cls = loss_cls + 1.0 * (loss_cls_mul1 + loss_cls_mul4 + loss_cls_mul9) #2.0

            loss = loss_cls \
                   + torch.abs(ba_loss - area_thr).mean(0) + norm_loss \
                   + torch.abs(ba_loss_aug - area_thr).mean(0) + norm_loss_aug \
                   + 0.0 * loss_clip + 0.0 * loss_token + 0.1 * loss_OT + 0.1 * loss_same

            loss.backward()
            optimizer.step()
            ##  count_cls_accuracy
            cur_batch = label.size(0)            
            cur_cls_acc_1 = 100. * compute_cls_acc(output1, label)
            cls_acc_1.updata(cur_cls_acc_1, cur_batch)
            loss_epoch_1.updata(loss_cls.data, 1)
            loss_epoch_2.updata(torch.abs(ba_loss).mean(0).data, 1)
            loss_epoch_3.updata(norm_loss.data, 1)

        print('Epoch:[{}/{}]\tstep:[{}/{}]\tloss_epoch_1:{:.3f}\tloss_epoch_2:{:.3f}\tloss_epoch_3:{:.3f}\tepoch_acc:{:.2f}%'.format(
                        epoch+1, args.epochs, step+1, len(Train_Loader), loss_epoch_1.avg,loss_epoch_2.avg,loss_epoch_3.avg,cls_acc_1.avg
                ))
        sys.stdout.log.flush()
        torch.save({'model':model.state_dict(),
                    'best_thr':0,
                    'epoch':epoch+1,
                    }, os.path.join(args.save_path, args.root + '/' + args.arch + '/' + str(epoch) +'best_loc.pth.tar'), _use_new_zipfile_serialization=False)


        
