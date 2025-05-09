import torch
import os
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import pandas as pd
import json

def load_test_bbox(root, test_gt_path,crop_size,resize_size):
    test_gt = []
    test_txt = []
    shift_size = (resize_size - crop_size) // 2
    with open(test_gt_path, 'r') as f:
        
        for line in f:
            temp_gt = []
            part_1, part_2 = line.strip('\n').split(';')
            img_path, w, h, _ = part_1.split(' ')
            part_2 = part_2[1:]
            bbox = part_2.split(' ')
            bbox = np.array(bbox, dtype=np.float32)
            box_num = len(bbox) // 4
            w, h = np.float32(w),np.float32(h)
            for i in range(box_num):
                bbox[4*i] = int(max(bbox[4*i] / w * resize_size - shift_size, 0))
                bbox[4*i+1] = int(max(bbox[4*i+1] / h * resize_size - shift_size, 0))
                bbox[4*i+2] = int(min(bbox[4*i+2] / w * resize_size - shift_size, crop_size - 1))
                bbox[4*i+3] = int(min(bbox[4*i+3] / h * resize_size - shift_size, crop_size - 1))
                temp_gt.append([bbox[4*i], bbox[4*i+1], bbox[4*i+2], bbox[4*i+3]])
            test_gt.append(temp_gt)
            img_path = img_path.replace("\\\\","\\")
            test_txt.append(img_path)
    final_dict = {}
    for k, v in zip(test_txt, test_gt):
        k = os.path.join(root, 'val', k)
        k = k.replace('/', '\\')
        final_dict[k] = v
    return final_dict

class ImageDataset(data.Dataset):
    def __init__(self, args, phase=None):
        args.num_classes = 1000
        self.args =args
        self.load_class_path = './Moduel/imagenet_efficientnet-b7_3rdparty_8xb32-aa-advprop_in1k.json'
        self.root = '/home/dell/data/dataset/ImageNet_2012/'
        self.test_txt_path = self.root + 'val_list.txt'
        self.test_gt_path = self.root + 'val_gt.txt'
        self.crop_size = args.crop_size
        self.resize_size = args.resize_size
        self.phase = args.phase if phase == None else phase
        self.num_classes = args.num_classes
        self.tencrop = args.tencrop
        if self.phase == 'train':
            self.img_dataset = ImageFolder(os.path.join(self.root, 'train'))
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        elif self.phase == 'test':
            self.img_dataset = ImageFolder(os.path.join(self.root, 'val'))
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            self.transform_tencrop = transforms.Compose([
                    transforms.Resize((self.resize_size, self.resize_size)),
                    transforms.TenCrop(args.crop_size),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(crop) for crop in crops])),
                ]) 
        self.img_dataset = self.img_dataset.imgs    
        self.test_bbox = load_test_bbox(self.root, self.test_gt_path,self.crop_size,self.resize_size)
        # new add
        self.pd_file = pd.read_csv("/home/dell/data/dataset/ImageNet_2012/train.txt", sep=" ", header=None,
                                   names=['ImageName', 'label'])

        with open(self.load_class_path, 'r') as f:
            self.name2result = json.load(f)

    def __getitem__(self, index):
        path, img_class = self.img_dataset[index]
          
        img = Image.open(path).convert('RGB')
        img_trans = self.transform(img)
        if self.phase == 'train':
            positive_image = self.fetch_positive(1, img_class, path.split('train')[-1])[0]
            return path, img_trans, img_class, positive_image
        else:
            name = path.split('val/')[1][0:-5]
            path = path.replace('/', '\\')
            bbox = self.test_bbox[path]
            bbox = np.array(bbox).reshape(-1)
            bbox = " ".join(list(map(str, bbox)))

            pre_lable = self.name2result[name]['pred_label']
            if self.tencrop:
                img_tencrop = self.transform_tencrop(img) 
                return img_trans, img_tencrop, img_class, bbox, path, pre_lable
            return img_trans, img_trans, img_class, bbox, path, pre_lable

    def __len__(self):
        return len(self.img_dataset)

    def fetch_positive(self, num_positive, label, path): #正样本数, 正样本类别标签, txt中源图像路径名
        path = path[1:]
        other_img_info = self.pd_file[(self.pd_file.label == label) & (self.pd_file.ImageName != path)]
        other_img_info = other_img_info.sample(min(num_positive, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join('/home/dell/data/dataset/ImageNet_2012/train', e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        positive_img = [self.transform(img) for img in other_img]
        return positive_img

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
