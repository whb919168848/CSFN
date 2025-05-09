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

def load_test_bbox(root, test_txt_path, test_gt_path,resize_size, crop_size):
    test_gt = []
    test_txt = []
    shift_size = (resize_size - crop_size) // 2.
    with open(test_txt_path, 'r') as f:
        for line in f:
            img_path = line.strip('\n').split(';')[0]
            test_txt.append(img_path)
    with open(test_gt_path, 'r') as f:        
        for line in f:
            x0, y0, x1, y1, h, w = line.strip('\n').split(' ')
            x0, y0, x1, y1, h, w = float(x0), float(y0), float(x1), float(y1), float(h), float(w)
            x0 = int(max(x0 / w * resize_size - shift_size, 0))
            y0 = int(max(y0 / h * resize_size - shift_size, 0))
            x1 = int(min(x1 / w * resize_size - shift_size, crop_size - 1))
            y1 = int(min(y1 / h * resize_size - shift_size, crop_size - 1))
            test_gt.append(np.array([x0, y0, x1, y1]).reshape(-1))
    final_dict = {}
    for k, v in zip(test_txt, test_gt):
        k = os.path.join(root, 'test', k)
        k = k.replace('/', '\\')
        final_dict[k] = v
    return final_dict

class ImageDataset(data.Dataset):
    def __init__(self, args ,phase=None):
        args.num_classes = 200
        self.args =args
        self.load_class_path = './Model/cub_efficientnetb7.json'
        self.root = '/home/dell/data/dataset/CUB_200_2011'
        self.test_txt_path = self.root + '/' + 'test_list.txt'
        self.test_gt_path = self.root + '/' + 'test_bounding_box.txt'
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
            self.img_dataset = ImageFolder(os.path.join(self.root, 'test'))
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
        self.test_bbox = load_test_bbox(self.root, self.test_txt_path, self.test_gt_path,self.resize_size, self.crop_size)

        #new add
        self.pd_file = pd.read_csv("/home/dell/data/dataset/CUB_200_2011/cub_bird_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])

        with open(self.load_class_path, 'r') as f:
            self.name2result = json.load(f)

    def __getitem__(self, index):
        path, img_class = self.img_dataset[index]
        img = Image.open(path).convert('RGB')      
        img_trans = self.transform(img)  
        if self.phase == 'train':
            # new add
            positive_image = self.fetch_positive(1, img_class, path.split('train')[-1])[0]
            return path, img_trans, img_class, positive_image
        else:
            name = path.split('test')[1][1:]

            path = path.replace('/', '\\')
            bbox = self.test_bbox[path]

            pre_lable = self.name2result[name]['pred_label']
            if self.tencrop:
                img_tencrop = self.transform_tencrop(img) 
                return img_trans, img_tencrop, img_class, bbox, path, pre_lable
            return img_trans, img_trans, img_class, bbox, path, pre_lable

    def __len__(self):
        return len(self.img_dataset)

    def fetch_positive(self, num_positive, label, path): #正样本数, 正样本类别标签, txt中源图像路径名
        path = 'images' + path
        other_img_info = self.pd_file[(self.pd_file.label == label) & (self.pd_file.ImageName != path)]
        other_img_info = other_img_info.sample(min(num_positive, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join('/home/dell/data/dataset/CUB_200_2011', e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        positive_img = [self.transform(img) for img in other_img]
        return positive_img

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

