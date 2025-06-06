import torch
from skimage import measure
import cv2
from utils.func import *
from torch.autograd import Variable 
from Model.SAT import *
from utils.accuracy import *
from DataLoader import *
import argparse
import os
import torch.nn as nn 
from utils.func import *
from Model import *
from sklearn.decomposition import NMF
import clip
from thop import profile
from ptflops import get_model_complexity_info
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from fvcore.nn import FlopCountAnalysis, parameter_count_table

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

def val_loc_one_epoch(args, Val_Loader, model, epoch=0):
    thr_num = len(args.threshold)
    cls_top1 = AverageMeter()
    cls_top5 = AverageMeter()
    IoU30 = [AverageMeter() for i in range(thr_num)]
    IoU50 = [AverageMeter() for i in range(thr_num)]
    IoU70 = [AverageMeter() for i in range(thr_num)]
    loc_gt = [AverageMeter() for i in range(thr_num)]
    loc_top1 = [AverageMeter() for i in range(thr_num)]
    loc_top5 = [AverageMeter() for i in range(thr_num)]
    best_thr_num = 0
    best_loc = 0


    classname_path = '/home/dell/data/dataset/CUB_200_2011/classes.txt'
    classname_list = []
    with open(classname_path, 'r') as f:
        for line in f:
            cls_name = line.strip('\n').split('.')[1]
            cls_name = cls_name.replace('_', ' ')
            classname_list.append(cls_name)

    with torch.no_grad():
        clip_model = '/home/dell/data/code/whb/pretrained/ViT-B-16.pt'
        device = 'cuda'
        encoder, _ = clip.load(clip_model, device=device)

        model.eval()

        for step, (img, img_tencrop, label, gt_boxes, path, pre_lable) in enumerate(Val_Loader):
            if args.evaluate == False:
                return None, None, None 
            img, img_tencrop, label, pre_lable = Variable(img).cuda(), Variable(img_tencrop).cuda(), label.cuda(), pre_lable.cuda()

            new_class_names = []
            for i in range(len(pre_lable)):
                new_class_names.append(classname_list[pre_lable[i]])

            fg_text_features = zeroshot_classifier(new_class_names, ['a photo of a {}.'], encoder)

            output1, map, map2 = model(x=img, label=label, phase='test', text=fg_text_features)

            map = map.data.cpu()
            map = np.array(map.data.cpu())

            batch = label.size(0)
            for i in range(batch):
                map_i = map[i]
                map_i = normalize_map(map_i, args.crop_size)

                gt_boxes_i = gt_boxes[i]
                if args.root == 'Stanford_Dogs' or args.root == 'ILSVRC':
                    gt_boxes_i = gt_boxes[i].strip().split(' ')
                    gt_boxes_i = np.array([float(gt_boxes_i[xxx]) for xxx in range(len(gt_boxes_i))])
                label_i = label[i].unsqueeze(0)
                output1_i = output1[i].unsqueeze(0)
                gt_boxes_i = np.reshape(gt_boxes_i,-1)
                gt_box_num = len(gt_boxes_i) // 4
                gt_boxes_i = np.reshape(gt_boxes_i,(gt_box_num,4))
                
                ##  tencrop_cls
                if args.tencrop:
                    output1_i,_,_ = model(img_tencrop[i], label=None, phase='test', text = fg_text_features[i].unsqueeze(0).expand(10, -1)) #
                    output1_i = nn.Softmax()(output1_i)
                    output1_i = torch.mean(output1_i,dim=0,keepdim=True)
                prec1, prec5 = accuracy(output1_i.data, label_i, topk=(1, 5))
                cls_top1.updata(prec1, 1)
                cls_top5.updata(prec5, 1)
                ##  loc_acc
                for j in range(thr_num):
                    highlight = np.zeros(map_i.shape)
                    highlight[map_i > args.threshold[j]] = 1
                    all_labels = measure.label(highlight)
                    highlight = np.zeros(highlight.shape)
                    highlight[all_labels == count_max(all_labels.tolist())] = 1
                    highlight = np.round(highlight * 255)
                    highlight_big = cv2.resize(highlight, (args.crop_size, args.crop_size), interpolation=cv2.INTER_NEAREST) 
                    props = measure.regionprops(highlight_big.astype(int))
                    best_bbox = [0, 0, args.crop_size, args.crop_size]
                    if len(props) == 0:
                        bbox = [0, 0, args.crop_size, args.crop_size]
                    else:
                        temp = props[0]['bbox']
                        bbox = [temp[1], temp[0], temp[3], temp[2]] 

                    max_iou = -1
                    for m in range(gt_box_num):
                        iou = IoU(bbox, gt_boxes_i[m])

                        if iou > max_iou:
                            max_iou = iou
                            max_box_num = m
                            best_bbox = bbox


                    loc_gt[j].updata(100, 1) if max_iou >= 0.5 else loc_gt[j].updata(0, 1)
                    ##  maxboxaccv2
                    IoU30[j].updata(100, 1) if max_iou >= 0.3 else IoU30[j].updata(0, 1)
                    IoU50[j].updata(100, 1) if max_iou >= 0.5 else IoU50[j].updata(0, 1)
                    IoU70[j].updata(100, 1) if max_iou >= 0.7 else IoU70[j].updata(0, 1) 
                    cls_loc = 100 if prec1 and max_iou >= 0.5 else 0
                    cls_loc_5 = 100 if prec5 and max_iou >= 0.5 else 0
                    loc_top1[j].updata(cls_loc, 1)
                    loc_top5[j].updata(cls_loc_5, 1)
                
                ## save_img
                if args.save_img_flag and ((args.root !='ILSVRC') or (args.root =='ILSVRC' and step<20)):
                    new_path = path[i].replace('\\', '/')
                    ori_img = cv2.imread(new_path)
                    ori_img = cv2.resize(ori_img, (args.resize_size, args.resize_size))
                    shift = (args.resize_size - args.crop_size) // 2
                    if shift > 0:
                        ori_img = ori_img[shift:-shift, shift:-shift, :]
                    heatmap = np.uint8(255 * map_i)
                    img_name = new_path.split('/')[-1]
                    heatmap = heatmap.astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    img_add = cv2.addWeighted(ori_img.astype(np.uint8), 0.5, heatmap.astype(np.uint8), 0.5, 0)

                    if args.save_box_flag:
                        cv2.rectangle(img_add, (int(best_bbox[0]), int(best_bbox[1])),
                                        (int(best_bbox[2]) , int(best_bbox[3])), (0, 255, 0), 4)
                        cv2.rectangle(img_add, (int(gt_boxes_i[max_box_num][0]), int(gt_boxes_i[max_box_num][1])),
                                        (int(gt_boxes_i[max_box_num][2]) , int(gt_boxes_i[max_box_num][3])), (0, 0, 255), 4)

                    if os.path.exists(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.save_img_dir) == 0:
                        os.mkdir(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.save_img_dir)
                    cv2.imwrite(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.save_img_dir + '/' + img_name, img_add)

            if args.root == 'ILSVRC' and (step+1) % (len(Val_Loader)//10) == 0:    
                best_loc = 0
                for j in range(thr_num):
                    if (loc_gt[j].avg + loc_top1[j].avg) > best_loc:
                        best_thr_num = j
                        best_loc = loc_gt[j].avg + loc_top1[j].avg
                print('step:[{}/{}]\t thr: {:.2f}  \t gt_loc : {:.2f}  \t loc_top1 : {:.2f} \t loc_top5 : {:.2f} '.format(
                    step+1, len(Val_Loader), args.threshold[best_thr_num], loc_gt[best_thr_num].avg, loc_top1[best_thr_num].avg, loc_top5[best_thr_num].avg))       
        print('Val Epoch : [{}][{}/{}]  \n'
             'cls_top1 : {:.2f} \t cls_top5 : {:.2f} '.format(epoch, step+1, len(Val_Loader), cls_top1.avg, cls_top5.avg))
        best_loc = 0
        # print(COUNT/5794)
        for j in range(thr_num):
            if (loc_gt[j].avg + loc_top1[j].avg) > best_loc:
                best_thr_num = j
                best_loc = loc_gt[j].avg + loc_top1[j].avg
        print('thr: {:.2f}  \t gt_loc : {:.2f}  \t loc_top1 : {:.2f} \t loc_top5 : {:.2f} \t '.format(args.threshold[best_thr_num], loc_gt[best_thr_num].avg, loc_top1[best_thr_num].avg, loc_top5[best_thr_num].avg))
        print('IoU30: {:.2f}  \t IoU50 : {:.2f}  \t IoU70 : {:.2f} \t Mean : {:.2f}  '.format(IoU30[best_thr_num].avg, IoU50[best_thr_num].avg, IoU70[best_thr_num].avg, (IoU30[best_thr_num].avg + IoU50[best_thr_num].avg + IoU70[best_thr_num].avg)/3))
    return loc_top1[best_thr_num].avg, loc_gt[best_thr_num].avg, args.threshold[best_thr_num]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help="[CUB_200_2011, ILSVRC, Fgvc_aircraft_2013b, Standford_Car, Stanford_Dogs]", 
                                  default='CUB_200_2011')
    parser.add_argument('--num_classes', type=int, default=200)             
    parser.add_argument('--save_path', type=str, default='logs')
    ##  image
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256) 
    ## save
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--tencrop', type=bool, default=False)
    parser.add_argument('--save_img_flag', type=bool, default=True)
    parser.add_argument('--save_box_flag', type=bool, default=False)
    parser.add_argument('--save_img_dir', type=str, default='save_img')

    parser.add_argument('--threshold', type=float, default=[0.3,0.35,0.4,0.45,0.5])
    parser.add_argument('--phase', type=str, default='test')  
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    ##  model
    parser.add_argument('--arch', type=str, help="[deit_sat_small_patch16_224]",
                                  default='deit_sat_small_patch16_224')  
    ##  GPU'                                  
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    ValData = eval(args.root).ImageDataset(args)
    Val_Loader = torch.utils.data.DataLoader(dataset=ValData, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
     
    model = eval(args.arch)(num_classes=args.num_classes, pretrained=False) 
    model = nn.DataParallel(model, device_ids=[int(ii) for ii in range(int(torch.cuda.device_count()))])
    # model=model
    model.cuda(device=0)

    checkpoint = torch.load('/home/dell/data/code/whb/CSFN/logs/CUB_200_2011/FINAL/best_loc.pth.tar')

    if 'module.loc_aug_token' in checkpoint['model'] :
        del checkpoint['model']['module.loc_aug_token']

    model.load_state_dict(checkpoint['model'], strict=False)

    val_loc_one_epoch(args, Val_Loader, model, epoch=0)