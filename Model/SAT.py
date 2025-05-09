import torch
import torch.nn as nn
from functools import partial
from utils.cfgs import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from utils.accuracy import *
from utils.func import *
from skimage import measure
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torch.nn.parameter import Parameter


__all__ = [
    'deit_sat_tiny_patch16_224', 'deit_sat_small_patch16_224', 'deit_sat_base_patch16_224',
]


def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=1,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    #C为代价矩阵M
    if not warm:
        a = torch.ones((C.shape[0],1)) / C.shape[0]
        a = a.cuda()

    K = torch.exp(-C/epsilon)

    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.mm(K.t(), a) #torch.mm 矩阵相乘
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.mm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI

def get_kernel(kernlen=3, nsig=6):    
    interval = (2*nsig+1.)/kernlen  
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)                                 
    kern1d = np.diff(st.norm.cdf(x))    
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))   
    kernel = kernel_raw/kernel_raw.sum()          
    return kernel

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim   
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.loc_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.loc_embed = nn.Parameter(torch.zeros(1,  1, embed_dim))

        self.loc_aug_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.loc_aug_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.loc_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.loc_token, std=.02)

        trunc_normal_(self.loc_aug_embed, std=.02)
        trunc_normal_(self.loc_aug_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'loc_embed', 'loc_token'} 

class CSFN(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)

        kernel = get_kernel(kernlen=3,nsig=6)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        self.fc = nn.Sequential(nn.Linear(512, self.embed_dim), )
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())
        self.head_new = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=16, dim_feedforward=2048, activation="gelu",
                                                   dropout=0.1, batch_first=False)
        self.Cross_image_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        loc_tokens = self.loc_token.expand(B, -1, -1)

        loc_aug_tokens = self.loc_aug_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x, loc_tokens, loc_aug_tokens), dim=1)#


        pos_embed = torch.cat([self.pos_embed, self.loc_embed, self.loc_aug_embed], 1)

        x = x + pos_embed
        x = self.pos_drop(x) 
        mask_all = []
        mask_aug_all = []
        mask_fuse_all = []

        for cur_depth, blk in enumerate(self.blocks):
            x, mask, mask_fuse, mask_aug = blk(x, cur_depth)# #

            mask_all.append(mask)
            mask_aug_all.append(mask_aug)
            mask_fuse_all.append(mask_fuse)


        x = self.norm(x)
        return x[:, 0], x[:, -2], x[:, 1:-2],  mask_all, x[:, -1], mask_aug_all, mask_fuse_all

    def forward(self, x, pos_x=None, label=None, phase=None, epoch=0, text = None):
        batch = x.size(0)
        if text is not None:
            if phase == 'train':
                text = torch.cat((text,text),dim=0)
            text = text.float()#.squeeze(1)
            text = self.fc(text).unsqueeze(1)
            text = nn.Parameter(text)
            self.loc_aug_token = text

        if phase == 'train':
            x = torch.cat((x,pos_x),dim=0)
            batch = x.size(0)

        x_cls, x_loc, x_patch, mask_all, x_loc_aug, mask_aug_all, mask_fuse_all = self.forward_features(x)

        n, p, c = x_patch.shape

        mask_all = torch.stack(mask_all)
        mask_all = mask_all[-3:,:,:,:,1:-2]
        mask_all = torch.mean(mask_all, dim=2)
        mask_all = torch.mean(mask_all, dim=0)
        mask_all = mask_all.reshape(batch,1,14,14)

        mask_aug_all = torch.stack(mask_aug_all)
        mask_aug_all = mask_aug_all[-3:, :, :, :, 1:-2]
        mask_aug_all = torch.mean(mask_aug_all, dim=2)

        mask_aug_all = torch.mean(mask_aug_all, dim=0)
        mask_aug_all = mask_aug_all.reshape(batch, 1, 14, 14)

        mask_fuse_all = torch.stack(mask_fuse_all)
        mask_fuse_all = mask_fuse_all[-3:, :, :, :, 1:-2]
        mask_fuse_all = torch.mean(mask_fuse_all, dim=2)

        mask_fuse_10th = mask_fuse_all[0]
        mask_fuse_11th = mask_fuse_all[1]
        mask_fuse_12th = mask_fuse_all[2]


        mask_nmf = torch.cat((mask_fuse_10th, mask_fuse_11th, mask_fuse_12th), dim=1)#

        mask_fuse_all = torch.mean(mask_fuse_all, dim=0)
        mask_fuse_all = mask_fuse_all.reshape(batch, 1, 14, 14)

        B, P, D = x_patch.shape
        W = H = int(P ** 0.5)
        x0 = x_cls
        x_p = x_patch.view(B, W, H, D).permute(0, 3, 1, 2)

        x10, x11, x12, x13 = self.aggregation(x_p[:, :, 0:8, 0:8]), self.aggregation(
            x_p[:, :, 0:8, 8:]), self.aggregation(x_p[:, :, 8:, 0:8]), self.aggregation(x_p[:, :, 8:, 8:])
        x20, x21, x22, x23, x24, x25, x26, x27, x28 = self.aggregation(x_p[:, :, 0:5, 0:5]), self.aggregation(
            x_p[:, :, 0:5, 5:11]), self.aggregation(x_p[:, :, 0:5, 11:]), \
                                                      self.aggregation(x_p[:, :, 5:11, 0:5]), self.aggregation(
            x_p[:, :, 5:11, 5:11]), self.aggregation(x_p[:, :, 5:11, 11:]), \
                                                      self.aggregation(x_p[:, :, 11:, 0:5]), self.aggregation(
            x_p[:, :, 11:, 5:11]), self.aggregation(x_p[:, :, 11:, 11:])

        x_mul = [i.unsqueeze(1) for i in [x0, x10, x11, x12, x13, x20, x21, x22, x23, x24, x25, x26, x27, x28]]
        x_mul = torch.cat(x_mul, dim=1)
        x_mul = self.Cross_image_encoder(x_mul).view(B, 14 * D)
        x_mul = torch.nn.functional.normalize(x_mul, p=2, dim=-1)
        x_mul = torch.reshape(x_mul, [B, 14, D])
        x_mul_1 = x_mul[:, 0, :].unsqueeze(1).permute(0, 2, 1).view(B, D, 1, 1)
        x_mul_4 = x_mul[:, 1:5, :].permute(0, 2, 1).view(B, D, 2, 2)
        x_mul_9 = x_mul[:, 5:, :].permute(0, 2, 1).view(B, D, 3, 3)
        x_mul_1 = self.head_new(x_mul_1)
        x_mul_4 = self.head_new(x_mul_4)
        x_mul_9 = self.head_new(x_mul_9)
        x_mul_logits = []
        x_mul_logits.append(self.avgpool(x_mul_1).squeeze(3).squeeze(2))
        x_mul_logits.append(self.avgpool(x_mul_4).squeeze(3).squeeze(2))
        x_mul_logits.append(self.avgpool(x_mul_9).squeeze(3).squeeze(2))

        if phase == 'train':
            lenth = int(batch/2)
            src_feature = mask_nmf[0:lenth,:,:].mean(1).unsqueeze(1)
            # B*1*196
            tar_feature = mask_nmf[lenth:,:,:].mean(1).unsqueeze(1)

            Fuse_map_src = []
            Fuse_map_tar = []
            for i in range(lenth):
                src_weights = 0.5 * torch.ones(196, 1).cuda()
                # 阶梯划分4个等级
                src_weights[mask_fuse_all[i].view(-1) > 0.4, :] = 0.8
                src_weights[mask_fuse_all[i].view(-1) > 0.5, :] = 0.9
                src_weights[mask_fuse_all[i].view(-1) > 0.6, :] = 1.0
                mu = src_weights / src_weights.sum()  # mu_s

                trg_weights = 0.5 * torch.ones(196, 1).cuda()
                # 阶梯划分4个等级
                trg_weights[mask_fuse_all[i + lenth].view(-1) > 0.4, :] = 0.8
                trg_weights[mask_fuse_all[i + lenth].view(-1) > 0.5, :] = 0.9
                trg_weights[mask_fuse_all[i + lenth].view(-1) > 0.6, :] = 1.0
                nu = trg_weights / trg_weights.sum()  # mu_t

                exp2 = 1.0
                eps = 0.5

                src_feat_norms = F.normalize(src_feature[i], p=2, dim=1, eps=1e-8)
                trg_feat_norms = F.normalize(tar_feature[i], p=2, dim=1, eps=1e-8)
                sim = torch.matmul(src_feat_norms.transpose(-1, -2), trg_feat_norms)
                sim = torch.pow(torch.clamp(sim, min=0), 1.0)
                cost = 1 - sim  # 代价矩阵M

                epsilon = eps
                cnt = 0
                epsilon1 = eps
                cnt1 = 0
                while True:
                    T = perform_sinkhorn(cost, epsilon, mu, nu)
                    if not torch.isnan(T).any():
                        if cnt > 0:
                            print(cnt)
                        break
                    else:  # Nan encountered caused by overflow issue is sinkhorn
                        epsilon *= 2.0
                        cnt += 1
                T = 196.0 * T  # rx_mul_logitse-scale PI
                T = torch.pow(torch.clamp(T, min=0), exp2)
                T = T * sim
                T = T.softmax(dim=1) # 196*196
                T_M = torch.matmul(T, tar_feature[i].transpose(-1, -2)).squeeze(-1).reshape([14, 14])

                src_map = src_feature[i].reshape([1, 14, 14]).squeeze(0)
                fuse_map = []
                fuse_map.append(src_map)
                fuse_map.append(T_M)
                fuse_map = torch.stack(fuse_map)
                fuse_map = torch.max(fuse_map, dim=0)[0]
                Fuse_map_src.append(fuse_map)

                while True:
                    _T = perform_sinkhorn(cost.t(), epsilon1, nu, mu)
                    if not torch.isnan(_T).any():
                        if cnt1 > 0:
                            print(cnt1)
                        break
                    else:  # Nan encountered caused by overflow issue is sinkhorn
                        epsilon1 *= 2.0
                        cnt1 += 1
                _T = 196.0 * _T  # re-scale PI
                _T = torch.pow(torch.clamp(_T, min=0), exp2)
                _T = _T * sim.t()
                _T = _T.softmax(dim=1) # 196*196
                _T_M = torch.matmul(_T, src_feature[i].transpose(-1, -2)).squeeze(-1).reshape([14, 14])

                tar_map = tar_feature[i].reshape([1, 14, 14]).squeeze(0)
                _fuse_map = []
                _fuse_map.append(tar_map)
                _fuse_map.append(_T_M)
                _fuse_map = torch.stack(_fuse_map)
                _fuse_map = torch.max(_fuse_map, dim=0)[0]
                Fuse_map_tar.append(_fuse_map)
            Fuse_map_src = torch.stack(Fuse_map_src).detach()
            Fuse_map_tar = torch.stack(Fuse_map_tar).detach()
            Fuse_map = torch.cat((Fuse_map_src,Fuse_map_tar),dim=0)


        x_patch = torch.reshape(x_patch , [n, int(p**0.5), int(p**0.5), c])  
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)

        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        mask_avg = mask_all.clone()
        mask_avg = F.conv2d(mask_avg, self.weight, padding=1)

        mask_avg_aug = mask_aug_all.clone()
        mask_avg_aug = F.conv2d(mask_avg_aug, self.weight, padding=1)

        if phase == 'train':
            return x_logits, mask_all, ((1-mask_avg)*mask_avg).view(batch,-1).mean(-1)\
                , x_loc, x_loc_aug, mask_aug_all, ((1-mask_avg_aug)*mask_avg_aug).view(batch,-1).mean(-1)\
                , x_patch, Fuse_map, x_mul_logits

        else:
            n, c, h, w = x_patch.shape

            mask_fuse_all = mask_fuse_all.reshape([n, h, w])
            mask_all = mask_all.reshape([n, h, w])
            mask_aug_all = mask_aug_all.reshape([n, h, w])


            return x_logits, mask_fuse_all, mask_nmf


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cur_depth=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask = nn.Sigmoid()(attn[:, :, -2, :].unsqueeze(2).mean(1).unsqueeze(1))
        mask_aug = nn.Sigmoid()(attn[:, :, -1, :].unsqueeze(2).mean(1).unsqueeze(1))
        mask_fuse = torch.cat((mask, mask_aug), dim=1)
        mask_fuse = torch.max(mask_fuse, dim=1)[0].unsqueeze(1)

        attn = attn.softmax(dim=-1)
        if cur_depth >= 9: # >=9
            attn = attn * mask_fuse

         
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, mask, mask_fuse, mask_aug#, weights


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, vis=vis)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, cur_depth=None):
        o, mask, mask_fuse, mask_aug = self.attn(self.norm1(x), cur_depth=cur_depth)#,  mask

        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, mask, mask_fuse, mask_aug#, weights

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
 

@register_model
def deit_sat_tiny_patch16_224(pretrained=False, **kwargs):
    model = CSFN(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def deit_sat_small_patch16_224(pretrained=False, **kwargs):
    model = CSFN(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    return model


@register_model
def deit_sat_base_patch16_224(pretrained=False, **kwargs):
    model = CSFN(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)