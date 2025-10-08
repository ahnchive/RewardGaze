import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.datasets.vision import VisionDataset

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

def conv3s1(in_planes, out_planes, groups):
    return nn.Conv2d(in_planes, out_planes, groups=groups, 
                     stride=1, kernel_size=3, padding=1, bias=False)    

# class Search(VisionDataset):
#     def __init__(self, root, split, transforms=None):
#         super(Search, self).__init__(root)
#         self.split = split
#         self.transforms = transforms
#         self.src_dir = root + 'images/320x512/'
#         self.tgt_dir = root + 'density/' + split + '/'
        
#         targets = ['bottle', 'chair', 'cup', 'fork', 'laptop', 'microwave', 
#                    'sink', 'stop sign', 'toilet', 'tv', 'bowl', 'car', 
#                    'clock', 'keyboard', 'knife', 'mouse', 'oven', 'potted plant']
#         self.mapping = dict(zip(targets, range(18)))
#         self.tgt = []
#         for target in targets:
#             self.tgt += os.listdir(self.tgt_dir+target)
    
#     def __len__(self):
#         return len(self.tgt)
            
#     def __getitem__(self, index):
#         src = Image.open(self.src[index]).convert('RGB')
#         tgt = Image.open(self.tgt[index])
#         ann = self.ann[index]
#         if self.transforms is not None:
#             src, tgt = self.transforms(src, tgt)
#         return src, tgt, ann
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class RandomResize(object):
    def __init__(self, min_w, max_w, ratio):
        self.min_w = min_w
        self.max_w = max_w
        self.ratio = ratio

    def __call__(self, image, target):
        size = random.randint(self.min_w, self.max_w)
        image = TF.resize(image, (int(size*self.ratio), size))
        target = TF.resize(target, (int(size*self.ratio), size))
        return image, target
    
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = TF.crop(image, *crop_params)
        target = TF.crop(target, *crop_params)
        return image, target
    
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target
    
class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        target = TF.to_tensor(target)
        return image, target
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target
    

class SPP(nn.Module):
    '''spatial pyramid pooling'''
    def __init__(self, in_planes=2048, out_planes=256, sizes=[1, 2, 4, 8]):
        super(SPP, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(sizes[0]),
            nn.Conv2d(in_planes, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(sizes[1]),
            nn.Conv2d(in_planes, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(sizes[2]),
            nn.Conv2d(in_planes, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(sizes[3]),
            nn.Conv2d(in_planes, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(5*out_planes, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.shape[-2:] 
        outs = [] 
        out = self.conv0(x) 
        outs.append(upsample(out, size))
        out = self.conv1(x) 
        outs.append(upsample(out, size))
        out = self.conv2(x) 
        outs.append(upsample(out, size))
        out = self.conv3(x) 
        outs.append(upsample(out, size))
        out = self.conv4(x) 
        outs.append(upsample(out, size))
        
        outs = torch.cat(outs, dim=1)
        return self.conv5(outs)

    
class FPN(nn.Module):
    '''feature pyramid network'''
    def __init__(self, in_planes=2048, out_planes=256, sizes=[1, 2, 4, 8]):
        super(FPN, self).__init__()
        self.spp = SPP(in_planes=in_planes, out_planes=out_planes, sizes=sizes)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes//8, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes//4, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes//2, out_planes, 1, groups=1, bias=False),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        
        self.blend0 = nn.Sequential(
            conv3s1(out_planes, out_planes=out_planes, groups=1),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.blend1 = nn.Sequential(
            conv3s1(out_planes, out_planes=out_planes, groups=1),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )
        self.blend2 = nn.Sequential(
            conv3s1(out_planes, out_planes=out_planes, groups=1),
            nn.GroupNorm(32, out_planes),
            nn.ReLU(inplace=True)
        )  
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, xs):
        outs = []
        out = self.spp(xs[-1]) 
        outs.append(out)
        
        size = xs[2].shape[-2:]
        out = upsample(out, size) + self.conv2(xs[2])
        out = self.blend2(out) 
        outs.append(out)
        size = xs[1].shape[-2:]
        out = upsample(out, size) + self.conv1(xs[1])
        out = self.blend1(out) 
        outs.append(out)
        size = xs[0].shape[-2:]
        out = upsample(out, size) + self.conv0(xs[0])
        out = self.blend0(out) 
        outs.append(out)
        return outs