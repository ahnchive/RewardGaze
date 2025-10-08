from PIL import Image
import torchvision.transforms as T
import torch
from torchvision.models import resnet50
import os
from os.path import join, isdir, isfile
import gzip
import numpy as np
from torch import nn
from tqdm import tqdm


device = 'cuda:0'
img_path = "./ISVN/img_features/"
target_path = "./ISVN/target_features/"
setting = 'TA'

fix_clusters = np.load('/data/add_disk1/zbyang/datasets/coco_search18/clusters.npy', allow_pickle=True).item()
task_img_pairs = list(fix_clusters.keys())
if setting == 'TP':
    task_img_pairs = list(filter(lambda x: x.split('-')[1] == 'present', task_img_pairs))
else: 
    task_img_pairs = list(filter(lambda x: x.split('-')[1] == 'absent', task_img_pairs))
    
target_im_h, target_im_w = 320, 512
im_h, im_w = 10, 16
patch_size = 32
L = 6 if setting == 'TP' else 10
pred_dicts = []
output_file = 'pred_TP_IVSN.npy' if setting == 'TP' else 'pred_TA_IVSN.npy'

for pair in tqdm(task_img_pairs):
    cond, task, img_name = pair.split('-')[-3:]
    task = task.replace(' ', '_')
    with gzip.GzipFile(join(img_path, task, img_name + '.npy.gz'), "r") as r:
        img_ftrs = np.transpose(np.load(r, allow_pickle=True), (1, 2, 0))
        r.close()

    with gzip.GzipFile(join(target_path, task + '.npy.gz'), "r") as r:
        task_ftrs = np.load(r, allow_pickle=True).reshape(-1, 1)
        r.close()

    att_map = img_ftrs @ task_ftrs
    att_map = att_map[:, :, 0]
    tmp_map = np.copy(att_map)
    
    pred_actions = []

    for l in range(L):
        if l > 0:
            att_map[pred_actions[l - 1]] = 0.  #IOR
        pred_actions.append(np.unravel_index(att_map.argmax(),
                                             att_map.shape))  #WTA

    #postprocess
    pred_traj_X = [target_im_w // 2]
    pred_traj_Y = [target_im_h // 2]

    for l in range(len(pred_actions)):
        y, x = pred_actions[l]
        pred_traj_X.append(
            (x / im_w * target_im_w) + patch_size // 2)
        pred_traj_Y.append(
            (y / im_h * target_im_h) + patch_size // 2)

    pred_dicts.append({
        'name': img_name + '.jpg',
        'task': task,
        'X': pred_traj_X,
        'Y': pred_traj_Y,
        'length': len(pred_traj_Y),
        'attn_map': tmp_map
    })

with open(output_file, 'wb') as f:
    np.save(f, pred_dicts, allow_pickle=True)
    f.close()
