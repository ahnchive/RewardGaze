# this program generates reward maps under save dir folder. run the following command run this file
# python generate_rewardmap.py --dataset-root '/home/young/hdd1/coco-search18/' --TAP 'TPtoTA' or 'TA' --map-type 'softmax_return' or 'return' or 'immediate_reward' --save-dir './results' 
# e.g., python generate_rewardmap.py --dataset-root '/home/young/hdd1/coco-search18/' --TAP 'TPtoTA' --map-type 'softmax_return' --save-dir './results' 

import sys
import argparse
from os.path import join
import copy
import random
import numpy as np
import pandas as pd 
from scipy import stats


import torch
import torch.nn.functional as F

from common.config import JsonConfig
from ffm_iql.builder import build 
from ffm_iql.evaluation import evaluate, sample_scanpaths

import matplotlib.pyplot as plt
import cv2
from PIL import ImageFont, Image, ImageDraw
font = ImageFont.truetype("Gidole-Regular.ttf", size=32)
from fixation_density import *




def normalize_map(m, softmax=False, alpha=0.05):
    if softmax:
        # softmax with alpha
        m = torch.Tensor(m).flatten()
        m = F.softmax(m/alpha)
        m = np.reshape(m.numpy(), (320, 512))
        #         m = np.exp(m/alpha) / np.sum(np.exp(m/alpha)) 

    # min-max normalization
    m = (m - m.min()) / (m.max() - m.min()) * 255
    m = m.astype(np.uint8)
    return m

def overlay_map(m, img):
    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    m = cv2.addWeighted(m, 0.5, img, 0.5, 0)
    return m

def save_map(m, img, overlay, path_savefile):
    if overlay:
        m = overlay_map(m, img)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        m = Image.fromarray(m)
        m.save(path_savefile)
    else:
        m = Image.fromarray(m).convert("L")
        m.save(path_savefile)
    

def check_directory_and_ask_permission_for_overwrite(SAVE_DIR):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)  
        print(f"Directory {SAVE_DIR} created")
    else:
        overwrite = input(f"The save folder {SAVE_DIR} already exists. Do you want to overwrite it? (yes/no): ").lower().strip()
        if overwrite == 'yes':
            pass
        else:
            print("Please change the save directory name")
            sys.exit()
        

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset-root', help="where coco-search18 dataset", type= str, required=True) #  '/home/young/hdd1/coco-search18/'
    parser.add_argument('--TAP', type=str, required=True)
    parser.add_argument('--map-type', type=str, required=True)
    # parser.add_argument('--overlay-to-image', type=int, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    
    #########
    # set args
    #########
    args = parser.parse_args()
    DATASET_ROOT = args.dataset_root
    TAP = args.TAP
    MAP_TYPE = args.map_type

    SAVE_DIR_OVERLAY = args.save_dir  + f'/visualize_{MAP_TYPE}_overlaid_{TAP}' 
    SAVE_DIR_MAPONLY = args.save_dir  + f'/visualize_{MAP_TYPE}_{TAP}'
    check_directory_and_ask_permission_for_overwrite(SAVE_DIR_OVERLAY)
    check_directory_and_ask_permission_for_overwrite(SAVE_DIR_MAPONLY)
    
    print(f"Generating reward maps for {TAP} with {MAP_TYPE}...")
    print(f"Visualizations will be saved in {SAVE_DIR_OVERLAY} and {SAVE_DIR_MAPONLY}")

    #######
    # set device and random seed
    #######
    device = torch.device('cuda:0') #torch.device('cpu')
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #######
    # load hparams config files
    #######
    hparams = JsonConfig('./configs/coco_search18_eDCB_IQL_TP_LIOR.json')
    # hparams = JsonConfig('./configs/coco_search18_eDCB_IQL_TA_LIOR.json')
    hparams_tp = JsonConfig('./configs/coco_search18_TP.json')
    hparams_ta = JsonConfig('./configs/coco_search18_TA.json')
    hparams_fv = JsonConfig('./configs/coco_search18_FV.json')

    if TAP=='TPtoTA':
        # change hparam to TA setup
        hparams_ta_training = JsonConfig('./configs/coco_search18_eDCB_IQL_TA_LIOR.json')
        maxtrajcopy = hparams.Data.max_traj_length
        hparams.Data = hparams_ta_training.Data
        hparams.Data.max_traj_length =maxtrajcopy
        hparams_ta.Data.max_traj_length =maxtrajcopy
        
    #######
    # build model
    #######
    print('\n====Building model and env...====')
    agent, train_gaze_loader, train_img_loader, valid_img_loader,\
        valid_img_loader_ta, valid_img_loader_fv, env, env_valid, global_step, bbox_annos,\
        human_cdf, fix_clusters, prior_maps_tp, prior_maps_ta, prior_maps_fv,\
        sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta, valid_gaze_loader_fv,\
        sps_test_tp, sps_test_ta, sps_test_fv = build(
            hparams, DATASET_ROOT, device, load_coco_annos=False)


    ########
    # run visualizations (code modified from evalaute in evaluation.py)
    ########
    env = env_valid
    model = agent

    if hparams.Data.TAP == 'TP':
        dataloader = valid_img_loader # train_img_loader for using training image
        pa= hparams_tp.Data
    #     pa['enforce_IOR'] = False
    #     print(pa)
    if hparams.Data.TAP == 'TA':
        dataloader =  valid_img_loader_ta
        pa= hparams_ta.Data
    #     pa['enforce_IOR'] = False

    model.eval()

    scanpaths, visualizations = sample_scanpaths(env, model, dataloader, pa,
                                                sample_action=False, 
                                                sample_scheme='Greedy', # this is not necessary
                                                sample_stop=False,
                                                save_visualization=True)
    

    ########
    # visualize reward map (overlaid with and without image)
    ########
    print(f'\n===reward maps generated for {len(visualizations)} images...')
    for sp, vis in zip(scanpaths, visualizations):
        task_name = vis[0] 
        image_name = vis[1]
        task_image_label = '_'.join([task_name, image_name])

        # load image
        # print(image_name)
        imagefile = join(DATASET_ROOT, 'images/{}/{}'.format(task_name.replace(' ', '_'), image_name))
        img= Image.open(imagefile)
        if img.size[0] != hparams.Data.im_w:
            img = img.resize((hparams.Data.im_w, hparams.Data.im_h))
        img_array = np.array(img)
        
        # load model fixation prediction
        fixs = list(zip(sp['X'], sp['Y']))
        n_step = len(fixs)
        
        # draw and save reward map for each fixation step
        maps = []
        for step in range(n_step-1):
            
            if MAP_TYPE == 'return':
                map = vis[2][step] 
            elif MAP_TYPE == 'softmax_return': # softmax return
                map = vis[4][step]  
            elif MAP_TYPE == 'immediate_reward': # immediate reward
                # alpha = hparams.Train['init_temp']
                # reward = vis[3][step] 
                map = vis[2][step] - vis[3][step] #return_minus_reward
                map = map*(map>0) # thresholding positive values
            
            # normalization
            map = normalize_map(map,softmax=True,alpha=0.05) # original map is a 2D array of size (320, 512), with very small values (because softmaxed), normalize change this to 0-255
            maps.append(map) 

            if TAP=='TP':
                save_filename = f"TP_{task_image_label.split('.')[0]}_step{step}.{task_image_label.split('.')[1]}"
            elif TAP=='TPtoTA':
                save_filename = f"TA_{task_image_label.split('.')[0]}_step{step}.{task_image_label.split('.')[1]}"
            
            save_map(map, img_array, overlay=True, path_savefile=f"{SAVE_DIR_OVERLAY}/{save_filename}")
            save_map(map, img_array, overlay=False, path_savefile=f"{SAVE_DIR_MAPONLY}/{save_filename}")
            
        # Concatenated reward map
        concatenated_maps = np.stack(maps, axis=-1)
        # aggregated_map = np.mean(concatenated_maps, axis=-1)
        aggregated_map = np.sum(concatenated_maps, axis=-1)
        aggregated_map = normalize_map(aggregated_map, softmax=False)

        if TAP=='TP':
            save_filename = f"TP_{task_image_label.split('.')[0]}_concat.{task_image_label.split('.')[1]}"
        elif TAP=='TPtoTA':
            save_filename = f"TA_{task_image_label.split('.')[0]}_concat.{task_image_label.split('.')[1]}"
        save_map(aggregated_map, img_array, overlay=True, path_savefile=f"{SAVE_DIR_OVERLAY}/{save_filename}")
        save_map(aggregated_map, img_array, overlay=False, path_savefile=f"{SAVE_DIR_MAPONLY}/{save_filename}")

if __name__ == '__main__':
    main()


    


