import argparse
import json 

from common.config import JsonConfig
from ffm_iql.builder import build, build_dataset
from ffm_iql.evaluation import evaluate_ablate_dcb, sample_scanpaths

import torch
import random
import numpy as np
import pandas as pd 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
# device = torch.device('cpu')
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

categories = ['bottle',
'bowl',
'car',
'chair',
'clock',
'cup',
'fork',
'keyboard',
'knife',
'laptop',
'microwave',
'mouse',
'oven',
'potted plant',
'sink',
'stop sign',
'toilet',
'tv']

SAVE_DIR = './results/DCBablation_SemSSupdated/'

def main(args):
    # load configs
    dataset_root = '/home/young/hdd1/coco-search18/'
    hparams = JsonConfig('./configs/coco_search18_eDCB_IQL_TP_LIOR.json')
    # hparams = JsonConfig('./configs/coco_search18_eDCB_IQL_TA_LIOR.json')

    TAP='TP' 
    # TAP='TPtoTA'
    evaluation_on_training = False # make True will give issue using fixclusters; TODO; find the bandwith number used for creating the exisiting clusters 
    
    # hparams.Train['force_IOR'] = False

    # save_visualization = False
    # log_dir = './results/explore_it'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # print("Log dir:", log_dir)
    # config_dir = os.path.dirname(args.hparams)

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
  

    
    # build model
    for taskname in categories:#[None] + categories: 

        if taskname is None:
            tasknameforprint = 'all'
        else:
            tasknameforprint = taskname
        
        print(f'build model and dataloader for {tasknameforprint} task')
        if evaluation_on_training:
            agent, train_gaze_loader, train_img_loader, valid_img_loader,\
            valid_img_loader_ta, valid_img_loader_fv, env, env_valid, global_step, bbox_annos,\
            human_cdf, fix_clusters, prior_maps_tp, prior_maps_ta, prior_maps_fv,\
            sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta, valid_gaze_loader_fv,\
            sps_test_tp, sps_test_ta, sps_test_fv, sps_train_tp, sps_train_ta, sps_train_fv= build(hparams, dataset_root, device, load_coco_annos=False, taskname=taskname, evaluation_on_training= evaluation_on_training)

        else:
            agent, train_gaze_loader, train_img_loader, valid_img_loader,\
                valid_img_loader_ta, valid_img_loader_fv, env, env_valid, global_step, bbox_annos,\
                human_cdf, fix_clusters, prior_maps_tp, prior_maps_ta, prior_maps_fv,\
                sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta, valid_gaze_loader_fv,\
                sps_test_tp, sps_test_ta, sps_test_fv = build(hparams, dataset_root, device, load_coco_annos=False, taskname=taskname, evaluation_on_training= evaluation_on_training)

        # insert log_dir for saving visualizations
        # code modified from evalaute in evaluation.py
        if evaluation_on_training:
            img_loader = train_img_loader
            gaze_loader = train_gaze_loader
            human_gt = sps_train_tp
        else:
            env= env_valid
            img_loader = valid_img_loader
            gaze_loader = valid_gaze_loader_tp
            human_gt =  sps_test_tp
            
        def run_evaluation():
            rst_tp, out_tp = None, None

            rst_tp, out_tp = evaluate_ablate_dcb(
                env,
                agent,
                img_loader,
                gaze_loader,
                hparams_tp.Data,
                bbox_annos,
                human_cdf,
                fix_clusters,
                prior_maps_tp,
                sss_strings,
                dataset_root,
                human_gt,
                sample_action=False,
                sample_scheme='Greedy',
                log_dir=None,
                save_visualization=False,
                return_by_image=True,
            )
            # print("TP:", rst_tp)

            return rst_tp, out_tp

        results = run_evaluation()
        # display(metrics[0].round(3))
        
        #save metrics results
        results[0].to_csv(f'{SAVE_DIR}/metric_{tasknameforprint}.csv', index=False)
        
        #save all cdf by images
        np.save(f'{SAVE_DIR}/output_{tasknameforprint}.npy', results[1])
        # with open(f'./results/DCBablation_v3/output_{tasknameforprint}.json', "w") as file:
        #     json.dump(results[1], file)
    
        print(f'=========== results saved for {tasknameforprint} task ========\n\n\n')
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add your command-line arguments here
    # Example:
    # parser.add_argument('--dataset', type=str, help='Path to the dataset')
    
    # args = parser.parse_args()
    main(args=None)