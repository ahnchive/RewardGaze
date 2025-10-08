import sys

sys.path.append('../common')

from common.environment import FFN_Env, DCB_Env
from common.dataset import process_data
from .models import FFNGeneratorCond_V2, DCB_IQL
from .sql import SoftQ
from .iql import irl_update, irl_update_critic
from common.config import JsonConfig
from common.utils import get_prior_maps, cutFixOnTarget
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
import types


def build_dataset(hparams, dataset_root, device, load_coco_annos=True, is_testing=True):
    # dir of pre-computed beliefs
    dataset_name = hparams.Data.name
    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()

    # load ground-truth human scanpaths
    is_osie = dataset_name == 'OSIE'
    if is_osie:
        with open(join(dataset_root, 'OSIE',
                       'osie_fixations.json')) as json_file:
            human_scanpaths = json.load(json_file)
        dataset_root = join(dataset_root, 'OSIE')
    else:
        with open(
                join(dataset_root,
                     'coco_search_fixations_512x320_on_target_allvalid.json')
        ) as json_file:
            human_scanpaths = json.load(json_file)

        n_tasks = 18
        # exclude incorrect scanpaths
        if hparams.Data.exclude_wrong_trials:
            human_scanpaths = list(
                filter(lambda x: x['correct'] == 1, human_scanpaths))
        human_scanpaths = list(
            filter(lambda x: x['fixOnTarget'] or x['condition'] == 'absent',
                   human_scanpaths))
        if (hparams.Data.include_freeview or hparams.Data.TAP == 'FV'):
            n_tasks += 1
            with open(
                    join(dataset_root,
                         'coco_freeview_fixations_512x320.json')) as json_file:
                fv_sps = json.load(json_file)
                human_scanpaths.extend(fv_sps)
    human_scanpaths_all = human_scanpaths

    # choose data to use: TP = target-present trials, TA = target-absent trials
    # else = all trials, FV = free-viewing trials
    human_scanpaths_ta = list(
        filter(lambda x: x['condition'] == 'absent', human_scanpaths_all))
    human_scanpaths_tp = list(
        filter(lambda x: x['condition'] == 'present', human_scanpaths_all))
    human_scanpaths_fv = list(
        filter(lambda x: x['condition'] == 'freeview', human_scanpaths_all))

    if hparams.Data.TAP == 'TP':
        human_scanpaths = list(
            filter(lambda x: x['condition'] != 'absent', human_scanpaths))
        human_scanpaths = list(
            filter(lambda x: x['fixOnTarget'], human_scanpaths))
        cutFixOnTarget(human_scanpaths, bbox_annos)
    elif hparams.Data.TAP == 'TA':
        human_scanpaths = list(
            filter(lambda x: x['condition'] != 'present', human_scanpaths))
    elif hparams.Data.TAP == 'FV':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'freeview', human_scanpaths))
        n_tasks = 1

    # process fixation data
    dataset = process_data(human_scanpaths,
                           dataset_root,
                           bbox_annos,
                           hparams,
                           human_scanpaths_all,
                           is_testing=is_testing,
                           sample_scanpath=False,
                           use_coco_annotation=load_coco_annos)

        # dataset.keys()
        # {
        #     'catIds': catIds,
        #     'img_train': train_img_dataset,
        #     'img_valid_TP': valid_img_dataset_TP,
        #     'img_valid_TA': valid_img_dataset_TA,
        #     'img_valid_FV': valid_img_dataset_FV,
        #     'img_valid': valid_img_dataset_all,
        #     'gaze_train': train_HG_dataset,
        #     'gaze_valid': valid_HG_dataset,
        #     'gaze_valid_TP': valid_HG_dataset_TP,
        #     'gaze_valid_TA': valid_HG_dataset_TA,
        #     'gaze_valid_FV': valid_HG_dataset_FV,
        #     'bbox_annos': target_annos,
        #     'fix_clusters': fix_clusters,
        #     'valid_scanpaths': valid_target_trajs_all,
        #     'human_cdf': human_mean_cdf,
        # }
# in 'img_train'
#         {
#             "task_id": 0 if is_fv else self.catIds[cat_name],
#             'img_name': img_name,
#             'cat_name': cat_name,
#             'lr_feats': lr,
#             'hr_feats': hr,
#             'history_map': history_map,
#             'fix_ind_map': fix_ind_map,
#             'init_fix': torch.FloatTensor(init_fix),
#             'label_coding': coding,
#             'action_mask': action_mask,
#             'condition': condition,
#         }
    
    return dataset

def build(hparams, dataset_root, device, load_coco_annos=True, taskname=None, evaluation_on_training=False):
    # dir of pre-computed beliefs
    dataset_name = hparams.Data.name
    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()

    # load ground-truth human scanpaths
    is_osie = dataset_name == 'OSIE'
    if is_osie:
        with open(join(dataset_root, 'OSIE',
                       'osie_fixations.json')) as json_file:
            human_scanpaths = json.load(json_file)
        dataset_root = join(dataset_root, 'OSIE')
    else:
        with open(
                join(dataset_root,
                     'coco_search_fixations_512x320_on_target_allvalid.json')
        ) as json_file:
            human_scanpaths = json.load(json_file)

        n_tasks = 18
        # exclude incorrect scanpaths
        if hparams.Data.exclude_wrong_trials:
            human_scanpaths = list(
                filter(lambda x: x['correct'] == 1, human_scanpaths))
        human_scanpaths = list(
            filter(lambda x: x['fixOnTarget'] or x['condition'] == 'absent',
                   human_scanpaths))
        if (hparams.Data.include_freeview or hparams.Data.TAP == 'FV'):
            n_tasks += 1
            with open(
                    join(dataset_root,
                         'coco_freeview_fixations_512x320.json')) as json_file:
                fv_sps = json.load(json_file)
                human_scanpaths.extend(fv_sps)
    
    
    # for dcb ablation studies, filtering scanpaths by tasks
    hparams.ablate_dcb =False
    if taskname is not None:
        hparams.ablate_dcb =True
        human_scanpaths = list(filter(lambda x: x['task'] == taskname, human_scanpaths))
        
    
    human_scanpaths_all = human_scanpaths

    # choose data to use: TP = target-present trials, TA = target-absent trials
    # else = all trials, FV = free-viewing trials
    human_scanpaths_ta = list(
        filter(lambda x: x['condition'] == 'absent', human_scanpaths_all))
    human_scanpaths_tp = list(
        filter(lambda x: x['condition'] == 'present', human_scanpaths_all))
    human_scanpaths_fv = list(
        filter(lambda x: x['condition'] == 'freeview', human_scanpaths_all))

    if hparams.Data.TAP == 'TP':
        human_scanpaths = list(
            filter(lambda x: x['condition'] != 'absent', human_scanpaths))
        human_scanpaths = list(
            filter(lambda x: x['fixOnTarget'], human_scanpaths))
        cutFixOnTarget(human_scanpaths, bbox_annos)
    elif hparams.Data.TAP == 'TA':
        human_scanpaths = list(
            filter(lambda x: x['condition'] != 'present', human_scanpaths))
    elif hparams.Data.TAP == 'FV':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'freeview', human_scanpaths))
        n_tasks = 1

    # process fixation data
    dataset = process_data(human_scanpaths,
                           dataset_root,
                           bbox_annos,
                           hparams,
                           human_scanpaths_all,
                           sample_scanpath=False,
                           use_coco_annotation=load_coco_annos)

    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers
    train_drop_last = False if evaluation_on_training else True
    train_HG_loader = DataLoader(dataset['gaze_train'],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=n_workers,
                                 drop_last= train_drop_last,
                                 pin_memory=True)

    print('num of training batches =', len(train_HG_loader))

    train_img_loader = DataLoader(dataset['img_train'],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  drop_last=train_drop_last,
                                  pin_memory=True)

    valid_img_loader = DataLoader(dataset['img_valid_TP'],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  drop_last=False,
                                  pin_memory=True)

    valid_img_loader_TA = DataLoader(dataset['img_valid_TA'],
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=n_workers,
                                     drop_last=False,
                                     pin_memory=True)
    valid_img_loader_FV = DataLoader(dataset['img_valid_FV'],
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=n_workers,
                                     drop_last=False,
                                     pin_memory=True)

    valid_HG_loader_TP = DataLoader(dataset['gaze_valid_TP'],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)
    valid_HG_loader_TA = DataLoader(dataset['gaze_valid_TA'],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)
    valid_HG_loader_FV = DataLoader(dataset['gaze_valid_FV'],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)

    ffn_dim = hparams.Model.foveal_feature_dim
    hidden_dim = hparams.Model.gen_hidden_size
    if hparams.Train.repr == 'FFN':
        q_net = FFNGeneratorCond_V2(ffn_dim,
                                    hidden_size=hidden_dim,
                                    num_targets=n_tasks,
                                    is_cumulative=True).to(device)
        env_func = FFN_Env
    elif hparams.Train.repr == 'DCB':
        catIds = dataset['catIds']
        task_eye = torch.eye(len(catIds)).to(device)
        num_tasks, task_emb_size = len(catIds), 32
        q_net = DCB_IQL(num_tasks,
                        task_emb_size,
                        task_eye,
                        134,
                        hparams.Data.max_traj_length,
                        dropout=hparams.Train.dropout,
                        enhance_DCB=hparams.Model.enhance_DCB,
                        forgetting_rate=hparams.Train.forgetting_rate).to(device)
        env_func = DCB_Env
    else:
        raise NotImplementedError

    if hparams.Train.parallel:
        q_net = torch.nn.DataParallel(q_net)
    if len(hparams.Model.checkpoint) > 0:
        ckp_pth =  hparams.Model.checkpoint
#         ckp_pth = join(hparams.Train.log_dir, hparams.Model.checkpoint)
        ckp = torch.load(ckp_pth)
        q_net.load_state_dict(ckp['model'])
        print(f"loaded weights from {ckp_pth}.")
    else:
        ckp = None

    env = env_func(hparams.Data,
                   max_step=hparams.Data.max_traj_length,
                   mask_size=hparams.Data.IOR_size,
                   status_update_mtd=hparams.Train.stop_criteria,
                   device=device,
                   ret_inhibition=hparams.Data.force_IOR_train)
    env_valid = env_func(hparams.Data,
                         max_step=hparams.Data.max_traj_length,
                         mask_size=hparams.Data.IOR_size,
                         status_update_mtd=hparams.Train.stop_criteria,
                         device=device,
                         ret_inhibition=hparams.Data.force_IOR_test)
    agent = SoftQ(q_net, batch_size, device, hparams.Train)
    agent.irl_update = types.MethodType(irl_update, agent)
    agent.ilr_update_critic = types.MethodType(irl_update_critic, agent)

    if ckp is not None:
        global_step = ckp['step']
        agent.critic_optimizer.load_state_dict(ckp['optimizer'])
    else:
        global_step = 0

    bbox_annos = dataset['bbox_annos']
    human_cdf = dataset['human_cdf']
    fix_clusters = dataset['fix_clusters']
    prior_maps_ta = get_prior_maps(human_scanpaths_ta, hparams.Data.im_w,
                                   hparams.Data.im_h)
    keys = list(prior_maps_ta.keys())
    for k in keys:
        prior_maps_ta[f'{k}'] = torch.tensor(prior_maps_ta.pop(k)).to(device)

    prior_maps_tp = get_prior_maps(human_scanpaths_tp, hparams.Data.im_w,
                                   hparams.Data.im_h)
    keys = list(prior_maps_tp.keys())
    for k in keys:
        prior_maps_tp[f'{k}'] = torch.tensor(prior_maps_tp.pop(k)).to(device)
    if len(human_scanpaths_fv) > 0:
        prior_maps_fv = get_prior_maps(human_scanpaths_fv, hparams.Data.im_w,
                                       hparams.Data.im_h)
        keys = list(prior_maps_fv.keys())
        for k in keys:
            prior_maps_fv[k] = torch.tensor(prior_maps_fv.pop(k)).to(device)
        # For freeview, we use the 'all' prior ap for evaluation
        for k in keys:
            prior_maps_fv[k] = prior_maps_fv['all']
    else:
        prior_maps_fv = None

    if is_osie:
        sss_strings = None
    else:
        sss_strings = np.load(join(dataset_root, hparams.Data.sem_seq_dir,
                                   'test.pkl'),
                              allow_pickle=True)
    sps_test_tp = list(
        filter(lambda x: x['split'] == 'test', human_scanpaths_tp))
    sps_test_ta = list(
        filter(lambda x: x['split'] == 'test', human_scanpaths_ta))
    sps_test_fv = list(
        filter(lambda x: x['split'] == 'test', human_scanpaths_fv))
    
    


    if evaluation_on_training: # add scanapth used for training
        sps_train_tp = list(
            filter(lambda x: x['split'] == 'train', human_scanpaths_tp))
        sps_train_ta = list(
            filter(lambda x: x['split'] == 'train', human_scanpaths_ta))
        sps_train_fv = list(
            filter(lambda x: x['split'] == 'train', human_scanpaths_fv))
    
        return (agent, train_HG_loader, train_img_loader, valid_img_loader,
                valid_img_loader_TA, valid_img_loader_FV, env, env_valid, \
                global_step, bbox_annos, human_cdf, fix_clusters, prior_maps_tp, \
                prior_maps_ta, prior_maps_fv, sss_strings, valid_HG_loader_TP, \
                valid_HG_loader_TA, valid_HG_loader_FV, sps_test_tp, sps_test_ta, \
                sps_test_fv, sps_train_tp, sps_train_ta, sps_train_fv)
    
    else:
        return (agent, train_HG_loader, train_img_loader, valid_img_loader,
                valid_img_loader_TA, valid_img_loader_FV, env, env_valid, \
                global_step, bbox_annos, human_cdf, fix_clusters, prior_maps_tp, \
                prior_maps_ta, prior_maps_fv, sss_strings, valid_HG_loader_TP, \
                valid_HG_loader_TA, valid_HG_loader_FV, sps_test_tp, sps_test_ta, \
                sps_test_fv)

