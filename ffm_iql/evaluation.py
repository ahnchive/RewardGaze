import sys

sys.path.append('../common')

from torch.distributions import Categorical
from common import utils, metrics
from tqdm import tqdm, trange
from copy import deepcopy
import torch
import numpy as np
import torch.nn.functional as F
from os.path import join
import json
import cv2, os
import pandas as pd

def pack_model_inputs(obs_fov, env):
    is_composite_state = isinstance(obs_fov, tuple)
    if is_composite_state:
        inputs = (*obs_fov, env.task_ids)
    else:
        inputs = (obs_fov, env.task_ids)
    return inputs

def softmax_2d(map2d, temperature=0.05):
    flat = map2d.flatten()
    exp = np.exp((flat - np.max(flat)) / temperature)
    softmax = exp / (np.sum(exp) + 1e-8)
    return softmax.reshape(map2d.shape)

def collect_trajs(env,
                  policy,
                  patch_num,
                  max_traj_length,
                  is_eval=False,
                  sample_action=True,
                  is_zero_shot=False,
                  sample_scheme='IG_MAX',
                  sample_stop=False,
                  save_visualization=False,
                  num_samples=30):
    """Collect trajectories."""

    obs_fov = env.observe()

    states = pack_model_inputs(obs_fov, env)
    act, q, p = policy.select_action(states,
                                     sample_action,
                                     action_mask=env.action_mask,
                                     sample_stop=sample_stop)
    status = [env.status]

    bs = len(env.img_names)
    i = 0
    actions, qs, ps = [], [], []
    next_Vs = []
    while i < max_traj_length:
        # Sample actions for estimating immediate reward for viusalization
        if save_visualization:  
            next_V = torch.zeros_like(q)
            for _ in range(num_samples):
                tmp_env = deepcopy(env)
                tmp_act, _, _ = policy.select_action(
                    states,
                    True,
                    action_mask=tmp_env.action_mask,
                    sample_stop=sample_stop)
                new_obs_fov, _ = tmp_env.step(tmp_act)
                tmp_states = pack_model_inputs(new_obs_fov, tmp_env)
                next_V[torch.arange(bs), tmp_act] += policy.getV(tmp_states)
            next_Vs.append(next_V / num_samples)

        new_obs_fov, curr_status = env.step(act)
        status.append(curr_status)
        actions.append(act)
        qs.append(q)
        ps.append(p)
        obs_fov = new_obs_fov
        states = pack_model_inputs(obs_fov, env)
        act, q, p = policy.select_action(states,
                                         sample_action,
                                         action_mask=env.action_mask,
                                         sample_stop=sample_stop)
        i = i + 1

    status = torch.stack(status[1:])
    actions = torch.stack(actions)
    
    # Estimate immediate reward
    if save_visualization:
        ps = torch.stack(ps)
        qs = torch.stack(qs)
        next_Vs = torch.stack(next_Vs)
        rs = qs - policy.gamma * next_Vs

    trajs = []
    for i in range(bs):
        ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
        if status[:, i].sum() == 0:
            ind = status.size(0)
        trajs.append({
            'actions': actions[:ind, i],
            'q_values': qs[:ind, i] if save_visualization else None,
            'rewards': rs[:ind, i] if save_visualization else None,
            'probs': ps[:ind, i] if save_visualization else None,
            'next_V': next_Vs[:ind, i] if save_visualization else None,
        })
    return trajs



def sample_scanpaths(
    env,
    model,
    dataloader,
    pa,
    sample_action=True,
    sample_scheme='CLS_MAX',
    sample_stop=False,
    save_visualization=False,
    save_qonly=False,
):
    env.max_step = pa.max_traj_length + 1

    # generating scanpaths
    all_actions, all_rewards = [], []
    print('Generating scanapths...')
    for subj_id in range(10):
        for batch in tqdm(dataloader):
            env.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            cond_batch = batch['condition']
            with torch.no_grad():
                env.reset()
                trajs = collect_trajs(
                    env,
                    model,
                    pa.patch_num,
                    pa.max_traj_length,
                    is_eval=True,
                    sample_action=sample_action,
                    sample_scheme=sample_scheme,
                    sample_stop=sample_stop,
                    save_visualization=save_visualization,
                )
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     cond_batch[i], trajs[i]['actions'])
                                    for i in range(env.batch_size)])
                if save_qonly:
                        for i in range(env.batch_size):
                            qs = trajs[i]['q_values'].view(-1, 1, 20, 32).squeeze(1).cpu().numpy().astype(np.float32)[:6]
                            all_rewards.append((cat_names_batch[i],
                                                img_names_batch[i], qs))
                else:
                    if save_visualization:
                        for i in range(env.batch_size):
                            qs = F.interpolate(
                                trajs[i]['q_values'].view(-1, 1, 20, 32),
                                size=(pa.im_h, pa.im_w),
                                mode='bilinear').squeeze(1).cpu().numpy().astype(np.float32)[:6]
                            rs = F.interpolate(
                                trajs[i]['rewards'].view(-1, 1, 20, 32),
                                size=(pa.im_h, pa.im_w),
                                mode='bilinear').squeeze(1).cpu().numpy().astype(np.float32)[:6]
                            ps = F.interpolate(
                                trajs[i]['probs'].view(-1, 1, 20, 32),
                                size=(pa.im_h, pa.im_w),
                                mode='bilinear').squeeze(1).cpu().numpy().astype(np.float32)[:6]
                            nextVs = F.interpolate(
                                trajs[i]['next_V'].view(-1, 1, 20, 32),
                                size=(pa.im_h, pa.im_w),
                                mode='bilinear').squeeze(1).cpu().numpy().astype(np.float32)[:6]

                            qs_out = []
                            for q in qs:
                                q_soft = softmax_2d(q)
                                q_norm = (q_soft - q_soft.min()) / (q_soft.max() - q_soft.min() + 1e-8)
                                qs_out.append(q_norm)
                            qs = np.array(qs_out)
                            ps = np.array([softmax_2d(p) for p in ps])

                            all_rewards.append((cat_names_batch[i],
                                                img_names_batch[i], qs, rs, ps, nextVs))

        if not sample_action:
            break
            
    scanpaths = utils.actions2scanpaths(all_actions, pa.patch_num)
#     np.save(f"pred_{pa.TAP}_our.npy", scanapths)
#     np.save(f"{pa.TAP}_reward_maps.npy", all_rewards)

    return scanpaths, all_rewards



def compute_conditional_saliency_metrics(pa, agent, dataloader,
                                         task_dep_prior_maps, ablate_channel='None'):
    with torch.no_grad():
        n_samples, info_gain, nss, auc= 0, 0, 0, 0

        for expert_batch in tqdm(dataloader):
            
            if ablate_channel != 'None':
                expert_batch['true_state'][:,ablate_channel,:,:] =0
                
            if agent.args.repr == 'FFN':
                expert_batch_state = (expert_batch['true_state'].to(
                    agent.device), expert_batch['normalized_fixations'].to(
                        agent.device),
                                      expert_batch['task_id'].to(agent.device))
            elif agent.args.repr == 'DCB':
                expert_batch_state = (expert_batch['true_state'].to(
                    agent.device), expert_batch['fix_ind_map'].to(
                        agent.device),
                                      expert_batch['task_id'].to(agent.device))
            else:
                raise NotImplementedError
            gt_next_fixs = (expert_batch['next_normalized_fixations'][:, -1] *
                            torch.tensor([pa.im_w, pa.im_h])).to(torch.long)

            q = agent.q_net(*expert_batch_state, force_IOR=False)
            bs = q.size(0)
            probs = F.softmax(q / agent.alpha, dim=1)
            probs = F.interpolate(probs.view(bs, 1, 20, 32),
                                  size=(pa.im_h, pa.im_w),
                                  mode='bilinear').view(bs, pa.im_h, pa.im_w)
            #probs = probs / (probs.sum(dim=(1,2), keepdim=True) + 1e-12)
            probs /= probs.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

            prior_maps = torch.stack([
                task_dep_prior_maps[task] for task in expert_batch['task_name']
            ])
            info_gain += metrics.compute_info_gain(probs, gt_next_fixs,
                                                   prior_maps)
            nss += metrics.compute_NSS(probs, gt_next_fixs)
            auc += metrics.compute_cAUC(probs, gt_next_fixs) 
            n_samples += bs

        info_gain /= n_samples
        nss /= n_samples
        auc /= n_samples

    return info_gain.item(), nss.item(), auc.item()


def evaluate(
    env,
    model,
    dataloader,
    gazeloader,
    pa,
    bbox_annos,
    human_cdf,
    fix_clusters,
    task_dep_prior_maps,
    semSS_strings,
    dataset_root,
    human_scanpath_test,
    sample_action=True,
    sample_scheme='CLS_MAX',
    sample_stop=False,
    log_dir=None,
    save_visualization=False,
):
    model.eval()
    TAP = pa.TAP
    if TAP == 'FV':
        cut1, cut2, cut3 = 4, 8, 16
    else:
        cut1, cut2, cut3 = 2, 4, 6

    scanpaths, visualizations = sample_scanpaths(env, model, dataloader, pa,
                                                 sample_action, sample_scheme,
                                                 sample_stop,
                                                 save_visualization)
    nonstop_scanpaths = deepcopy(scanpaths)

    if save_visualization:
        for vi, vis_name in enumerate(
            ['return_maps', 'reward_maps', 'prob_maps']):
            torch.save(visualizations, join(log_dir, f'{vis_name}_{TAP}.pt'))
            rm_dir = join(log_dir, f'{vis_name}_{TAP}')
            if not os.path.exists(rm_dir):
                os.makedirs(rm_dir)
            for i in trange(len(visualizations)):
                task_name, img_name = visualizations[i][:2]
                rm_task_dir = join(rm_dir, task_name)
                if not os.path.exists(rm_task_dir):
                    os.makedirs(rm_task_dir)
                img = cv2.imread(
                    join(dataset_root, 'images', task_name.replace(' ', '_'),
                         img_name))
                for j, m in enumerate(visualizations[i][2 + vi]):
                    # m = (np.clip(m, -1, 1) + 1) / 2 * 255
                    m = (m - m.min()) / (m.max() - m.min()) * 255
                    m = m.astype(np.uint8)
                    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
                    m = cv2.addWeighted(m, 0.5, img, 0.5, 0)
                    cv2.imwrite(
                        join(rm_task_dir, img_name.replace('.', f'_{j}.')), m)

    print('Computing metrics...')
    metrics_dict = {}
    if TAP == 'TP':
        if not sample_stop:
            utils.cutFixOnTarget(scanpaths, bbox_annos)
        # search effiency
        mean_cdf, _ = utils.compute_search_cdf(scanpaths, bbox_annos,
                                               pa.max_traj_length)
        metrics_dict.update(
            dict(
                zip([f"TFP_top{i}" for i in range(1, len(mean_cdf))],
                    mean_cdf[1:])))

        # tfp auc
        metrics_dict['TFP-AUC'] = metrics.compute_cdf_auc(mean_cdf)
        
        # probability mismatch
        metrics_dict['prob_mismatch'] = np.sum(
            np.abs(human_cdf[:len(mean_cdf)] - mean_cdf))


    # sequence score
    ss_2steps = metrics.get_seq_score(scanpaths, fix_clusters, cut1,
                                      True) # use nonstop_scanpaths for non cutFixOnTarget
    ss_4steps = metrics.get_seq_score(scanpaths, fix_clusters, cut2,
                                      True)
    ss_6steps = metrics.get_seq_score(scanpaths, fix_clusters, cut3,
                                      True)
    ss = metrics.get_seq_score(scanpaths, fix_clusters, pa.max_traj_length,
                               False)

    sss_2steps = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, cut1,
        f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', True)
    sss_4steps = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, cut2,
        f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', True)
    sss_6steps = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, cut3,
        f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', True)
    sss = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, pa.max_traj_length,
        f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', False)

    metrics_dict.update({
        f"{TAP}_seq_score_max": ss,
        f"{TAP}_seq_score_{cut1}steps": ss_2steps,
        f"{TAP}_seq_score_{cut2}steps": ss_4steps,
        f"{TAP}_seq_score_{cut3}steps": ss_6steps,
        f"{TAP}_semantic_seq_score_max": sss,
        f"{TAP}_semantic_seq_score_{cut1}steps": sss_2steps,
        f"{TAP}_semantic_seq_score_{cut2}steps": sss_4steps,
        f"{TAP}_semantic_seq_score_{cut3}steps": sss_6steps,
    })

    # temporal spatial saliency metrics
    if not sample_action:
        ig, nss, auc = compute_conditional_saliency_metrics(pa, model, gazeloader,
                                                       task_dep_prior_maps)
        metrics_dict.update({
            f"{TAP}_cIG": ig,
            f"{TAP}_cNSS": nss,
            f"{TAP}_cAUC": auc,
        })

    if sample_stop:
        sp_len_diff = []
        for traj in scanpaths:
            gt_trajs = list(
                filter(
                    lambda x: x['task'] == traj['task'] and x['name'] == traj[
                        'name'], human_scanpath_test))
            sp_len_diff.append(
                len(traj['X']) -
                np.array([len(traj['X']) for traj in gt_trajs]))
        sp_len_diff = np.abs(np.concatenate(sp_len_diff))
        metrics_dict[f'{TAP}_sp_len_err_mean'] = sp_len_diff.mean()
        metrics_dict[f'{TAP}_sp_len_err_std'] = sp_len_diff.std()
        metrics_dict[f'{TAP}_avg_sp_len'] = np.mean(
            [len(x['X']) for x in scanpaths])

    if not sample_action:
        prefix = sample_scheme + '_'
        keys = list(metrics_dict.keys())
        for k in keys:
            metrics_dict[prefix + k] = metrics_dict.pop(k)

    if log_dir is not None:
        for sp in scanpaths:
            sp['X'] = sp['X'].tolist()
            sp['Y'] = sp['Y'].tolist()
        with open(join(log_dir, f'predictions_{TAP}.json'), 'w') as f:
            json.dump(scanpaths, f, indent=4)
        with open(join(log_dir, f'metrics_{TAP}.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    model.train()

    return metrics_dict




def sample_scanpaths_ablate_dcb(
    env,
    model,
    dataloader,
    pa,
    sample_action=True,
    sample_scheme='CLS_MAX',
    sample_stop=False,
    save_visualization=False,
    ablate_channel='None',
):
    env.max_step = pa.max_traj_length + 1

    # generating scanpaths
    all_actions, all_rewards = [], []
    print(f'Generating scanapths for {ablate_channel} channel ablated')
    for subj_id in range(10):
        for batch in tqdm(dataloader):
            if ablate_channel != 'None':
                batch['lr_feats'][:,ablate_channel,:,:]=0
                batch['hr_feats'][:,ablate_channel,:,:]=0
            env.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            cond_batch = batch['condition']
            with torch.no_grad():
                env.reset()
                trajs = collect_trajs(
                    env,
                    model,
                    pa.patch_num,
                    pa.max_traj_length,
                    is_eval=True,
                    sample_action=sample_action,
                    sample_scheme=sample_scheme,
                    sample_stop=sample_stop,
                    save_visualization=save_visualization,
                )
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     cond_batch[i], trajs[i]['actions'])
                                    for i in range(env.batch_size)])

                if save_visualization:
                    for i in range(env.batch_size):
                        qs = F.interpolate(
                            trajs[i]['q_values'].view(-1, 1, 20, 32),
                            size=(pa.im_h, pa.im_w),
                            mode='bilinear').squeeze(1).cpu().numpy().astype(np.float16)[:6]
                        rs = F.interpolate(
                            trajs[i]['rewards'].view(-1, 1, 20, 32),
                            size=(pa.im_h, pa.im_w),
                            mode='bilinear').squeeze(1).cpu().numpy().astype(np.float16)[:6]
                        ps = F.interpolate(
                            trajs[i]['probs'].view(-1, 1, 20, 32),
                            size=(pa.im_h, pa.im_w),
                            mode='bilinear').squeeze(1).cpu().numpy().astype(np.float16)[:6]
                        all_rewards.append((cat_names_batch[i],
                                            img_names_batch[i], qs, rs, ps))

        if not sample_action:
            break
            
    scanpaths = utils.actions2scanpaths(all_actions, pa.patch_num)
#     np.save(f"pred_{pa.TAP}_our.npy", scanapths)
#     np.save(f"{pa.TAP}_reward_maps.npy", all_rewards)

    return scanpaths, all_rewards




def evaluate_ablate_dcb(
    env,
    model,
    dataloader,
    gazeloader,
    pa,
    bbox_annos,
    human_cdf,
    fix_clusters,
    task_dep_prior_maps,
    semSS_strings,
    dataset_root,
    human_scanpath_test,
    sample_action=True,
    sample_scheme='CLS_MAX',
    sample_stop=False,
    log_dir=None,
    save_visualization=False,
    
    dcb_channels=None,
    return_by_image=False,
    evaluation_on_training=False,
):
    model.eval()
    TAP = pa.TAP
    
    # if TAP == 'FV':
    #     cut1, cut2, cut3 = 4, 8, 16
    # else:
    #     cut1, cut2, cut3 = 2, 4, 6

    # computing metrics task-wise and save in df
    # for loop over all 134 channels...
    

    if dcb_channels == None:
        dcb_channels= ['None'] + [i for i in range(134)] #[i for i in range(134)]
    else:
        dcb_channels = ['None'] + dcb_channels
        
    TFPs, TFP_AUCs, SSs, SemSSs, cIGs, cNSSs, cAUCs, SPRs = ([] for i in range(8))
    
    output={}
    for dcb in dcb_channels:
        output[dcb]={}
        scanpaths, visualizations = sample_scanpaths_ablate_dcb(env, model, dataloader, pa,
                                                                sample_action, sample_scheme,
                                                                sample_stop, save_visualization,
                                                                ablate_channel=dcb)
        output[dcb]['scanpaths'] =scanpaths
        output[dcb]['visualizations']= visualizations
        
        print(f'Computing metrics for {dcb} channel ablated')
        

        if TAP == 'TP':
            if not sample_stop:
                utils.cutFixOnTarget(scanpaths, bbox_annos)
                
            # search effiency
            mean_cdf, cdf_byimage = utils.compute_search_cdf(scanpaths, bbox_annos,
                                                   pa.max_traj_length, return_by_image=return_by_image)
            output[dcb]['cdf']=cdf_byimage 
            
            TFPs.append(mean_cdf[-1])

            tfpauc = metrics.compute_cdf_auc(mean_cdf)
            TFP_AUCs.append(tfpauc)
            
        ss = metrics.get_seq_score(scanpaths, fix_clusters, max_step=pa.max_traj_length, truncate_gt=True)
        SSs.append(ss)
        
        # sem ss for training not ready, semss only has testing images.
        if not evaluation_on_training:
            sem_ss = metrics.get_semantic_seq_score(scanpaths, semSS_strings, pa.max_traj_length, f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', False)
            SemSSs.append(sem_ss)    


        # temporal spatial saliency metrics
        if not sample_action:
            ig, nss, auc = compute_conditional_saliency_metrics(pa, model, gazeloader,
                                                           task_dep_prior_maps, ablate_channel=dcb)
            cIGs.append(ig)
            cNSSs.append(nss)
            cAUCs.append(auc)


        # scanpath ratio
        spr = metrics.compute_avgSPRatio(scanpaths, bbox_annos, pa.max_traj_length)
        SPRs.append(spr)

        # if not sample_action:
        #     prefix = sample_scheme + '_'
        #     keys = list(metrics_dict.keys())
        #     for k in keys:
        #         metrics_dict[prefix + k] = metrics_dict.pop(k)

        # if log_dir is not None:
        #     for sp in scanpaths:
        #         sp['X'] = sp['X'].tolist()
        #         sp['Y'] = sp['Y'].tolist()
        #     with open(join(log_dir, f'predictions_{TAP}.json'), 'w') as f:
        #         json.dump(scanpaths, f, indent=4)
        #     with open(join(log_dir, f'metrics_{TAP}.json'), 'w') as f:
        #         json.dump(metrics_dict, f, indent=4)

    df = pd.DataFrame()
    df['Ablated_Channel'] = dcb_channels
    df['TFP'] = TFPs 
    df['TFP-AUCs'] = TFP_AUCs
    df['SS'] = SSs
    df['SemSS'] = SemSSs
    df['cIG'] = cIGs
    df['cNSS'] = cNSSs
    df['cAUC'] = cAUCs
    df['SPR'] = SPRs
    
    model.train()

    return df, output


## inner loop over tasks when computing metrics, output is categorywise df
## the dataloader should be task-wise for computing conditional saliency though
# def evaluate_ablate_dcb(
#     env,
#     model,
#     dataloader,
#     gazeloader,
#     pa,
#     bbox_annos,
#     human_cdf,
#     fix_clusters,
#     task_dep_prior_maps,
#     semSS_strings,
#     dataset_root,
#     human_scanpath_test,
#     sample_action=True,
#     sample_scheme='CLS_MAX',
#     sample_stop=False,
#     log_dir=None,
#     save_visualization=False,
# ):
#     model.eval()
#     TAP = pa.TAP
    
#     # if TAP == 'FV':
#     #     cut1, cut2, cut3 = 4, 8, 16
#     # else:
#     #     cut1, cut2, cut3 = 2, 4, 6

#     # computing metrics task-wise and save in df
#     # for loop over all 134 channels...
    
#     all_scanpaths, visualizations = sample_scanpaths(env, model, dataloader, pa,
#                                                  sample_action, sample_scheme,
#                                                  sample_stop,
#                                                  save_visualization)
 


#     tasks = list(np.unique([s['task'] for s in all_scanpaths]))
#     TFPs, SSs, SemSSs, cIGs, cNSSs, cAUCs, SPRs = ([] for i in range(7))
#     for task in tasks:
#         scanpaths = [s for s in all_scanpaths if s['task']==task]

#         print('Computing metrics...')

#         if TAP == 'TP':
#             if not sample_stop:
#                 utils.cutFixOnTarget(scanpaths, bbox_annos)
                
#             # search effiency
#             mean_cdf, _ = utils.compute_search_cdf(scanpaths, bbox_annos,
#                                                    pa.max_traj_length)
#             TFPs.append(mean_cdf[-1])

#         ss = metrics.get_seq_score(scanpaths, fix_clusters, pa.max_traj_length, True)
#         SSs.append(ss)
        
#         sem_ss = metrics.get_semantic_seq_score(scanpaths, semSS_strings, pa.max_traj_length, f'{dataset_root}/{pa.sem_seq_dir}/segmentation_maps', False)
#         SemSSs.append(ss)    


#         # temporal spatial saliency metrics
#         if not sample_action:
#             ig, nss, auc = compute_conditional_saliency_metrics(pa, model, gazeloader,
#                                                            task_dep_prior_maps)
#             cIGs.append(ig)
#             cNSSs.append(nss)
#             cAUCs.append(auc)


#         # scanpath ratio
#         spr = metrics.compute_avgSPRatio(scanpaths, bbox_annos, pa.max_traj_length)
#         SPRs.append(spr)

#         # if not sample_action:
#         #     prefix = sample_scheme + '_'
#         #     keys = list(metrics_dict.keys())
#         #     for k in keys:
#         #         metrics_dict[prefix + k] = metrics_dict.pop(k)

#         # if log_dir is not None:
#         #     for sp in scanpaths:
#         #         sp['X'] = sp['X'].tolist()
#         #         sp['Y'] = sp['Y'].tolist()
#         #     with open(join(log_dir, f'predictions_{TAP}.json'), 'w') as f:
#         #         json.dump(scanpaths, f, indent=4)
#         #     with open(join(log_dir, f'metrics_{TAP}.json'), 'w') as f:
#         #         json.dump(metrics_dict, f, indent=4)

#     df = pd.DataFrame()
#     df['task'] = tasks
#     df['TFP'] = TFPs  
#     df['SS'] = SSs
#     df['SemSS'] = SemSSs
#     df['cIG'] = cIGs
#     df['cNSS'] = cNSSs
#     df['cAUC'] = cAUCs
#     df['SPR'] = SPRs
    
#     model.train()

#     return df
