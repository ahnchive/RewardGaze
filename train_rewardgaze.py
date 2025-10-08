"""
Foveated Feature Map Training Script.
This script is a simplified version of the training script in 
https://github.com/cvlab-stonybrook/Scanpath_Prediction
"""
import argparse
import os
import random

import numpy as np

import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ffm_iql.builder import build
from common.config import JsonConfig
from ffm_iql.evaluation import pack_model_inputs, evaluate
from ffm_iql.replay_buffer import Memory

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams',
                        type=str,
                        help='hyper parameters config file path')
    parser.add_argument('--dataset-root', type=str, help='dataset root path')
    parser.add_argument('--cuda-id', type=int, default=0, help='cuda id')
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='perform evaluation only')
    parser.add_argument('--save-visualization',
                        action='store_true',
                        help='save visualization maps to <log_dir>')
    return parser.parse_args()


def get_batch(policy_buffer, expert_demos_iter, expert_loader, pa):
    """Get batch ready for training."""
    try:
        expert_batch = next(expert_demos_iter)
    except StopIteration:
        expert_demos_iter = iter(expert_loader)
        expert_batch = next(expert_demos_iter)
    if pa.repr == 'FFN':
        expert_batch_state = (expert_batch['true_state'],
                              expert_batch['normalized_fixations'],
                              expert_batch['task_id'])
        expert_batch_next_state = (expert_batch['true_state'],
                                   expert_batch['next_normalized_fixations'],
                                   expert_batch['task_id'])
    elif pa.repr == 'DCB':
        expert_batch_state = (expert_batch['true_state'],
                              expert_batch['fix_ind_map'],
                              expert_batch['task_id'])
        expert_batch_next_state = (expert_batch['next_true_state'],
                                   expert_batch['next_fix_ind_map'],
                                   expert_batch['task_id'])
    else:
        raise NotImplementedError
    expert_batch_action = expert_batch['true_action'].squeeze()
    expert_batch_aux = expert_batch['centermaps'] if pa.use_det_loss else None
    expert_batch_is_done = torch.ones_like(expert_batch_action,
                                           dtype=torch.bool)
    expert_batch = (expert_batch_state, expert_batch_next_state,
                    expert_batch_action, expert_batch_is_done,
                    expert_batch_aux)
    policy_batch = online_memory_replay.get_samples(pa.batch_size)
    return policy_batch, expert_batch, expert_demos_iter


def log_dict(writer, scalars, step, prefix):
    for k, v in scalars.items():
        writer.add_scalar(prefix + "/" + k, v, step)

def run_evaluation():
    rst_tp, rst_ta, rst_fv = None, None, None
    if hparams.Data.TAP in ['TP', 'TAP']:
        rst_tp = evaluate(
            env_valid,
            agent,
            valid_img_loader,
            valid_gaze_loader_tp,
            hparams_tp.Data,
            bbox_annos,
            human_cdf,
            fix_clusters,
            prior_maps_tp,
            sss_strings,
            dataset_root,
            sps_test_tp,
            sample_action=False,
            sample_scheme='Greedy',
            log_dir=log_dir,
            save_visualization=save_visualization,
        )
        print("TP:", rst_tp)
    if hparams.Data.TAP in ['TA', 'TAP']:
        rst_ta = evaluate(
            env_valid,
            agent,
            valid_img_loader_ta,
            valid_gaze_loader_ta,
            hparams_ta.Data,
            bbox_annos,
            human_cdf,
            fix_clusters,
            prior_maps_ta,
            sss_strings,
            dataset_root,
            sps_test_ta,
            sample_action=False,
            sample_scheme='Greedy',
            log_dir=log_dir,
            save_visualization=save_visualization,
        )
        print("TA", rst_ta)
    if hparams.Data.TAP in ['FV', 'ALL']:
        rst_fv = evaluate(
            env_valid,
            agent,
            valid_img_loader_fv,
            valid_gaze_loader_fv,
            hparams_fv.Data,
            bbox_annos,
            human_cdf,
            fix_clusters,
            prior_maps_fv,
            sss_strings,
            dataset_root,
            sps_test_fv,
            sample_action=False,
            sample_scheme='Greedy',
            log_dir=log_dir,
            save_visualization=save_visualization,
        )
        print("FV", rst_fv)
    return rst_tp, rst_ta, rst_fv

if __name__ == '__main__':
    args = parse_args()
    hparams = JsonConfig(args.hparams)
    config_dir = os.path.dirname(args.hparams)
    hparams_tp = JsonConfig(os.path.join(config_dir, 'coco_search18_TP.json'))
    hparams_ta = JsonConfig(os.path.join(config_dir, 'coco_search18_TA.json'))
    hparams_fv = JsonConfig(os.path.join(config_dir, 'coco_search18_FV.json'))
    
    log_dir = hparams.Train.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("Log dir:", log_dir)
    save_visualization = args.save_visualization
    dataset_root = args.dataset_root
    device = torch.device('cuda:{}'.format(args.cuda_id))
    
    agent, train_gaze_loader, train_img_loader, valid_img_loader,\
        valid_img_loader_ta, valid_img_loader_fv, env, env_valid, global_step, bbox_annos,\
        human_cdf, fix_clusters, prior_maps_tp, prior_maps_ta, prior_maps_fv,\
        sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta, valid_gaze_loader_fv,\
        sps_test_tp, sps_test_ta, sps_test_fv = build(
            hparams, dataset_root, device, load_coco_annos=not args.eval_only)
    if args.eval_only:
        run_evaluation()
    else:
        writer = SummaryWriter(log_dir)
        log_folder_runs = "./runs/{}".format(log_dir.split('/')[-1])
        if not os.path.exists(log_folder_runs):
            os.system(f"mkdir -p {log_folder_runs}")

        # Write configuration file to the log dir
        hparams.dump(log_dir, 'config.json')

        print_every = 20
        max_iters = hparams.Train.max_iters
        save_every = hparams.Train.checkpoint_every
        eval_every = hparams.Train.evaluate_every

        replay_memory = hparams.Train.replay_memory
        initial_memory = hparams.Train.initial_memory
        online_memory_replay = Memory(replay_memory, SEED)

        expert_demos_iter = iter(train_gaze_loader)
        s_epoch = int(global_step / len(train_img_loader))
        last_time = datetime.datetime.now()

        # Initial evaluation
        # rst_tp, rst_ta, rst_fv = run_evaluation()
        # if rst_tp is not None:
        #     log_dict(writer, rst_tp, global_step, "eval_TP")
        # if rst_ta is not None:
        #     log_dict(writer, rst_ta, global_step, "eval_TA")
        # if rst_fv is not None:
        #     log_dict(writer, rst_fv, global_step, "eval_FV")
        # writer.add_scalar('epoch', 0, global_step)
        
        # Start training
        for i_epoch in range(s_epoch, int(1e5)):
            for i_batch, batch in enumerate(train_img_loader):
                # run policy to collect trajactories for a batch of images
                env.set_data(batch)
                obs_fov = env.observe()
                states = pack_model_inputs(obs_fov, env)
                act, _, _ = agent.select_action(
                    states,
                    True,
                    action_mask=env.action_mask,
                    sample_stop=env.pa.has_stop,
                )
                i_step = 0
                while i_step < hparams.Data.max_traj_length and env.status.min() < 1:
                    i_step += 1
                    new_obs_fov, curr_status = env.step(act)

                    new_states = pack_model_inputs(new_obs_fov, env)
                    online_memory_replay.add_batch(
                        (states, new_states, act, curr_status,
                        batch['centermaps'] if hparams.Train.use_det_loss else curr_status))
                    states = new_states
                    act, _, _ = agent.select_action(
                        states,
                        True,
                        action_mask=env.action_mask,
                        sample_stop=env.pa.has_stop,
                    )

                    if online_memory_replay.size() <= initial_memory:
                        continue

                    # Start learning
                    policy_batch, expert_batch, expert_demos_iter = get_batch(
                        online_memory_replay,
                        expert_demos_iter,
                        train_gaze_loader,
                        hparams.Train
                    )
                    
                    losses = agent.irl_update(
                        policy_batch,
                        expert_batch,
                        global_step,
                        hparams.Data.patch_num,
                    )

                    if global_step % print_every == print_every - 1 and losses is not None:
                        time = datetime.datetime.now()
                        eta = str((time - last_time) / print_every * (
                            max_iters - global_step))
                        last_time = time
                        time = str(time)
                        log_txt = "[{}], eta: {}, iter: {}, progress: {:.3f}, epoch: {}".format(
                            time[time.rfind(' ') + 1:time.rfind('.')],
                            eta[:eta.rfind('.')], global_step,
                            (global_step / max_iters) * 100, i_epoch)
                        for k, v in losses.items():
                            log_txt += ", {}: {:.3f}".format(k, v)
                        print(log_txt)
                        log_dict(writer, losses, global_step, 'train')

                    # Evaluate
                    if global_step % eval_every == eval_every - 1:
                        rst_tp, rst_ta, rst_fv = run_evaluation()
                        if rst_tp is not None:
                            log_dict(writer, rst_tp, global_step, "eval_TP")
                        if rst_ta is not None:
                            log_dict(writer, rst_ta, global_step, "eval_TA")
                        if rst_fv is not None:
                            log_dict(writer, rst_fv, global_step, "eval_FV")

                        writer.add_scalar('epoch', i_epoch, global_step)
                        os.system(f"cp {log_dir}/events* {log_folder_runs}")

                    if global_step % save_every == save_every - 1:
                        save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")
                        torch.save(
                            {
                                'model': agent.q_net.state_dict(),
                                'optimizer': agent.critic_optimizer.state_dict(),
                                'step': global_step + 1,
                            },
                            save_path,
                        )
                        print(f"Saved checkpoint to {save_path}.")
                    global_step += 1
                if global_step >= max_iters:
                    print("Exit training!")
                    break
            else:
                continue
            break  # Break outer loop

        # Copy to log file to ./runs
        os.system(f"cp {log_dir}/events* {log_folder_runs}")
