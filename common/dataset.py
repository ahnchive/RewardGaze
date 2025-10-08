from os.path import join
from torchvision import transforms
from torch.nn import Identity
import numpy as np
from pycocotools.coco import COCO
from .data import DCB_IRL, DCB_Human_Gaze
from .utils import compute_search_cdf, preprocess_fixations
from .utils import cutFixOnTarget, get_prior_maps
from .data import CFI_IRL, CFI_Human_Gaze
from .data import FFN_IRL, FFN_Human_Gaze, VDTrans_Human_Gaze, SPTrans_Human_Gaze


def process_data(target_trajs,
                 dataset_root,
                 target_annos,
                 hparams,
                 target_trajs_all,
                 is_testing=False,
                 sample_scanpath=False,
                 use_coco_annotation=False,
                 out_of_subject_eval=False):

    print("using", hparams.Train.repr, 'dataset:', hparams.Data.name, 'TAP:',
          hparams.Data.TAP)
    if use_coco_annotation:
        annFile = join(dataset_root, hparams.Data.coco_anno_dir,
                       'instances_train2017.json')
        coco_annos_train = COCO(annFile)
        annFile = join(dataset_root, hparams.Data.coco_anno_dir,
                       'instances_val2017.json')
        coco_annos_val = COCO(annFile)
        coco_annos = (coco_annos_train, coco_annos_val)
    else:
        coco_annos = None

    # Rescale fixations and images if necessary
    is_osie = hparams.Data.name == 'OSIE'
    if is_osie:
        ori_h, ori_w = 600, 800
        rescale_flag = hparams.Data.im_h != ori_h
    else:
        ori_h, ori_w = 320, 512
        rescale_flag = hparams.Data.im_h != ori_h
    if rescale_flag:
        print(
            f"Rescaling image and fixation to {hparams.Data.im_h}x{hparams.Data.im_w}"
        )
        size = (hparams.Data.im_h, hparams.Data.im_w)
        ratio_h = hparams.Data.im_h / ori_h
        ratio_w = hparams.Data.im_w / ori_w
        for traj in target_trajs_all:
            traj['X'] = np.array(traj['X']) * ratio_w
            traj['Y'] = np.array(traj['Y']) * ratio_h
            traj['rescaled'] = True

    if hparams.Train.repr == 'DCB':
        DCB_HR_dir = join(dataset_root, 'DCBs/HR/')
        DCB_LR_dir = join(dataset_root, 'DCBs/LR/')
    elif hparams.Train.repr == 'CFI' or hparams.Train.repr == 'FFN':

        transform_train = transforms.Compose([
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.Resize(size=size) if rescale_flag else Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size=size) if rescale_flag else Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError

    valid_target_trajs_all = list(
        filter(lambda x: x['split'] == 'test', target_trajs_all))
    if is_osie and not rescale_flag:
        fix_clusters = np.load(f'{dataset_root}/clusters_600x800.npy',
                               allow_pickle=True).item()
    else:
        fix_clusters = np.load(f'{dataset_root}/clusters_testval.npy', 
                               allow_pickle=True).item()
    for k, v in fix_clusters.items():
        if isinstance(v['strings'], list):
            break
        # remove other subjects' data if "subject" is specified
        if hparams.Data.subject > -1:
            try:
                v['strings'] = [v['strings'][hparams.Data.subject]]
            except:
                v['strings'] = []
        else:
            v['strings'] = list(v['strings'].values())
    if is_osie:
        scene_labels = None
    else:
        scene_labels = np.load(f'{dataset_root}/scene_label_dict.npy',
                               allow_pickle=True).item()

    target_init_fixs = {}
    for traj in target_trajs_all:
        key = traj['task'] + '_' + traj['name']
        if hparams.Data.name == 'COCO-Search18':
            # Force center initialization for COCO-Search18
            target_init_fixs[key] = (0.5, 0.5)  
        else:
            target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                     traj['Y'][0] / hparams.Data.im_h)
    if hparams.Train.zero_shot:
        catIds = np.load(join(dataset_root, 'all_target_ids.npy'),
                         allow_pickle=True).item()
    elif hparams.ablate_dcb:
        catIds = {'bottle': 0, 'bowl': 1, 'car': 2, 'chair': 3, 'clock': 4, 'cup': 5, 'fork': 6, 'keyboard': 7, 'knife': 8, 'laptop': 9, 'microwave': 10, 'mouse': 11, 'oven': 12, 'potted plant': 13, 'sink': 14, 'stop sign': 15, 'toilet': 16, 'tv': 17}
        
    else:
        cat_names = list(np.unique([x['task'] for x in target_trajs]))
        catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    human_mean_cdf = None
    if is_testing:
        # testing fixation data
        test_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        assert len(test_target_trajs) > 0, 'no testing data found!'
        test_task_img_pair = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in test_target_trajs
        ])

        # print statistics
        traj_lens = list(map(lambda x: x['length'], test_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of train trajs = {}'.format(len(test_target_trajs)))

        if hparams.Data.TAP == 'TP':
            human_mean_cdf, _ = compute_search_cdf(
                test_target_trajs, target_annos, hparams.Data.max_traj_length)
            print('target fixation prob (test).:', human_mean_cdf)

        # load image data
        if hparams.Train.repr == 'DCB':
            test_img_dataset = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                       target_init_fixs, test_task_img_pair,
                                       target_annos, hparams.Data, catIds)
        elif hparams.Train.repr == 'CFI':
            test_img_dataset = CFI_IRL(dataset_root, target_init_fixs,
                                       test_task_img_pair, target_annos,
                                       transform_test, hparams.Data, catIds)
        elif hparams.Train.repr == 'FFN':
            test_img_dataset = FFN_IRL(dataset_root, target_init_fixs,
                                       test_task_img_pair, target_annos,
                                       transform_test, hparams.Data, catIds)

        return {
            'catIds': catIds,
            'img_test': test_img_dataset,
            'bbox_annos': target_annos,
            'gt_scanpaths': test_target_trajs,
            'fix_clusters': fix_clusters
        }

    else:
        # training fixation data
        train_target_trajs = list(
            filter(lambda x: x['split'] == 'train', target_trajs))
        # print statistics
        traj_lens = list(map(lambda x: x['length'], train_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of train trajs = {}'.format(len(train_target_trajs)))

        train_task_img_pair = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in train_target_trajs
        ])
        train_fix_labels = preprocess_fixations(
            train_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            # truncate_num=hparams.Data.max_traj_length,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_osie=is_osie,
        )

        # validation fixation data
        valid_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        # print statistics
        traj_lens = list(map(lambda x: x['length'], valid_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print('average valid scanpath length : {:.3f} (+/-{:.3f})'.format(
            avg_traj_len, std_traj_len))
        print('num of valid trajs = {}'.format(len(valid_target_trajs)))

        valid_task_img_pair = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs
        ])

        tp_trajs = list(
            filter(
                lambda x: x['condition'] == 'present' and x['split'] == 'test',
                target_trajs_all))
        if len(tp_trajs) > 0:
            human_mean_cdf, _ = compute_search_cdf(
                tp_trajs, target_annos, hparams.Data.max_traj_length)
            print('target fixation prob (valid).:', human_mean_cdf)

        valid_fix_labels = preprocess_fixations(
            valid_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_osie=is_osie,
        )
        valid_target_trajs_TP = list(
            filter(lambda x: x['condition'] == 'present',
                   valid_target_trajs_all))
        valid_fix_labels_TP = preprocess_fixations(
            valid_target_trajs_TP,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            is_osie=is_osie,
        )
        valid_target_trajs_TA = list(
            filter(lambda x: x['condition'] == 'absent',
                   valid_target_trajs_all))
        valid_fix_labels_TA = preprocess_fixations(
            valid_target_trajs_TA,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_osie=is_osie,
        )
        valid_target_trajs_FV = list(
            filter(lambda x: x['condition'] == 'freeview',
                    valid_target_trajs_all))
        if len(valid_target_trajs_FV) > 0:
            valid_fix_labels_FV = preprocess_fixations(
                valid_target_trajs_FV,
                hparams.Data.patch_size,
                hparams.Data.patch_num,
                hparams.Data.im_h,
                hparams.Data.im_w,
                has_stop=hparams.Data.has_stop,
                sample_scanpath=sample_scanpath,
                discretize_fix=hparams.Data.discretize_fix,
                is_osie=is_osie,
            )
        else:
            valid_fix_labels_FV = []
        valid_fix_labels_all = preprocess_fixations(
            valid_target_trajs_all,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_osie=is_osie,
        )
        valid_task_img_pair_TP = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'present'
        ])
        valid_task_img_pair_TA = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'absent'
        ])
        valid_task_img_pair_FV = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'freeview'
        ])
        valid_task_img_pair_all = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
        ])
        if hparams.Train.repr == 'DCB':
            # load image data
            train_img_dataset = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                        target_init_fixs, train_task_img_pair,
                                        target_annos, hparams.Data, catIds)
            valid_img_dataset_TP = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                           target_init_fixs,
                                           valid_task_img_pair_TP,
                                           target_annos, hparams.Data, catIds)
            valid_img_dataset_TA = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                           target_init_fixs,
                                           valid_task_img_pair_TA,
                                           target_annos, hparams.Data, catIds)
            valid_img_dataset_FV = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                           target_init_fixs,
                                           valid_task_img_pair_FV,
                                           target_annos, hparams.Data, catIds)
            valid_img_dataset_all = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                            target_init_fixs,
                                            valid_task_img_pair_all,
                                            target_annos, hparams.Data, catIds)

            # load human gaze data
            train_HG_dataset = DCB_Human_Gaze(DCB_HR_dir,
                                              DCB_LR_dir,
                                              train_fix_labels,
                                              train_task_img_pair,
                                              target_annos,
                                              hparams.Data,
                                              catIds,
                                              mix_match=hparams.Data.mix_match)
            valid_HG_dataset = DCB_Human_Gaze(
                DCB_HR_dir,
                DCB_LR_dir,
                valid_fix_labels,
                valid_task_img_pair,
                target_annos,
                hparams.Data,
                catIds,
                blur_action=hparams.Data.blur_action)

            valid_HG_dataset_TP = DCB_Human_Gaze(
                DCB_HR_dir,
                DCB_LR_dir,
                valid_fix_labels_TP,
                valid_task_img_pair_TP,
                target_annos,
                hparams.Data,
                catIds,
                blur_action=hparams.Data.blur_action)

            valid_HG_dataset_TA = DCB_Human_Gaze(
                DCB_HR_dir,
                DCB_LR_dir,
                valid_fix_labels_TA,
                valid_task_img_pair_TA,
                target_annos,
                hparams.Data,
                catIds,
                blur_action=hparams.Data.blur_action)
            if valid_fix_labels_FV:
                valid_HG_dataset_FV = DCB_Human_Gaze(
                    DCB_HR_dir,
                    DCB_LR_dir,
                    valid_fix_labels_FV,
                    valid_task_img_pair_FV,
                    target_annos,
                    hparams.Data,
                    catIds,
                    blur_action=hparams.Data.blur_action)
            else:
                valid_HG_dataset_FV = None
            valid_HG_dataset_all = DCB_Human_Gaze(
                DCB_HR_dir,
                DCB_LR_dir,
                valid_fix_labels_all,
                valid_task_img_pair_all,
                target_annos,
                hparams.Data,
                catIds,
                blur_action=hparams.Data.blur_action)

        elif hparams.Train.repr == 'CFI':
            # load image data
            train_img_dataset = CFI_IRL(dataset_root, target_init_fixs,
                                        train_task_img_pair, target_annos,
                                        transform_train, hparams.Data, catIds)
            valid_img_dataset_TP = CFI_IRL(dataset_root, target_init_fixs,
                                           valid_task_img_pair_TP,
                                           target_annos, transform_train,
                                           hparams.Data, catIds)
            valid_img_dataset_TA = CFI_IRL(dataset_root, target_init_fixs,
                                           valid_task_img_pair_TA,
                                           target_annos, transform_train,
                                           hparams.Data, catIds)

            if hparams.Data.TAP == 'TAP':
                valid_img_info_TP = list(
                    filter(lambda x: x.split('_')[-1] == 'present',
                           valid_task_img_pair))
                valid_img_info_TA = list(
                    set(valid_task_img_pair) - set(valid_img_info_TP))
                valid_TP_imgset = CFI_IRL(dataset_root, target_init_fixs,
                                          valid_img_info_TP, target_annos,
                                          transform_test, hparams.Data, catIds)
                valid_TA_imgset = CFI_IRL(dataset_root, target_init_fixs,
                                          valid_img_info_TA, target_annos,
                                          transform_test, hparams.Data, catIds)
                valid_img_dataset = {
                    "TP": valid_TP_imgset,
                    "TA": valid_TA_imgset
                }
            else:
                valid_img_dataset = CFI_IRL(dataset_root, target_init_fixs,
                                            valid_task_img_pair, target_annos,
                                            transform_test, hparams.Data,
                                            catIds)

            # load human gaze data
            train_HG_dataset = CFI_Human_Gaze(dataset_root,
                                              train_fix_labels,
                                              target_annos,
                                              scene_labels,
                                              hparams.Data,
                                              transform_train,
                                              catIds,
                                              blur_action=False)
            valid_HG_dataset = CFI_Human_Gaze(dataset_root,
                                              valid_fix_labels,
                                              target_annos,
                                              scene_labels,
                                              hparams.Data,
                                              transform_test,
                                              catIds,
                                              blur_action=True)
            valid_HG_dataset_FV = None
        elif hparams.Train.repr == 'FFN':
            # load image data
            train_img_dataset = FFN_IRL(dataset_root, None,
                                        train_task_img_pair, target_annos,
                                        transform_train, hparams.Data, catIds,
                                        coco_annos=coco_annos)
            valid_img_dataset_all = FFN_IRL(dataset_root, None,
                                            valid_task_img_pair_all, target_annos,
                                            transform_test, hparams.Data, catIds,
                                            coco_annos=None)
            valid_img_dataset_TP = FFN_IRL(dataset_root, None,
                                           valid_task_img_pair_TP, target_annos,
                                           transform_test, hparams.Data,
                                           catIds, coco_annos=None)
            valid_img_dataset_TA = FFN_IRL(dataset_root, None,
                                           valid_task_img_pair_TA,
                                           target_annos, transform_test,
                                           hparams.Data, catIds, None)
            valid_img_dataset_FV = FFN_IRL(dataset_root, None,
                                           valid_task_img_pair_FV,
                                           target_annos, transform_test,
                                           hparams.Data, catIds, None)

            if hparams.Model.name == 'FoveatedFeatureNet':
                gaze_dataset_func = FFN_Human_Gaze
            elif hparams.Model.name == 'VDTransformer':
                gaze_dataset_func = VDTrans_Human_Gaze
            elif hparams.Model.name == 'HAT':
                gaze_dataset_func = SPTrans_Human_Gaze
            elif hparams.Model.name == 'FOM':
                gaze_dataset_func = SPTrans_Human_Gaze    
            else:
                raise NotImplementedError

            train_HG_dataset = gaze_dataset_func(dataset_root,
                                                 train_fix_labels,
                                                 target_annos,
                                                 scene_labels,
                                                 hparams.Data,
                                                 transform_train,
                                                 catIds,
                                                 blur_action=True,
                                                 coco_annos=coco_annos)
            valid_HG_dataset = gaze_dataset_func(dataset_root,
                                                 valid_fix_labels,
                                                 target_annos,
                                                 scene_labels,
                                                 hparams.Data,
                                                 transform_test,
                                                 catIds,
                                                 blur_action=True,
                                                 coco_annos=None)
            valid_HG_dataset_TP = gaze_dataset_func(dataset_root,
                                                    valid_fix_labels_TP,
                                                    target_annos,
                                                    scene_labels,
                                                    hparams.Data,
                                                    transform_test,
                                                    catIds,
                                                    blur_action=True,
                                                    coco_annos=None)
            valid_HG_dataset_TA = gaze_dataset_func(dataset_root,
                                                    valid_fix_labels_TA,
                                                    target_annos,
                                                    scene_labels,
                                                    hparams.Data,
                                                    transform_test,
                                                    catIds,
                                                    blur_action=True,
                                                    coco_annos=None)
            if len(valid_target_trajs_FV) > 0:
                valid_HG_dataset_FV = gaze_dataset_func(dataset_root,
                                                        valid_fix_labels_FV,
                                                        target_annos,
                                                        scene_labels,
                                                        hparams.Data,
                                                        transform_test,
                                                        catIds,
                                                        blur_action=True,
                                                        coco_annos=None)
            else:
                valid_HG_dataset_FV = None
            valid_HG_dataset_all = gaze_dataset_func(dataset_root,
                                                     valid_fix_labels_all,
                                                     target_annos,
                                                     scene_labels,
                                                     hparams.Data,
                                                     transform_test,
                                                     catIds,
                                                     blur_action=True,
                                                     coco_annos=None)

        cutFixOnTarget(target_trajs, target_annos)
        print("num of training and eval fixations = {}, {}".format(
            len(train_HG_dataset), len(valid_HG_dataset)))
        print("num of training and eval images = {}, {} (TP), {} (TA), {}(FV)".
              format(len(train_img_dataset), len(valid_img_dataset_TP),
                     len(valid_img_dataset_TA), len(valid_img_dataset_FV)))

        return {
            'catIds': catIds,
            'img_train': train_img_dataset,
            'img_valid_TP': valid_img_dataset_TP,
            'img_valid_TA': valid_img_dataset_TA,
            'img_valid_FV': valid_img_dataset_FV,
            'img_valid': valid_img_dataset_all,
            'gaze_train': train_HG_dataset,
            'gaze_valid': valid_HG_dataset,
            'gaze_valid_TP': valid_HG_dataset_TP,
            'gaze_valid_TA': valid_HG_dataset_TA,
            'gaze_valid_FV': valid_HG_dataset_FV,
            'bbox_annos': target_annos,
            'fix_clusters': fix_clusters,
            'valid_scanpaths': valid_target_trajs_all,
            'human_cdf': human_mean_cdf,
        }
