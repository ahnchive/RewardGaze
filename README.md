# Reward Gaze

Official PyTorch implementation of the paper:  
**Reward-based Modeling of Goal-directed Gaze Control**

---

## Overview

Goal-directed visual search in natural scenes is a complex behavior that requires flexible integration of vision, memory, and contextual knowledge.  
This repository introduces a **reward-based framework** that unifies these processes by learning target-specific reward functions directly from human fixation data using **inverse reinforcement learning (IRL)**.

The paper is currently under review. Please cite this repository until an official citation is available.


```bibtex
@misc{rewardgaze2025,
  title   = {Reward-based Modeling of Goal-directed Gaze Control},
  author  = {Ahn, Seo-Young},
  year    = {2025},
  note    = {Under review},
  howpublished = {\url{https://github.com/RewardGaze}}
}
```


---

## Repository Contents

- **Training script** for the Reward Gaze model.  
- **Analysis notebooks** used in the paper.  
- **Pretrained models** (checkpoints provided under `./pretrained`).  
- **Configs** for running experiments (`./configs`).  
- **Baseliness** for running baseline model predictions (`./baselines`).  

This repo is under **active maintenance**.

---

## Dataset Preparation

The current model is trained on the [COCO-Search18](https://sites.google.com/view/cocosearch/home) dataset.  

### dataset root structure
   ```
   <dataset_root>
       ├── bbox_annos.npy
       ├── coco_search18_fixations_TP_train.json
       ├── coco_search18_fixations_TP_validation.json
       └── DCBs/
           ├── HR/   # High-resolution belief maps
           └── LR/   # Low-resolution belief maps
   ```

**Note**: In this paper, images and fixations were rescaled to **512×320**. The original dataset was collected on a **1680×1050 display**.  
Precomputed belief maps and rescaled fixations used in the paper can be downloaded [here](https://drive.google.com/open?id=1spD2_Eya5S5zOBO3NKILlAjMEC3_gKWc).

---

## Installation

- **Python**: 3.10  
- **detectron2**
- **torch** 
- **torchvision**  
- **torchaudio**  
- **opencv-python==4.7.0.72**  
- **scikit-learn==1.2.1**  
- **scipy==1.10.1**  
- **numpy==1.23.0**  
- **pandas**

## Pretrained Models

- Model checkpoints are stored under `./pretrained`.  
- Each configuration file for training/evaluation is stored in `./configs`.

---

## Training & Evaluation

### Train a model
```bash
python train_rewardmap.py \
    --hparams ./configs/coco_search18_eDCB_IQL_TP_LIOR.json \
    --dataset-root <dataset_root>
```

### Evaluate a trained model
- Update `log_dir` in the config file to point to your experiment directory.
```bash
python train_rewardmap.py \
    --hparams ./configs/coco_search18_eDCB_IQL_TP_LIOR.json \
    --dataset-root <dataset_root> \
    --eval-only
```

### Save intermediate results
- Includes return maps, immediate reward maps, and probability maps.
```bash
python train_rewardmap.py \
    --hparams ./configs/coco_search18_eDCB_IQL_TP_LIOR.json \
    --dataset-root <dataset_root> \
    --eval-only \
    --save-visualization
```
