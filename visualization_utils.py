

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Mapping, Tuple, Optional, List

import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
import random

from fixation_density import *




# ───────────────────────────────────────────────────────────────
# general helpers
# ───────────────────────────────────────────────────────────────



def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    return (np.sqrt((X - x)**2 + (Y - y)**2) <= r).astype(np.uint8)

def minmax_normalize(m):
    m = m.astype(np.float32)
    return (m - m.min()) / (m.max() - m.min() + 1e-8) 

def compute_reward(m, mask):
    return (m * mask).sum()

def format_p(p):
    if p < 0.001:
        return 'p <.001'
    elif p < 0.01:
        return 'p <.01'
    elif p < 0.05:
        return 'p <.05'
    else:
        return 'p >.05'



# ───────────────────────────────────────────────────────────────
# reward maps plotting
# ───────────────────────────────────────────────────────────────


@dataclass
class BaselineSpec:
    name: str                 # e.g., "detector", "hat"
    pred_path: str            # e.g., "./baselines/Foveated Detector/pred_{tap}_detection_hilow_aug.npy"
                             # you can use "{tap}" inside if you want to parameterize by cfg.tap

@dataclass
class VizConfig:
    """All tunable parameters for the visualisation."""
    n_visualisations: int = 50
    maps_to_visualize: Sequence[str] = None
    plot_entropy: bool = True
    imglist_to_save: Optional[Sequence[str]] = None
    imglist_to_visualize: Sequence[str] | str = "all"
    tap: Optional[str] = None                   
    save_path: Optional[str] = None       
    overlay: bool = True
    alpha: float = 0.15                    # softmax temperature for reward maps
    marker_radius: int = 20               # circle radius for fixations
    arrow_width: int = 3                  # arrow thickness
    baselines: Sequence[BaselineSpec] = ()
    draw_nstep: Optional[int] = None       # number of steps to draw, if None, all model steps are drawn


def _draw_blank_cell(ax, title: str | None = None, keep_title: bool = True):
    # wipe anything that might have been there
    ax.clear()
    # optional column header
    if keep_title and title:
        ax.set_title(title)
    # kill all axis visuals
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_frame_on(False)
    ax.set_axis_off()
    # make the axes patch fully transparent
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
    ax.patch.set_visible(False)


def _load_and_resize_image(path: Path, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(path)
    if img.size != size:
        img = img.resize(size)
    return img


def _choose_human_scanpath(
    human_scanpaths: Sequence[Mapping],
    task_image_label: str,
    n_steps: int,
) -> Tuple[List[Tuple[int, int, float]], bool]:
    """Return a representative human scanpath of sufficient length."""
    scanpaths_image = [
        s for s in human_scanpaths if f"{s['task']}_{s['name']}" == task_image_label
    ]
    candidates = []
    for sp in scanpaths_image:
        if len(sp["X"]) >= n_steps:
            fixs = list(zip(sp["X"], sp["Y"], sp["T"]))
            candidates.append(fixs)
    if candidates:
        return random.choice(candidates), True
    # nobody long enough
    print("All human scanpaths shorter; will not draw human fixations.")
    return [], False



def _draw_reward_map(ax, return_map, img_orig, cfg: VizConfig, title: str | None):
    _setup_axis(ax, title)
    overlay = (
        _normalize_and_overlay(return_map, img_orig, softmax=False, alpha=cfg.alpha)
        if cfg.overlay else return_map
    )
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

# def _draw_prob_map(ax, prob_map, img_orig, cfg: VizConfig, title: str | None):
#     _setup_axis(ax, title)
#     if cfg.plot_entropy:
#         entropy = stats.entropy(prob_map.flatten(), base=2)
#         ax.text(
#             prob_map.shape[1] - 2,
#             prob_map.shape[0] - 10,
#             f"{entropy:.2f}",
#             color="white",
#             fontsize=20,
#             ha="right",
#         )

#     overlay = (
#         _normalize_and_overlay(prob_map, img_orig, softmax=False, alpha=cfg.alpha)
#         if cfg.overlay
#         else prob_map
#     )
#     ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def _draw_prob_map(ax, prob_map, img_orig, cfg, title: str | None, mark_max=False):
    # max_entropy = np.log(512*320)/np.log(2)
    _setup_axis(ax, title)
    if cfg.plot_entropy:
        entropy = stats.entropy(prob_map.flatten(), base=2)
        ax.text(prob_map.shape[1]-2, prob_map.shape[0]-10, f"{entropy:.2f}",
                color="white", fontsize=20, ha="right")

    overlay = (_normalize_and_overlay(prob_map, img_orig, softmax=False, alpha=cfg.alpha)
               if cfg.overlay else prob_map)
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    if mark_max:
        y, x = np.unravel_index(np.nanargmax(prob_map), prob_map.shape)
        ax.add_patch(plt.Circle((x, y), radius=8, fill=False,
                                edgecolor='blue', linewidth=2))

def _draw_scanpath(
    ax,
    img: Image.Image,
    fixations: Sequence[Tuple[int, int]],
    step: int,
    cfg: VizConfig,
    title: str | None,
):
    _setup_axis(ax, title)

    # 1) Show background image
    ax.imshow(img)

    # 2) Lock the axes to the image extent and stop autoscaling
    ax.set_xlim(0, img.size[0])        # x spans full image width
    ax.set_ylim(img.size[1], 0)        # y spans full image height (origin upper‑left)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(False)                # or ax.set_autoscale_on(False)

    # 3) Draw the scan‑path
    r = cfg.marker_radius
    for s in range(step + 1):
        ax.arrow(
            fixations[s][0],
            fixations[s][1],
            fixations[s + 1][0] - fixations[s][0],
            fixations[s + 1][1] - fixations[s][1],
            width=cfg.arrow_width,
            color="yellow",
            clip_on=False,             # don't expand limits if arrow pokes out
        )
        _draw_fixation_circle(ax, fixations[s], r, label=s + 1)

    _draw_fixation_circle(
        ax, fixations[step + 1], r, label=step + 2, highlight=True
    )
    
def _draw_fixation_circle(ax, xy, radius, label: int, highlight: bool = False):
    circle = plt.Circle(
        xy,
        radius=radius,
        edgecolor="red",
        facecolor="yellow",
        alpha=0.5,
        clip_on=False,             # allow parts outside axes without padding
    )
    ax.add_patch(circle)
    ax.annotate(
        str(label),
        xy=(xy[0], xy[1] + 3),
        weight="bold",
        fontsize=15,
        color="red" if highlight else "black",
        ha="center",
        va="center",
        clip_on=False,
    )

def _setup_axis(ax, title: str | None):
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _normalize_and_overlay(m, img, softmax=True, alpha=0.15):
    if softmax:
        m = torch.Tensor(m).flatten()
        m = F.softmax(m/alpha)
        m = np.reshape(m.numpy(), (320, 512))
    
    m = (m - m.min()) / (m.max() - m.min()) * 255 # normalize to 0-255
    m = m.astype(np.uint8)
    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    m = cv2.addWeighted(m, 0.5, img, 0.5, 0)
    return m


def visualize_rewardmaps(
    vis_category: Sequence,               # your pre‑computed vis tuples
    scanpath_category: Sequence,          # model scanpaths aligned with vis_category
    human_scanpaths: Sequence[Mapping],   # list of dicts with keys 'task', 'name', 'X', 'Y', 'T'
    dataset_root: str | Path,
    hparams,                              # lightning / OmegaConf like object
    cfg: VizConfig = VizConfig()
) -> None:
    """Main entry point.  Loops through ``vis_category`` and renders figures."""

    # check valid maps
    ALLOWED_MAPS = {
        "immediate reward",
        "expected future reward",
        "total return",
        "action prob",
        "reward scanpath",
        "baseline attention map",
        "baseline scanpath",
        "human scanpath",
        "human fdm",
    }
    invalid = [m for m in cfg.maps_to_visualize if m not in ALLOWED_MAPS]
    if invalid:
        raise ValueError(
            f"Unrecognized map type(s): {invalid}. "
            f"Allowed values are: {sorted(ALLOWED_MAPS)}"
        )

    # save directory
    dataset_root = Path(dataset_root)
    n_vis = min(cfg.n_visualisations, len(vis_category))

    if len(cfg.imglist_to_save) > 0 and cfg.save_path is None:
        raise ValueError("specify save_dir to save images")
    

    if cfg.save_path is not None:
        if Path(cfg.save_path).exists():
            raise FileNotFoundError(f"the current save_path exist, check for overwrite")

    # ───────────────────────────────────────────────────────────────────────────
    # Baseline predictions loading (supports multiple baselines)
    # ───────────────────────────────────────────────────────────────────────────
    baselines_data = {}  # name -> loaded npy (list of dicts)
    if cfg.baselines and len(cfg.baselines) > 0:
        for b in cfg.baselines:
            # If pred_path includes "{tap}", format it using cfg.tap
            path = b.pred_path.format(tap=cfg.tap) if "{tap}" in b.pred_path else b.pred_path
            arr = np.load(path, allow_pickle=True)
            baselines_data[b.name] = arr
            print(f"Loaded baseline '{b.name}' from {path} with {len(arr)} items")

        
    for idx in range(n_vis):
        vis = vis_category[idx]
        sp = scanpath_category[idx]

        task_name, image_name = vis[0], vis[1]
        task_image_label = f"{task_name}_{image_name}"

        # filter by user‑requested subset
        if cfg.imglist_to_visualize != "all" and image_name not in cfg.imglist_to_visualize:
            continue

        print(f"Rendering {image_name}")
        imgpath = dataset_root / f"images/{task_name.replace(' ', '_')}/{image_name}"
        img = _load_and_resize_image(imgpath, (hparams.Data.im_w, hparams.Data.im_h))
        img_orig = np.array(img.copy())

        model_fixs: List[Tuple[int, int]] = list(zip(sp["X"], sp["Y"]))
        if cfg.draw_nstep is not None:
            n_steps = cfg.draw_nstep
        else:
            n_steps = len(model_fixs)

        human_fixs, matched_humanscanpath_available = _choose_human_scanpath(
            human_scanpaths, task_image_label, n_steps
        )


        # ------------------------------------------------------------------ #
        # figure layout
        # ------------------------------------------------------------------ #
        n_rows = n_steps - 1

        # Count non-baseline visualizations first
        base_map_types = set(cfg.maps_to_visualize or [])
        # these two keys trigger per-baseline columns:
        want_baseline_map = "baseline attention map" in base_map_types
        want_baseline_sp  = "baseline scanpath"      in base_map_types

        # Count columns: non-baseline entries (each adds 1 col)
        non_baseline_cols = sum(mt not in {"baseline attention map", "baseline scanpath"} for mt in base_map_types)

        # Add baseline columns: 2 per baseline if enabled in maps_to_visualize
        n_baselines = len(baselines_data)
        baseline_cols = (n_baselines if want_baseline_map else 0) + (n_baselines if want_baseline_sp else 0)

        n_cols = non_baseline_cols + baseline_cols

        # If human scanpath requested but not available, subtract 1 (only for that single col)
        if 'human scanpath' in base_map_types and not matched_humanscanpath_available:
            n_cols -= 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        axs = axs.flatten()

        for step in range(n_rows):        
            
            # vis = (cat_names_batch[i], img_names_batch[i], qs, rs, ps, nextVs, gamma) 
            # q:total return action value, r: immediate reward, p: prob,  # rs = qs - policy.gamma * next_Vs

            subfig_i = 0
            baseline_pair_drawn = False      # NEW
            for map_type in cfg.maps_to_visualize:

                if map_type == "human scanpath":
                    if matched_humanscanpath_available:
                        _draw_scanpath(
                            ax=axs[step * n_cols + subfig_i],
                            img=img,
                            fixations=human_fixs,
                            step=step,
                            cfg=cfg,
                            title="human scanpath" if step == 0 else None,
                        )
                        subfig_i += 1

                if map_type == "human fdm":
                    ax = axs[step * n_cols + subfig_i]
                    if step == 0:
                        ax.set_title('Human fdm')
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # convert fixdata to tuple and aggregate across all participants  
                    scanpath_tuple = []
                    for scanpath in human_scanpaths:
                        if len(scanpath['X']) > step+1:
                            current_scanpath_tuple = list(zip([int(scanpath['X'][step+1])], [int(scanpath['Y'][step+1])], [scanpath['T'][step+1]]))
                            scanpath_tuple.extend(current_scanpath_tuple)


                    # get background image
                    tmpimg = cv2.imread(str(imgpath))
                    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

                    # heatmap variables 
                    display_height, display_width, _ = tmpimg.shape
                    alpha = 0.7  #transparancy of the heatmap
                    gaussianwh = 150
                    gaussiansd = 15

                    # draw fdm
                    draw_heatmap(scanpath_tuple, (display_width, display_height), alpha=alpha, savefilename=None, 
                                imagefile=imgpath, gaussianwh=gaussianwh, gaussiansd=gaussiansd, ax=ax)
                    
                    subfig_i += 1

                #######################
                # reward maps
                #######################

                if map_type == "immediate reward":
                    _draw_reward_map(
                        ax=axs[step * n_cols + subfig_i],
                        return_map=vis[3][step],
                        img_orig=img_orig,
                        cfg=cfg,
                        title="immediate reward" if step == 0 else None,
                    ) # return map q = r + gamma*v
                    subfig_i += 1


                if map_type == "expected future reward":
                    _draw_reward_map(
                        ax=axs[step * n_cols + subfig_i],
                        return_map=vis[5][step],
                        img_orig=img_orig,
                        cfg=cfg,
                        title="expected future reward" if step == 0 else None,
                    )
                    subfig_i += 1

                if map_type == "total return":
                    _draw_reward_map(
                        ax=axs[step * n_cols + subfig_i],
                        return_map=vis[2][step],
                        img_orig=img_orig,
                        cfg=cfg,
                        title="total return" if step == 0 else None,
                    )
                    subfig_i += 1

                if map_type == "action prob":
                    _draw_prob_map(
                        ax=axs[step * n_cols + subfig_i],
                        prob_map=vis[4][step],
                        img_orig=img_orig,
                        cfg=cfg,
                        title="action prob" if step == 0 else None,
                    )
                    subfig_i += 1

                            
                # prob is softmax version of r, code below is playing with temp param and normalization
                # alpha=0.15
    #             f = vis[4][step]
    #             f = (f- f.min()) / (f.max() - f.min()) 
    #             f = torch.Tensor(f).flatten()
    #             f = F.softmax(f/alpha)
    #             f = np.reshape(f.numpy(), (320, 512))

                if map_type == "reward scanpath":
                    _draw_scanpath(
                        ax=axs[step * n_cols + subfig_i],
                        img=img,
                        fixations=model_fixs,
                        step=step,
                        cfg=cfg,
                        title='reward scanpath' if step == 0 else None,
                    )
                    subfig_i += 1


                # ─────────────────────────────────────────────────────────────
                # BASELINE COLUMNS: draw as (map → scanpath) per baseline, ONCE
                # ─────────────────────────────────────────────────────────────
                # baselines are step+1 because they start from center, while reward start from 1 fixaiton
                if map_type in ["baseline attention map", "baseline scanpath"]:
                    if baseline_pair_drawn:
                        continue  # already drew the pair this step

                    want_baseline_map = "baseline attention map" in base_map_types
                    want_baseline_sp  = "baseline scanpath"      in base_map_types

                    for bname, bpred in baselines_data.items():
                        # find baseline record for this image/task
                        pred_list = [it for it in bpred if it['name'] == image_name and it['task'] == task_name]
                        pred = pred_list[0] if len(pred_list) > 0 else None

                        # --- reward/attention map column ---
                        if bname == 'detector':
                            bstep = step+1
                        else:
                            bstep = step
                        if want_baseline_map:
                            title = f"{bname} map" if step == 0 else None
                            if (pred is not None and 'attn_map' in pred
                                    and len(pred['attn_map']) > bstep
                                    and pred['attn_map'][bstep] is not None):
                                _draw_prob_map(
                                    ax=axs[step * n_cols + subfig_i],
                                    prob_map=pred['attn_map'][bstep],
                                    img_orig=img_orig,
                                    cfg=cfg,
                                    title=title,
                                    mark_max=False,
                                )
                            else:
                                _draw_blank_cell(axs[step * n_cols + subfig_i], title=title)
                            subfig_i += 1

                        # --- scanpath column ---
                        if want_baseline_sp:
                            title = f"{bname} scanpath" if step == 0 else None
                            if (pred is not None and 'X' in pred and 'Y' in pred
                                    and len(pred['X']) > step+1 and len(pred['Y']) > step+1):
                                _draw_scanpath(
                                    ax=axs[step * n_cols + subfig_i],
                                    img=img,
                                    fixations=list(zip(pred["X"], pred["Y"])),
                                    step=step,
                                    cfg=cfg,
                                    title=title,
                                )
                            else:
                                _draw_blank_cell(axs[step * n_cols + subfig_i], title=title)
                            subfig_i += 1

                    baseline_pair_drawn = True
                    continue

        fig.suptitle(task_image_label)
        plt.tight_layout()

        if cfg.imglist_to_save and image_name in cfg.imglist_to_save:
            this_save_path = cfg.save_path + f"{task_image_label}.png"
            plt.savefig(this_save_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {this_save_path}")

        plt.show()







# ───────────────────────────────────────────────────────────────
# reward depletion
# ───────────────────────────────────────────────────────────────


def _safe_linregress(x, y):
    """Return (slope, intercept, r, p) or (nan, nan, nan, nan) if degenerate."""
    try:
        if len(x) < 2 or np.allclose(np.max(x), np.min(x)):
            return np.nan, np.nan, np.nan, np.nan
        slope, intercept, r, p, _ = stats.linregress(x, y)
        return slope, intercept, r, p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

_COLOR_MAP = {'ours': "#0015FF", 'detector': "#000000", 'hat': "#814825"}
_LABEL_MAP = {'ours': "reward", 'detector': "detector", 'hat': "HAT"}

def _compute_stats_dict(df_cat):
    """Return {model: (r, p)} for one category sub-DF."""
    stats_dict = {}
    for model_name, g in df_cat.groupby('model'):
        _, _, r, p = _safe_linregress(g['backN'].values, g['IOR'].values)
        stats_dict[model_name] = (None, None) if np.isnan(r) else (r, p)
    return stats_dict



def plot_joint_multi(df, save_path, use_legend=True, legend_loc='lower left', title=None,
                     show_xlabel=True, show_ylabel=True, 
                     ylabel = 'priority depletion (%)', ytickslist = [0, 50, 100],
                     stats_dict=None, show_plot=False):  # new: dict of {model: (r, p)}
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # Style to match plot_joint
    sns.set_style('white', rc={'xtick.bottom': True, 'ytick.left': True})

    fig, ax = plt.subplots(figsize=(3, 3))

    # Explicit color/label mapping
    color_map = {
        'ours':    "#0015FF",
        'detector':   "#000000",
        'hat':      "#814825",
    }
    label_map = {
        'ours': "reward",
        'detector': "detector",
        'hat': "HAT",
    }

    # Plot in fixed order: ours → detector → hat
    ordered_models = [m for m in ['ours', 'detector', 'hat'] if m in df['model'].unique()]

    for m in ordered_models:
        g = df[df['model'] == m]
        if len(g) == 0:
            continue

        # Legend label: add r, p if available
        base_label = label_map[m]
        if stats_dict and m in stats_dict:
            r, p = stats_dict[m]
            if r is not None and p is not None:
                base_label = f"{base_label} (r={r:.2f}, p={p:.2f})"

        sns.regplot(
            data=g,
            x='backN',
            y='IOR',
            ax=ax,
            color=color_map[m],
            x_estimator=np.mean,
            x_ci=68,
            ci=68,
            label=base_label,
            scatter_kws={'s': 18, 'alpha': 0.9},
            line_kws={'linewidth': 2, 'alpha': 0.3}
        )

    # Axes setup (match plot_joint)
    ax.set_xlim(1, 5.1)
    ax.set_ylim(ytickslist[0]-0.1, 1.1)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=10)
    ax.set_yticks(ytickslist)
    ax.set_yticklabels([int(100*y) for y in ytickslist], fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Labels
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel("number of fixations", fontsize=11)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    if title:
        ax.set_title(title, fontsize=11)

    # Legend
    if use_legend:
        ax.legend(
            fontsize=6,
            frameon=True,
            loc=legend_loc,
            # bbox_to_anchor=(1.02, 0.5),
            # bbox_to_anchor=(0.02, 0.17),
            title=None
        )
        # ax.legend(
        #         fontsize=6,
        #         frameon=True,
        #         loc="best",   # let matplotlib automatically choose
        #         title=None
        #     )
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[saved] {save_path}')

    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_joint_multi_on_ax(df, ax,
                           use_legend=False,
                           legend_loc='lower left',
                           title=None,
                           show_xlabel=True,
                           show_ylabel=True,
                           ytickslist=[0, 50, 100],
                           ylabel = "priority depletion (%)",
                           stats_dict=None):
    """Same look as your plot_joint_multi, but draws on a provided Axes."""
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator
    sns.set_style('white', rc={'xtick.bottom': True, 'ytick.left': True})

    ordered_models = [m for m in ['ours', 'detector', 'hat'] if m in df['model'].unique()]

    for m in ordered_models:
        g = df[df['model'] == m]
        if len(g) == 0:
            continue

        # Legend label: add r, p if available
        base_label = _LABEL_MAP[m]
        if stats_dict and m in stats_dict:
            r, p = stats_dict[m]
            if r is not None and p is not None:
                base_label = f"{base_label} (r={r:.2f}, p={p:.2f})"

        sns.regplot(
            data=g,
            x='backN',
            y='IOR',
            ax=ax,
            color=_COLOR_MAP[m],
            x_estimator=np.mean,
            x_ci=68,
            ci=68,
            label=base_label,
            scatter_kws={'s': 18, 'alpha': 0.9},
            line_kws={'linewidth': 2, 'alpha': 0.3}
        )

    # Axes setup (match your original)
    ax.set_xlim(1, 5.1)
    ax.set_ylim(ytickslist[0]-0.1, 1.1)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=10)
    ax.set_yticks(ytickslist)
    ax.set_yticklabels([int(100*y) for y in ytickslist], fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Labels
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel("number of fixations", fontsize=11)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    if title:
        ax.set_title(title, fontsize=11)

    # Per-axes legend toggle (we usually disable in grid and use a fig-level legend)
    if use_legend:
        ax.legend(
            fontsize=6,
            frameon=True,
            loc=legend_loc,
            # bbox_to_anchor=(0.02, 0.17),
            title=None
        )
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()



def plot_all_categories_grid(
    idf,                          # long DF with columns: category, model, backN, IOR, ...
    gazestat,                     # DF that has per-category 'nfix' (for titles)
    save_path="./results/savefig/TP_reward_depletion_GRID_6x3.png",
    categories=None,              # optional explicit order/list of categories (length ≤ 18)
    nrows=6,
    ncols=3,
    ylabel='priority depletion (%)',
    ytickslist=[0, 50, 100],
    legend_loc='lower left',
):
    """
    Make one big figure with a 6×3 grid of per-category panels.
    - Only first column shows y-labels.
    - Only bottom row shows x-labels.
    - Single figure-level legend at bottom.
    """
    # Build nfix lookup for titles
    nfix_lookup = gazestat.set_index('category')['nfix'].to_dict()

    # Determine which categories to plot (up to nrows*ncols)
    if categories is None:
        categories = list(pd.unique(idf['category']))
        categories = sorted(categories)  # or your preferred order
    max_cells = nrows * ncols
    categories = categories[:max_cells]

    # Create grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows),  constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, cat in enumerate(categories):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        subdf = idf[idf['category'] == cat].copy()
        stats_dict = _compute_stats_dict(subdf)

        # Title with nfix (2 decimals) if available
        nfix_val = nfix_lookup.get(cat, np.nan)
        nfix_text = f"{nfix_val:.2f}" if pd.notna(nfix_val) else "NA"

        show_ylabel = (c == 0)
        show_xlabel = (r == nrows - 1)

        # Draw one panel
        plot_joint_multi_on_ax(
            subdf, ax,
            use_legend=True,                  # per-panel legend OFF
            title=f"{cat} ({nfix_text})",
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            ylabel = ylabel,
            ytickslist= ytickslist,
            legend_loc= legend_loc,
            stats_dict=stats_dict
        )

    # Turn off any unused axes (if fewer than nrows*ncols categories)
    for idx in range(len(categories), max_cells):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)



    # Spacing
    plt.subplots_adjust(
        left=0.08, right=0.98, top=0.96, bottom=0.07,
        hspace=0.45, wspace=0.25
    )

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {save_path}")


# def plot_joint(data, category, corr, pvalue, hvalue=None, ax=None, save_path=None, overlay_baseline=False):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from matplotlib.ticker import MultipleLocator

#     sns.set_style('white', rc={'xtick.bottom': True, 'ytick.left': True})

#     created_fig = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(3, 2.7))
#         created_fig = True

#     sns.regplot(
#         data=data,
#         x='backN',
#         y='IOR',
#         ax=ax,
#         color= "#0015B2",
#         x_estimator=np.mean,
#         x_ci=68,
#         ci=68
#     )


#     # Optionally overlay IOR_baseline
#     if overlay_baseline and 'IOR_baseline' in data.columns:
#         sns.regplot(
#             data=data,
#             x='backN',
#             y='IOR_baseline',
#             ax=ax,
#             color="#696966",
#             x_estimator=np.mean,
#             x_ci=68,
#             ci=68,
#             label='IOR_baseline'
#         )

#     mean = data.groupby('backN')['IOR'].mean().mean()
#     print(f"{category} average mean estimate: {mean:.3f}")

#     # Text annotations
#     ax.text(1.2, 0.1, f'r = {round(corr, 3)}, {pvalue}', fontstyle='italic')
#     label = f'{category}' if hvalue is None else f'{category} ({hvalue})'
#     ax.text(1.2, 0.17, label, fontweight="bold", fontsize=11)

#     # Axis formatting
#     ax.set_xlim(1, 5)
#     ax.set_ylim(0, 0.5)
#     ax.set_xticks(range(1, 6))
#     ax.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=10)
#     ax.set_yticks([0, 0.5, 1.0])
#     ax.set_yticklabels(['0', '50', '100'], fontsize=10)
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.set_xlabel("num of fixation", fontsize=11)
#     ax.set_ylabel("reward depletion (%)", fontsize=11)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Save or show
#     if created_fig:
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         else:
#             plt.show()

#     return mean


# def plot_joint(data, category, corr, pvalue, hvalue=None, save_path=None):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from matplotlib.ticker import MultipleLocator

#     sns.set_style('white', rc={'xtick.bottom': True, 'ytick.left': True})

#     g = sns.JointGrid(data=data, x='backN', y='IOR', xlim=(1, 5), ylim=(0, 0.5))
#     g = g.plot_joint(sns.regplot, color="#696966", x_estimator=np.mean, x_ci=68, ci=68) # if 68-> SE

#     mean = g.ax_joint.collections[0].get_offsets()[:, 1].mean()
#     print(f"{category} average mean estimate: {mean:.3f}")

#     g.ax_marg_x.remove()
#     g.ax_marg_y.remove()

#     g.ax_joint.text(1.2, 0.1, f'r = {round(corr, 3)}, {pvalue}', fontstyle='italic')
#     label = f'{category}' if hvalue is None else f'{category} ({hvalue})'
#     g.ax_joint.text(1.2, 0.17, label, fontweight="bold", fontsize=15)

#     g.fig.set_size_inches((3.5, 3))
#     g.set_axis_labels(xlabel='num of fixation', ylabel='reward depletion (%)', fontsize=12)

#     g.ax_joint.xaxis.set_major_locator(MultipleLocator(1))
#     g.ax_joint.set_xticks(range(1, 6))
#     g.ax_joint.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=11)
#     g.ax_joint.set_yticks([0, 0.5, 1.0])
#     g.ax_joint.set_yticklabels(['0', '50', '100'], fontsize=11)

#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     else:   
#         plt.show()

#     return mean



# def plot_joint(info):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from matplotlib.ticker import MultipleLocator

#     # Unpack dictionary
#     data = info['data']
#     category = info.get('category', 'unknown')
#     corr = info.get('corr')
#     pvalue = info.get('pvalue')
#     corr_baseline = info.get('corr_baseline', None)
#     pvalue_baseline = info.get('pvalue_baseline', None)
#     hvalue = info.get('hvalue', None)
#     ax = info.get('ax', None)
#     save_path = info.get('save_path', None)
#     overlay_baseline = info.get('overlay_baseline', False)
#     baseline_name = info.get('baseline_name', 'nobaseline')
#     use_legend = info.get('use_legend', True)

#     sns.set_style('white', rc={'xtick.bottom': True, 'ytick.left': True})

#     created_fig = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(3, 2.7))
#         created_fig = True

#     # Plot your model’s IOR (blue)
#     sns.regplot(
#         data=data,
#         x='backN',
#         y='IOR',
#         ax=ax,
#         color="#0015B2",
#         x_estimator=np.mean,
#         x_ci=68,
#         ci=68,
#         label='Reward'
#     )

#     # Optionally overlay baseline
#     if overlay_baseline and 'IOR_baseline' in data.columns:
#         sns.regplot(
#             data=data,
#             x='backN',
#             y='IOR_baseline',
#             ax=ax,
#             color="#696966",
#             x_estimator=np.mean,
#             x_ci=68,
#             ci=68,
#             label='Target'
#         )


#     # Text annotations
#     text_annotation = {
#         "corr_model": f"r_reward = {round(corr, 3)}, {pvalue}",
#         "corr_baseline": f"r_{baseline_name} = {round(corr_baseline, 3)}, {pvalue_baseline}" if corr_baseline is not None else "",
#         "label": f'{category}' if hvalue is None else f'{category} ({hvalue})'
#     }

#     ax.text(1.2, 0.11, text_annotation["corr_model"], fontsize=10)
#     if text_annotation["corr_baseline"]:
#         ax.text(1.2, 0.04, text_annotation["corr_baseline"],  fontsize=10)
#     ax.text(1.2, 0.18, text_annotation["label"], fontweight="bold", fontsize=11)

#     # Axis and styling
#     ax.set_xlim(1, 5)
#     ax.set_ylim(0, 0.5)
#     ax.set_xticks(range(1, 6))
#     ax.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=10)
#     ax.set_yticks([0, 0.5, 1.0])
#     ax.set_yticklabels(['0', '50', '100'], fontsize=10)
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Axis labels
#     if info.get('show_ylabel', True):
#         ax.set_ylabel("reward depletion (%)", fontsize=11)
#     else:
#         ax.set_ylabel("")
#         ax.set_yticklabels([])
#         # ax.tick_params(left=False)

#     if info.get('show_xlabel', True):
#         ax.set_xlabel("num of fixation", fontsize=11)
#     else:
#         ax.set_xlabel("")
#         ax.set_xticklabels([])
#         # ax.tick_params(bottom=False)
        
#     # Legend
#     if use_legend:
#         ax.legend(
#         fontsize=9,
#         frameon=True,
#         loc='center left',
#         bbox_to_anchor = (0, 0.4),
#         )
#     else:
#         if ax.get_legend() is not None:
#             ax.get_legend().remove()



#     if created_fig:
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         else:
#             plt.show()

#     # # Sanity check for average mean
#     # mean = data.groupby('backN')['IOR'].mean().mean()
#     # print(f"{category} average mean estimate: {mean:.3f}")
#     # return mean

