"""Helpers shared by the walkthrough / analysis notebooks.

Kept out of the notebooks so the cells only show the training logic.
"""
import os
import sys
import importlib
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = os.environ.get('SMN_REPO', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAT_NAMES = ['ped_crossing', 'divider', 'boundary']
CAT_COLORS = ['tab:blue', 'tab:orange', 'tab:green']
CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def setup_env():
    """cd to the repo, make `plugin` importable and register its modules."""
    os.chdir(REPO)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.environ.setdefault('TORCH_HOME', os.path.join(REPO, '.cache/torch'))
    importlib.import_module('plugin')


def to_device(batch, device=0):
    """DataContainer batch from the dataloader -> plain tensors / lists on GPU."""
    from mmcv.parallel import scatter
    return scatter(batch, [device])[0]


# ---------------------------------------------------------------- visualization
def denorm_img(img, img_norm_cfg):
    """(3, H, W) normalized BGR tensor -> (H, W, 3) uint8 RGB."""
    mean = np.array(img_norm_cfg['mean'])[:, None, None]
    std = np.array(img_norm_cfg['std'])[:, None, None]
    x = img.detach().cpu().numpy() * std + mean
    x = np.clip(x, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return x[..., ::-1] if not img_norm_cfg.get('to_rgb', False) else x


def show_cams(imgs, img_norm_cfg, title=None, cam_names=None):
    """imgs: (6, 3, H, W)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(denorm_img(imgs[i], img_norm_cfg))
        ax.set_title(cam_names[i] if cam_names else f'cam {i}', fontsize=9)
        ax.axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _bev_axes(ax, roi_size, title=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(roi_size[1] / roi_size[0])
    ax.plot(0.5, 0.5, marker='>', color='red', ms=10)  # ego, x-forward
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f'{-roi_size[0]/2:.0f}m', '0', f'{roi_size[0]/2:.0f}m'])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([f'{-roi_size[1]/2:.0f}m', '0', f'{roi_size[1]/2:.0f}m'])
    if title:
        ax.set_title(title, fontsize=10)


def plot_gt_vectors(ax, vectors, roi_size, title='GT'):
    """vectors: {label: [array (num_permute, num_pts, 2) or (num_pts, 2)]}, normalized to [0, 1]."""
    for label, lines in vectors.items():
        for line in lines:
            line = np.asarray(line)
            if line.ndim == 3:
                line = line[0]
            ax.plot(line[:, 0], line[:, 1], color=CAT_COLORS[label], lw=2)
            ax.scatter(line[:, 0], line[:, 1], color=CAT_COLORS[label], s=4)
    _bev_axes(ax, roi_size, title)


def plot_pred_lines(ax, lines, scores, roi_size, thr=0.3, title='pred'):
    """lines: (Q, num_pts, 2) normalized; scores: (Q, num_cls) logits."""
    lines = lines.detach().cpu().numpy()
    probs = scores.detach().sigmoid().cpu().numpy()
    labels, conf = probs.argmax(-1), probs.max(-1)
    for q in np.where(conf > thr)[0]:
        ax.plot(lines[q, :, 0], lines[q, :, 1], color=CAT_COLORS[labels[q]],
                lw=1 + 2 * conf[q], alpha=float(min(1.0, 0.3 + conf[q])))
    _bev_axes(ax, roi_size, f'{title} (>{thr})')


def feat_to_rgb(feat):
    """(C, H, W) feature -> (H, W, 3) via PCA, for eyeballing feature maps."""
    C, H, W = feat.shape
    x = feat.detach().float().cpu().reshape(C, -1).T  # CPU: GPU cusolver fails when memory is nearly full
    x = x - x.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(x, q=3, center=False)
    y = x @ V[:, :3]
    y = (y - y.min(0)[0]) / (y.max(0)[0] - y.min(0)[0] + 1e-6)
    return y.reshape(H, W, 3).cpu().numpy()


# ---------------------------------------------------------------- recording
class FeatureRecorder:
    """Records module outputs with forward hooks.

    rec = FeatureRecorder(model, {'img_backbone': model.backbone.img_backbone, ...})
    model(...); rec['img_backbone']  -> list of outputs (one per call)
    """

    def __init__(self, modules):
        self.records = OrderedDict((k, []) for k in modules)
        self.handles = [m.register_forward_hook(self._hook(k)) for k, m in modules.items()]

    def _hook(self, name):
        def fn(module, inputs, output):
            self.records[name].append(_detach(output))
        return fn

    def clear(self):
        for v in self.records.values():
            v.clear()

    def remove(self):
        for h in self.handles:
            h.remove()

    def __getitem__(self, k):
        return self.records[k]


def _detach(x):
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, (list, tuple)):
        return type(x)(_detach(v) for v in x)
    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    return x


def describe(x, name=''):
    """Print shapes of nested tensors."""
    if torch.is_tensor(x):
        print(f'{name:<40s} {tuple(x.shape)}  dtype={x.dtype}  '
              f'mean={x.float().mean():.3f} std={x.float().std():.3f}')
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            describe(v, f'{name}[{i}]')
    elif isinstance(x, dict):
        for k, v in x.items():
            describe(v, f'{name}.{k}')
    else:
        print(f'{name:<40s} {type(x).__name__}')


def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6
