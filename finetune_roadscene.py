import subprocess
import sys
import os

for pkg, mod in [('mamba-ssm', 'mamba_ssm'), ('causal-conv1d', 'causal_conv1d'),
                  ('yacs', 'yacs'), ('kornia', 'kornia'), ('einops', 'einops'),
                  ('loguru', 'loguru')]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from kornia.utils import create_meshgrid
from einops.einops import rearrange
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.jamma.jamma import JamMa
from src.jamma.backbone import CovNextV2_nano
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config

ROADSCENE_DIR  = SCRIPT_DIR.parent / 'RoadScene'
VIS_DIR        = ROADSCENE_DIR / 'crop_HR_visible'
IR_DIR         = ROADSCENE_DIR / 'cropinfrared'
IMG_W, IMG_H   = 320, 256
BATCH_SIZE     = 2
NUM_EPOCHS     = 30
LR             = 5e-5
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR       = SCRIPT_DIR / 'finetune_output'
VIZ_SAMPLES    = 4
MAX_SHIFT      = 0.12
SUB_W          = 0.5
JAMMA_URL      = 'https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt'

SAVE_DIR.mkdir(exist_ok=True)
(SAVE_DIR / 'checkpoints').mkdir(exist_ok=True)
(SAVE_DIR / 'viz').mkdir(exist_ok=True)


class RoadSceneDataset(Dataset):
    def __init__(self, vis_dir, ir_dir, img_w=IMG_W, img_h=IMG_H,
                 max_shift=MAX_SHIFT, augment=True):
        self.img_w, self.img_h = img_w, img_h
        self.max_shift = max_shift
        self.augment = augment

        vis_dir, ir_dir = Path(vis_dir), Path(ir_dir)
        vs = {p.stem for p in vis_dir.glob('*.jpg')}
        irs = {p.stem for p in ir_dir.glob('*.jpg')}
        stems = sorted(vs & irs)
        self.pairs = [(vis_dir / f'{s}.jpg', ir_dir / f'{s}.jpg') for s in stems]
        logger.info(f'Dataset: {len(self.pairs)} visible-IR pairs')

    def _load(self, path, gray=False):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(path))
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        if gray:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _rand_H(self):
        H, W = self.img_h, self.img_w
        src = np.float32([[0,0],[W,0],[W,H],[0,H]])
        noise = np.random.uniform(-1, 1, (4, 2)).astype(np.float32)
        dst = src + noise * np.float32([self.max_shift * W, self.max_shift * H])
        return cv2.getPerspectiveTransform(src, dst)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vp, ip = self.pairs[idx]
        vis = self._load(vp, gray=False)
        ir  = self._load(ip, gray=True)

        H0 = self._rand_H() if self.augment else np.eye(3, dtype=np.float32)
        H1 = self._rand_H() if self.augment else np.eye(3, dtype=np.float32)

        kw = dict(dsize=(self.img_w, self.img_h), flags=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT)
        vis_w = cv2.warpPerspective(vis, H0, **kw)
        ir_w  = cv2.warpPerspective(ir,  H1, **kw)

        H_rel = H1 @ np.linalg.inv(H0)

        img0 = torch.from_numpy(vis_w).float().permute(2,0,1) / 255.0
        img1 = torch.from_numpy(ir_w).float().permute(2,0,1) / 255.0
        Ht   = torch.from_numpy(H_rel.astype(np.float32))

        return {
            'imagec_0':  img0,
            'imagec_1':  img1,
            'H_0to1':    Ht,
            'pair_names': (vp.name, ip.name),
        }


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == 'pair_names':
            out[k] = tuple(zip(*[b[k] for b in batch]))
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


@torch.no_grad()
def _warp_H(kpts, H, h_tgt, w_tgt):
    N, L, _ = kpts.shape
    ones = torch.ones(N, L, 1, device=kpts.device, dtype=kpts.dtype)
    kh   = torch.cat([kpts, ones], -1)
    wh   = (H @ kh.transpose(1,2)).transpose(1,2)
    w    = wh[..., :2] / (wh[..., [2]] + 1e-6)
    valid = (w[...,0] > 0) & (w[...,0] < w_tgt-1) & \
            (w[...,1] > 0) & (w[...,1] < h_tgt-1)
    return valid, w


@torch.no_grad()
def compute_supervision_coarse_h(data, config):
    dev = data['imagec_0'].device
    N, _, H0, W0 = data['imagec_0'].shape
    _, _, H1, W1 = data['imagec_1'].shape
    sc = config['JAMMA']['RESOLUTION'][0]
    h0, w0 = H0//sc, W0//sc
    h1, w1 = H1//sc, W1//sc

    Hm  = data['H_0to1'].to(dev)
    Him = torch.inverse(Hm)

    g0 = create_meshgrid(h0, w0, False, dev).reshape(1, h0*w0, 2).repeat(N,1,1)
    g1 = create_meshgrid(h1, w1, False, dev).reshape(1, h1*w1, 2).repeat(N,1,1)
    p0 = sc * g0
    p1 = sc * g1

    v0, wp0 = _warp_H(p0, Hm,  H1, W1)
    v1, wp1 = _warp_H(p1, Him, H0, W0)
    wp0[~v0] = 0
    wp1[~v1] = 0

    c0 = wp0 / sc
    c1 = wp1 / sc

    def oob(pt, w, h):
        return (pt[...,0]<0)|(pt[...,0]>=w)|(pt[...,1]<0)|(pt[...,1]>=h)

    r0 = c0.round().long()
    r1 = c1.round().long()
    ni1 = r0[...,0] + r0[...,1]*w1
    ni0 = r1[...,0] + r1[...,1]*w0
    ni1[oob(r0,w1,h1)|~v0] = 0
    ni0[oob(r1,w0,h0)|~v1] = 0

    a1 = torch.arange(h0*w0, device=dev)[None].repeat(N,1)
    a0 = torch.arange(h1*w1, device=dev)[None].repeat(N,1)
    a1[ni1==0] = 0
    a0[ni0==0] = 0
    ab = torch.arange(N, device=dev).unsqueeze(1)

    cm = torch.zeros(N, h0*w0, h1*w1, device=dev)
    cm[ab, a1, ni1] = 1
    cm[ab, ni0, a0] = 1
    cm[:, 0, 0] = False

    b_ids, i_ids, j_ids = cm.nonzero(as_tuple=True)
    if len(b_ids) == 0:
        b_ids = torch.tensor([0], device=dev)
        i_ids = torch.tensor([0], device=dev)
        j_ids = torch.tensor([0], device=dev)

    data.update({
        'conf_matrix_gt':   cm,
        'spv_b_ids':        b_ids,
        'spv_i_ids':        i_ids,
        'spv_j_ids':        j_ids,
        'num_candidates_max': int(b_ids.shape[0]),
        'spv_w_pt0_i':      wp0,
        'spv_pt1_i':        p1,
        'dataset_name':     ['roadscene'] * N,
    })


@torch.no_grad()
def compute_supervision_fine_h(data, config):
    W_f  = config['JAMMA']['FINE_WINDOW_SIZE']
    sc_c = config['JAMMA']['RESOLUTION'][0]
    sc_f = config['JAMMA']['RESOLUTION'][1]
    sfc  = sc_c // sc_f

    dev = data['imagec_0'].device
    N, _, H0, W0 = data['imagec_0'].shape
    _, _, H1, W1 = data['imagec_1'].shape
    h0f, w0f = H0//sc_f, W0//sc_f
    h1f, w1f = H1//sc_f, W1//sc_f

    b_ids = data['b_ids_fine']
    i_ids = data['i_ids_fine']
    j_ids = data['j_ids_fine']

    if len(b_ids) == 0:
        data.update({'conf_matrix_f_gt': torch.zeros(1,W_f**2,W_f**2,device=dev)})
        return

    Hm  = data['H_0to1'].to(dev)
    Him = torch.inverse(Hm)
    stride_f = data['hw0_f'][0] // data['hw0_c'][0]
    pad = 0 if W_f % 2 == 0 else W_f // 2

    def make_window_grid(hf, wf, n, sc):
        g = create_meshgrid(hf, wf, False, dev).repeat(n,1,1,1) * sc
        g = rearrange(g, 'n h w c -> n c h w')
        g = F.unfold(g, kernel_size=(W_f,W_f), stride=stride_f, padding=pad)
        return rearrange(g, 'n (c ww) l -> n l ww c', ww=W_f**2)

    g0 = make_window_grid(h0f, w0f, N, sc_f)[b_ids, i_ids]  # [M, W^2, 2]
    g1 = make_window_grid(h1f, w1f, N, sc_f)[b_ids, j_ids]

    M  = b_ids.shape[0]
    Hbm  = Hm[b_ids]
    Hibm = Him[b_ids]

    def batch_warp(pts, H):
        ph = torch.cat([pts, torch.ones(M,W_f**2,1,device=dev)],-1)
        wh = (H @ ph.transpose(1,2)).transpose(1,2)
        return wh[...,:2] / (wh[...,[2]] + 1e-6)

    wp0 = batch_warp(g0, Hbm)
    wp1 = batch_warp(g1, Hibm)

    c0f = torch.stack([i_ids % data['hw0_c'][1],
                       i_ids // data['hw0_c'][1]], dim=1).float() * sfc - pad
    c1f = torch.stack([j_ids % data['hw1_c'][1],
                       j_ids // data['hw1_c'][1]], dim=1).float() * sfc - pad

    w0f_rel = wp0 / sc_f - c1f[:,None,:]
    w1f_rel = wp1 / sc_f - c0f[:,None,:]

    def oob(pt, w, h):
        return (pt[...,0]<0)|(pt[...,0]>=w)|(pt[...,1]<0)|(pt[...,1]>=h)

    r0 = w0f_rel.round().long()
    r1 = w1f_rel.round().long()
    ni1 = r0[...,0] + r0[...,1]*W_f
    ni0 = r1[...,0] + r1[...,1]*W_f
    ni1[oob(r0,W_f,W_f)] = 0
    ni0[oob(r1,W_f,W_f)] = 0

    loop = torch.stack([ni0[b][ni1[b]] for b in range(M)], dim=0)
    ref  = torch.arange(W_f**2, device=dev)[None].repeat(M,1)
    ok   = (loop == ref)
    ok[:,0] = False

    cmf = torch.zeros(M, W_f**2, W_f**2, device=dev)
    bi, ii = torch.where(ok)
    ji = ni1[bi, ii]
    cmf[bi, ii, ji] = 1
    data.update({'conf_matrix_f_gt': cmf})


class CustomLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lc = config['jamma']['loss']

    def _focal(self, pred, gt, pos_w, neg_w=None):
        nw  = neg_w if neg_w is not None else self.lc['pos_weight']
        alpha, gamma = self.lc['focal_alpha'], self.lc['focal_gamma']
        pm, nm = gt > 0, gt == 0
        pw_use = pos_w
        if not pm.any():
            pm[0,0,0] = True; pw_use = 0.
        if not nm.any():
            nm[0,0,0] = True; nw = 0.
        pred = pred.clamp(1e-6, 1-1e-6)
        lp = -alpha*(1-pred[pm]).pow(gamma)*pred[pm].log()
        ln = -alpha*pred[nm].pow(gamma)*(1-pred[nm]).log()
        return pw_use*lp.mean() + nw*ln.mean()

    def forward(self, data):
        pw = self.lc['pos_weight']
        c01 = data['conf_matrix_0_to_1'].clamp(1e-6,1-1e-6)
        c10 = data['conf_matrix_1_to_0'].clamp(1e-6,1-1e-6)
        gt  = data['conf_matrix_gt']
        pm  = gt == 1
        pw_c = pw
        if not pm.any():
            pm[0,0,0] = True; pw_c = 0.
        alpha, gamma = self.lc['focal_alpha'], self.lc['focal_gamma']
        lp = -alpha*(1-c01[pm]).pow(gamma)*c01[pm].log()
        lp += -alpha*(1-c10[pm]).pow(gamma)*c10[pm].log()
        lc = pw_c * lp.mean() * self.lc['coarse_weight']

        lf = self._focal(data['conf_matrix_fine'], data['conf_matrix_f_gt'], pw) \
             * self.lc['fine_weight']

        loss = lc + lf
        scalars = {'loss_c': lc.detach().cpu(), 'loss_f': lf.detach().cpu()}

        m_bids = data.get('m_bids')
        pts0   = data.get('mkpts0_f_train')
        pts1   = data.get('mkpts1_f_train')
        if m_bids is not None and pts0 is not None and len(pts0) > 0:
            Hpm  = data['H_0to1'][m_bids]
            ph   = torch.cat([pts0, torch.ones(len(pts0),1,device=pts0.device)],-1)
            wh   = (Hpm @ ph.unsqueeze(-1)).squeeze(-1)
            wp0  = wh[:,:2] / (wh[:,[2]] + 1e-6)
            dist = (wp0 - pts1).norm(dim=-1)
            ok   = dist < 4.0
            ls   = (dist[ok].mean() if ok.any() else dist.mean() * 1e-9) * SUB_W
            loss = loss + ls
            scalars['loss_sub'] = ls.detach().cpu()

        scalars['loss'] = loss.detach().cpu()
        data.update({'loss': loss, 'loss_scalars': scalars})


def load_pretrained(backbone, matcher, device):
    logger.info('Downloading JamMa pretrained weights...')
    state = torch.hub.load_state_dict_from_url(JAMMA_URL, file_name='jamma.ckpt',
                                               map_location='cpu')['state_dict']
    bk = {k[9:]: v for k, v in state.items() if k.startswith('backbone.')}
    mt = {k[8:]: v for k, v in state.items() if k.startswith('matcher.')}
    backbone.load_state_dict(bk, strict=True)
    matcher.load_state_dict(mt, strict=True)
    logger.info('Pretrained weights loaded.')


def tensor_to_np(t):
    img = t.permute(1,2,0).cpu().numpy()
    return (img * 255).clip(0,255).astype(np.uint8)


def draw_matches(img0, img1, kp0, kp1, conf, max_kp=150):
    H, W = img0.shape[:2]
    gap = 4
    canvas = np.full((H, 2*W+gap, 3), 30, dtype=np.uint8)
    canvas[:, :W]      = img0
    canvas[:, W+gap:]  = img1
    if len(kp0) == 0:
        return canvas
    idx = np.argsort(-conf)[:max_kp]
    cmap = plt.cm.plasma
    for i in idx:
        c_val = float(conf[i])
        c_rgb = tuple(int(x*255) for x in cmap(c_val)[:3])
        x0, y0 = int(kp0[i,0]), int(kp0[i,1])
        x1, y1 = int(kp1[i,0])+W+gap, int(kp1[i,1])
        cv2.line(canvas,(x0,y0),(x1,y1),c_rgb,1,cv2.LINE_AA)
        cv2.circle(canvas,(x0,y0),3,c_rgb,-1)
        cv2.circle(canvas,(x1,y1),3,c_rgb,-1)
    return canvas


@torch.no_grad()
def run_inference(backbone, matcher, batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    backbone(batch)
    matcher(batch, mode='test')
    kp0 = batch.get('mkpts0_f', batch.get('mkpts0_c', torch.zeros(0,2))).cpu().numpy()
    kp1 = batch.get('mkpts1_f', batch.get('mkpts1_c', torch.zeros(0,2))).cpu().numpy()
    cf  = batch.get('mconf_f', batch.get('mconf', torch.zeros(len(kp0)))).cpu().numpy()
    return kp0, kp1, cf


def compute_transfer_errors(kp0, kp1, H_np):
    if len(kp0) == 0:
        return np.array([])
    kp0h = np.concatenate([kp0, np.ones((len(kp0),1))], axis=1)
    wkp0 = (H_np @ kp0h.T).T
    wkp0 = wkp0[:,:2] / (wkp0[:,[2]] + 1e-6)
    return np.linalg.norm(wkp0 - kp1, axis=1)


def visualize_epoch(backbone, matcher, dataset, epoch, config, device, loss_history):
    backbone.eval(); matcher.eval()
    n = min(VIZ_SAMPLES, len(dataset))
    idxs = np.linspace(0, len(dataset)-1, n, dtype=int)

    fig = plt.figure(figsize=(22, 6*n))
    gs_outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.45)

    all_errors = []
    all_n_matches = []

    for row, idx in enumerate(idxs):
        sample = dataset[idx]
        batch = collate_fn([sample])
        for k,v in batch.items():
            if isinstance(v, torch.Tensor): batch[k] = v.to(device)

        kp0, kp1, cf = run_inference(backbone, matcher, batch, device)
        H_np = batch['H_0to1'][0].cpu().numpy()
        errs = compute_transfer_errors(kp0, kp1, H_np)
        all_errors.extend(errs.tolist())
        all_n_matches.append(len(kp0))

        img0 = tensor_to_np(batch['imagec_0'][0].cpu())
        img1 = tensor_to_np(batch['imagec_1'][0].cpu())
        canvas = draw_matches(img0, img1, kp0, kp1, cf)

        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[row],
                                                    wspace=0.08)

        ax0 = fig.add_subplot(gs_inner[0, :2])
        ax0.imshow(canvas[:,:,::-1])
        n_inliers = int((errs < 2.0).sum()) if len(errs) > 0 else 0
        ax0.set_title(f'Pair {idx} | {len(kp0)} matches | {n_inliers} inliers (<2px) | '
                      f'median err {np.median(errs):.2f}px' if len(errs)>0 else
                      f'Pair {idx} | 0 matches', fontsize=10)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs_inner[0, 2])
        if len(errs) > 0:
            ax1.hist(errs.clip(0, 30), bins=40, color='steelblue', edgecolor='k', linewidth=0.4)
            ax1.axvline(2.0, color='red', linestyle='--', linewidth=1.2, label='2px')
            ax1.axvline(np.median(errs), color='orange', linestyle='--', linewidth=1.2,
                        label=f'median={np.median(errs):.1f}px')
            ax1.legend(fontsize=8)
        ax1.set_xlabel('Transfer error (px)', fontsize=9)
        ax1.set_ylabel('Count', fontsize=9)
        ax1.set_title('Error distribution', fontsize=10)

    viz_dir = SAVE_DIR / 'viz' / f'epoch_{epoch:03d}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f'Epoch {epoch} — JamMa fine-tuned on RoadScene (visible↔IR)', fontsize=13)
    fig.savefig(viz_dir / 'matches.png', bbox_inches='tight', dpi=110)
    plt.close(fig)

    if loss_history:
        plot_loss_curves(loss_history, viz_dir / 'loss.png')

    backbone.train(); matcher.train()
    return all_errors, all_n_matches


def plot_loss_curves(history, path):
    keys = ['loss', 'loss_c', 'loss_f', 'loss_sub']
    colors = ['black', 'steelblue', 'coral', 'mediumseagreen']
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    for k, c in zip(keys, colors):
        if k in history and len(history[k]) > 0:
            vals = history[k]
            ax.plot(vals, label=k, color=c, linewidth=1.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Losses'); ax.legend(); ax.grid(alpha=0.3)

    ax2 = axes[1]
    if 'n_matches' in history and len(history['n_matches']) > 0:
        ax2.plot(history['n_matches'], color='purple', linewidth=1.8, label='avg matches/pair')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('# matches')
        ax2.set_title('Matches per pair'); ax2.legend(); ax2.grid(alpha=0.3)
    if 'median_err' in history and len(history['median_err']) > 0:
        ax3 = ax2.twinx()
        ax3.plot(history['median_err'], color='darkorange', linewidth=1.8,
                 linestyle='--', label='median err (px)')
        ax3.set_ylabel('Median transfer error (px)', color='darkorange')
        ax3.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def visualize_before_after(backbone, matcher, dataset, before_state, config, device):
    n = min(4, len(dataset))
    idxs = np.linspace(0, len(dataset)-1, n, dtype=int)

    fig, axes = plt.subplots(n, 2, figsize=(20, 5*n))
    fig.suptitle('Before (pretrained) vs After (fine-tuned) on visible↔IR', fontsize=13)

    for row, idx in enumerate(idxs):
        sample = dataset[idx]
        batch_after = collate_fn([sample])
        for k,v in batch_after.items():
            if isinstance(v, torch.Tensor): batch_after[k] = v.to(device)

        batch_before = {k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in batch_after.items()}
        batch_before['imagec_0'] = batch_after['imagec_0'].clone()
        batch_before['imagec_1'] = batch_after['imagec_1'].clone()

        backbone.load_state_dict(before_state['backbone'])
        matcher.load_state_dict(before_state['matcher'])
        backbone.eval(); matcher.eval()
        kp0_b, kp1_b, cf_b = run_inference(backbone, matcher, batch_before, device)

        backbone.load_state_dict(before_state['_after_backbone'])
        matcher.load_state_dict(before_state['_after_matcher'])
        backbone.eval(); matcher.eval()
        kp0_a, kp1_a, cf_a = run_inference(backbone, matcher, batch_after, device)

        H_np = batch_after['H_0to1'][0].cpu().numpy()
        img0 = tensor_to_np(batch_after['imagec_0'][0].cpu())
        img1 = tensor_to_np(batch_after['imagec_1'][0].cpu())

        err_b = compute_transfer_errors(kp0_b, kp1_b, H_np)
        err_a = compute_transfer_errors(kp0_a, kp1_a, H_np)

        cb = draw_matches(img0, img1, kp0_b, kp1_b, cf_b)
        ca = draw_matches(img0, img1, kp0_a, kp1_a, cf_a)

        for col, (canvas, errs, tag) in enumerate([
            (cb, err_b, f'Pretrained | {len(kp0_b)} matches | med={np.median(errs):.1f}px' if len(errs)>0 else f'Pretrained | 0 matches'),
            (ca, err_a, f'Fine-tuned  | {len(kp0_a)} matches | med={np.median(errs):.1f}px' if len(errs)>0 else f'Fine-tuned  | 0 matches'),
        ]):
            ax = axes[row, col] if n > 1 else axes[col]
            ax.imshow(canvas[:,:,::-1])
            ax.set_title(tag, fontsize=10)
            ax.axis('off')

    fig.tight_layout()
    fig.savefig(SAVE_DIR / 'viz' / 'before_after.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def train_step(batch, backbone, matcher, loss_fn, config, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    compute_supervision_coarse_h(batch, config)
    backbone(batch)
    matcher(batch, mode='train')
    compute_supervision_fine_h(batch, config)
    loss_fn(batch)
    return batch['loss'], batch['loss_scalars']


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = get_cfg_defaults()
    config = lower_config(cfg)

    logger.info(f'Device: {DEVICE}')
    logger.info('Building model...')

    backbone = CovNextV2_nano().to(DEVICE)
    matcher  = JamMa(config=config['jamma']).to(DEVICE)

    load_pretrained(backbone, matcher, DEVICE)

    before_bk_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
    before_mt_state = {k: v.cpu().clone() for k, v in matcher.state_dict().items()}

    loss_fn = CustomLoss(config).to(DEVICE)

    dataset   = RoadSceneDataset(VIS_DIR, IR_DIR, augment=True)
    val_dset  = RoadSceneDataset(VIS_DIR, IR_DIR, augment=False)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, collate_fn=collate_fn, drop_last=True)

    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in matcher.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [{'params': matcher.parameters(), 'lr': LR},
         {'params': backbone.parameters(), 'lr': LR * 0.1}],
        weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR*0.05)

    history = {k: [] for k in ['loss','loss_c','loss_f','loss_sub','n_matches','median_err']}

    logger.info('Starting fine-tuning...')

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == 6:
            for p in backbone.parameters():
                p.requires_grad_(True)
            logger.info('Backbone unfrozen at epoch 6.')

        backbone.train(); matcher.train()
        ep_scalars = {k: [] for k in ['loss','loss_c','loss_f','loss_sub']}
        t0 = time.time()

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            try:
                loss, scalars = train_step(batch, backbone, matcher, loss_fn, config, DEVICE)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(matcher.parameters()), 0.5)
                optimizer.step()
                for k in ep_scalars:
                    if k in scalars:
                        ep_scalars[k].append(float(scalars[k]))
            except Exception as e:
                logger.warning(f'Step {step} failed: {e}')
                continue

        scheduler.step()

        for k in ['loss','loss_c','loss_f','loss_sub']:
            v = ep_scalars[k]
            history[k].append(float(np.mean(v)) if v else 0.0)

        errs, nm = visualize_epoch(backbone, matcher, val_dset, epoch, config, DEVICE, history)
        history['n_matches'].append(float(np.mean(nm)) if nm else 0.0)
        history['median_err'].append(float(np.median(errs)) if errs else 0.0)

        elapsed = time.time() - t0
        logger.info(
            f'Epoch {epoch:3d}/{NUM_EPOCHS} | '
            f'loss={history["loss"][-1]:.4f} | '
            f'loss_c={history["loss_c"][-1]:.4f} | '
            f'loss_f={history["loss_f"][-1]:.4f} | '
            f'matches={history["n_matches"][-1]:.0f} | '
            f'med_err={history["median_err"][-1]:.2f}px | '
            f'{elapsed:.0f}s'
        )

        ckpt = {
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'matcher':  matcher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history':  history,
        }
        torch.save(ckpt, SAVE_DIR / 'checkpoints' / f'epoch_{epoch:03d}.pt')
        torch.save(ckpt, SAVE_DIR / 'checkpoints' / 'latest.pt')

    plot_loss_curves(history, SAVE_DIR / 'viz' / 'final_loss_curves.png')

    before_state = {
        'backbone': before_bk_state,
        'matcher':  before_mt_state,
        '_after_backbone': {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
        '_after_matcher':  {k: v.cpu().clone() for k, v in matcher.state_dict().items()},
    }
    visualize_before_after(backbone, matcher, val_dset, before_state, config, DEVICE)

    visualize_final_summary(history)
    logger.info(f'Done. Outputs saved to {SAVE_DIR}')


def visualize_final_summary(history):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['loss'])+1)
    ax.plot(epochs, history['loss'],   'k-',  lw=2, label='total')
    ax.plot(epochs, history['loss_c'], 'b-',  lw=1.5, label='coarse')
    ax.plot(epochs, history['loss_f'], 'r-',  lw=1.5, label='fine')
    if any(v > 0 for v in history['loss_sub']):
        ax.plot(epochs, history['loss_sub'], 'g--', lw=1.5, label='sub-px')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Loss components')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['n_matches'], 'purple', lw=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Avg matches / pair')
    ax2.set_title('Matches per pair'); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['median_err'], 'darkorange', lw=2)
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Median transfer error (px)')
    ax3.set_title('Match accuracy (lower = better)'); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, :])
    if history['median_err']:
        best_epoch = int(np.argmin(history['median_err'])) + 1
        ax4.bar(list(epochs), history['median_err'], color='steelblue', alpha=0.7)
        ax4.axvline(best_epoch, color='red', linestyle='--', lw=2,
                    label=f'Best epoch: {best_epoch}')
        ax4.set_xlabel('Epoch'); ax4.set_ylabel('Median transfer error (px)')
        ax4.set_title('Transfer error per epoch'); ax4.legend(); ax4.grid(alpha=0.3, axis='y')

    fig.suptitle('JamMa Fine-tuning on RoadScene (visible↔IR) — Summary', fontsize=14)
    fig.savefig(SAVE_DIR / 'viz' / 'summary.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Summary saved to {SAVE_DIR}/viz/summary.png')


if __name__ == '__main__':
    main()
