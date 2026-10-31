"""MonST3R on SCARED, scored with the SAME code as pipeline/benchmark/eval_scared.py.

eval_scared.eval_ckpt is model-agnostic once two hooks are swapped: ``infer_sequence`` (images ->
per-frame world-to-cam extrinsics + depth) and ``load_vggt_for_eval``. This script replaces them
with MonST3R's pairwise inference + global alignment and then calls eval_scared.main(), so chunking
(independent overlapping chunks), snippet ATE, the afsfm per-frame-median depth protocol and the
results.json layout are identical to the Dyn-VGGT numbers. All eval_scared CLI flags apply;
--out_dir is required (MonST3R ckpts do not live under logs/<exp>/ckpts/).

Depth: MonST3R works on a 512-wide, multiple-of-16-high centre crop (1280x1024 -> 512x400). It is
resized to the original width and edge-padded back to the original height (12 px top/bottom), so
the eval sees a full-resolution map and eval_scared's VGGT-specific resize step is bypassed.
--single_view runs MonST3R on the pair (img, img) and keeps view 0 (its monocular mode).
Pose is the optimised camera trajectory of each chunk (c2w -> w2c).

Usage (from repo root):
    python pipeline/benchmark/eval_monst3r_scared.py --ckpts logs/monst3r_scared_ft/checkpoint-10.pth \
        --split test/pose --n_frames 0 --chunk_size 64 --overlap 16 --no_depth --out_dir <dir>
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.benchmark import eval_monst3r_lesion as L   # puts reference/monst3r on sys.path
from pipeline.benchmark import eval_scared as ES

NITER, LR, SCHEDULE = 300, 0.01, "linear"   # MonST3R's demo defaults


def _to_full_res(depth, true_hw, orig_wh):
    """(H',W') crop-space depth -> (orig_h, orig_w), edge-padded where the centre crop cut rows."""
    h, w = int(true_hw[0]), int(true_hw[1])
    depth = np.asarray(depth, dtype=np.float32).reshape(h, w)
    ow, oh = orig_wh
    th = int(round(h * ow / w))
    d = cv2.resize(depth, (ow, th), interpolation=cv2.INTER_LINEAR)
    top = max((oh - th) // 2, 0)
    bot = max(oh - th - top, 0)
    if top or bot:
        d = np.pad(d, ((top, bot), (0, 0)), mode="edge")
    return d[:oh]


def monst3r_infer(model, image_paths, device="cuda", **_):
    orig_wh = Image.open(image_paths[0]).size
    if len(image_paths) == 1:
        scene, imgs = L.run_monst3r_pair(model, image_paths[0], image_paths[0], device)
        depths = [scene.get_depthmaps()[0].detach().cpu().numpy()]
        poses = None
    else:
        scene, imgs = L.run_monst3r_sequence(model, image_paths, device, NITER, SCHEDULE, LR)
        depths = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
        poses = scene.get_im_poses().detach().cpu()
    out = {"depth": np.stack([_to_full_res(d, imgs[i]["true_shape"].reshape(-1), orig_wh)
                              for i, d in enumerate(depths)])}
    if poses is not None:
        out["extrinsic"] = np.stack([L.c2w_to_extrinsic(p).numpy() for p in poses])
    else:   # single view: pose is skipped upstream, keep the key present
        out["extrinsic"] = np.tile(np.eye(4)[:3][None], (1, 1, 1)).astype(np.float32)
    return out


def main():
    # MonST3R ckpts pickle an argparse.Namespace; torch>=2.6 refuses it by default.
    torch.serialization.add_safe_globals([argparse.Namespace])
    ES.load_vggt_for_eval = lambda ckpt, device="cuda": L.load_model(ckpt, device)
    ES.infer_sequence = monst3r_infer
    ES.resize_pred_to_gt = lambda pred, meta: pred
    ES.main()


if __name__ == "__main__":
    main()
