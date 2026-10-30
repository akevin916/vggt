"""SCARED clips in OpenD4RT's training-sample format (fine-tuning baseline B).

Mirrors OpenD4RT's own depth+pose datasets (src/data/co3d_raw_dataset.py, scannet): frames
squashed to 256x256, queries built by its ``build_queries_from_depth`` -- source-depth points
unprojected with the GT pose and reprojected into the target / reference cameras under a
static-scene assumption, with a depth-consistency check that drops points that moved. Query
timing follows its train_effective.yaml (4096 queries, 20% hard, P(t_tgt == t_cam) = 0.4,
static_local_global deltas) and temporal stride 1..2. No colour / crop augmentation, as in
that config.

SCARED stores depth as uint16 / 100 = mm and world-to-cam extrinsics in mm; both are
converted to metres (OpenD4RT's schema). The loss normalises by mean depth, so this only
keeps values in the range the log1p transform was tuned on.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.eval.d4rt_infer import _import_open_d4rt

_import_open_d4rt()                                   # puts OpenD4RT's `src` on sys.path
from src.data.depth_query_builder import build_queries_from_depth  # noqa: E402
from src.data.raw_augment import (STATIC_LOCAL_GLOBAL_TGT_DELTA_CHOICES,  # noqa: E402
                                  STATIC_LOCAL_GLOBAL_TGT_DELTA_PROBS)

DEPTH_SCALE = 100.0      # uint16 counts per mm


class ScaredD4RTDataset(Dataset):
    def __init__(self, root: str, split: str = "train", clip_frames: int = 48,
                 image_hw=(256, 256), length: int = 1500, stride_range=(1, 2),
                 queries_per_clip: int = 4096, hard_query_ratio: float = 0.2,
                 prob_t_tgt_equals_t_cam: float = 0.4, seed: int = 42):
        self.h, self.w = image_hw
        self.T, self.length, self.seed = clip_frames, length, seed
        self.stride_range = stride_range
        self.qcfg = dict(queries_per_clip=queries_per_clip, hard_query_ratio=hard_query_ratio,
                         prob_t_tgt_equals_t_cam=prob_t_tgt_equals_t_cam,
                         t_src_tgt_delta_choices=STATIC_LOCAL_GLOBAL_TGT_DELTA_CHOICES,
                         t_src_tgt_delta_probs=STATIC_LOCAL_GLOBAL_TGT_DELTA_PROBS)
        self.seqs = []
        split_dir = os.path.join(root, split)
        for ds in sorted(os.listdir(split_dir)):
            for kf in sorted(os.listdir(os.path.join(split_dir, ds))):
                d = os.path.join(split_dir, ds, kf)
                if not os.path.isdir(os.path.join(d, "image_left")):
                    continue
                fids = np.loadtxt(os.path.join(d, "cam_data", "frames.txt"), dtype=np.int64, ndmin=1)
                if len(fids) < clip_frames:
                    continue
                E = np.loadtxt(os.path.join(d, "cam_data", "extrinsics.txt")).reshape(-1, 3, 4)
                fx, fy, cx, cy = np.loadtxt(os.path.join(d, "cam_data", "intrinsics.txt"))
                self.seqs.append(dict(dir=d, fids=fids, E=E, K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])))
        if not self.seqs:
            raise ValueError(f"no SCARED keyframes with >= {clip_frames} frames under {split_dir}")
        # a keyframe is drawn in proportion to its length, so every frame is equally likely
        w = np.array([len(s["fids"]) for s in self.seqs], dtype=np.float64)
        self.seq_p = w / w.sum()

    def __len__(self):
        return self.length

    def _rng(self, index):
        info = torch.utils.data.get_worker_info()
        base = torch.initial_seed() if info is not None else self.seed
        return np.random.default_rng([int(base) % (2**32), int(index)])

    def __getitem__(self, index):
        rng = self._rng(index)
        s = self.seqs[int(rng.choice(len(self.seqs), p=self.seq_p))]
        n = len(s["fids"])
        max_stride = max(1, min(self.stride_range[1], (n - 1) // (self.T - 1)))
        stride = int(rng.integers(self.stride_range[0], max_stride + 1))
        start = int(rng.integers(0, n - (self.T - 1) * stride))
        idxs = start + stride * np.arange(self.T)

        video, depth, valid = [], [], []
        src_h = src_w = None
        for i in idxs:
            fid = int(s["fids"][i])
            img = cv2.cvtColor(cv2.imread(os.path.join(s["dir"], "image_left", f"{fid:06d}.png")),
                               cv2.COLOR_BGR2RGB)
            src_h, src_w = img.shape[:2]
            video.append(cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA))
            d = cv2.imread(os.path.join(s["dir"], "depth_left", f"{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
            d = cv2.resize(d.astype(np.float32) / DEPTH_SCALE / 1000.0, (self.w, self.h),
                           interpolation=cv2.INTER_NEAREST)                    # metres
            depth.append(d)
            valid.append(np.isfinite(d) & (d > 0))
        K = s["K"].copy()
        K[0] *= self.w / src_w
        K[1] *= self.h / src_h
        K[2] = [0, 0, 1]
        t_wc = []
        for i in idxs:
            w2c = np.vstack([s["E"][i], [0, 0, 0, 1.]])
            w2c[:3, 3] /= 1000.0                                                # mm -> m
            t_wc.append(np.linalg.inv(w2c))
        video = np.stack(video).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        depth, valid = np.stack(depth), np.stack(valid)
        k_arr = np.repeat(K[None].astype(np.float32), self.T, 0)
        t_wc = np.stack(t_wc).astype(np.float32)
        cam_valid = np.ones(self.T, dtype=np.bool_)

        query, target, mask, _ = build_queries_from_depth(
            rng=rng, depth=depth, depth_valid=valid, k_seq=k_arr, t_wc_seq=t_wc,
            camera_valid=cam_valid, **self.qcfg)
        return {
            "video": torch.from_numpy(video),
            "aspect_ratio": torch.tensor([src_w / src_h], dtype=torch.float32),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32)
                      for k, v in query.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
        }
