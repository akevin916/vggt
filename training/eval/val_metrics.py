"""Online depth / pose metrics for trainer validation loops."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import torch

from eval.depth_metrics import depth_evaluation
from eval.pose_metrics import eval_pose_metrics, extrinsics_w2c_to_tum
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

_DEPTH_KEYS = ("abs_rel", "delta_1", "rmse")
_POSE_KEYS = ("ate", "rpe_trans", "rpe_rot")


class ValMetricsAccumulator:
    """Accumulate median-scaled depth and ATE/RPE pose metrics over val batches."""

    def __init__(self, max_depth: float = 80.0, min_depth_pixels: int = 100):
        self.max_depth = max_depth
        self.min_depth_pixels = min_depth_pixels
        self.reset()

    def reset(self) -> None:
        self._depth = {k: 0.0 for k in _DEPTH_KEYS}
        self._depth_frames = 0
        self._pose = {k: 0.0 for k in _POSE_KEYS}
        self._pose_seqs = 0

    def update(self, predictions: Mapping, batch: Mapping) -> Dict[str, float]:
        """Update accumulators from one val batch; returns batch-local averages."""
        batch_metrics: Dict[str, float] = {}

        if "depth" in predictions and "depths" in batch:
            batch_metrics.update(self._update_depth(predictions, batch))

        if "pose_enc" in predictions and "extrinsics" in batch:
            batch_metrics.update(self._update_pose(predictions, batch))

        return batch_metrics

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._depth_frames > 0:
            for k in _DEPTH_KEYS:
                out[k] = self._depth[k] / self._depth_frames
            out["depth_frames"] = float(self._depth_frames)
        if self._pose_seqs > 0:
            for k in _POSE_KEYS:
                out[k] = self._pose[k] / self._pose_seqs
            out["pose_seqs"] = float(self._pose_seqs)
        return out

    def _update_depth(self, predictions: Mapping, batch: Mapping) -> Dict[str, float]:
        pred_depth = predictions["depth"]
        if pred_depth.ndim == 5:
            pred_depth = pred_depth[..., 0]

        gt_depth = batch["depths"]
        masks = batch["point_masks"]

        batch_depth = {k: 0.0 for k in _DEPTH_KEYS}
        n_frames = 0

        bsz, seq_len = gt_depth.shape[:2]
        for b in range(bsz):
            for s in range(seq_len):
                m = depth_evaluation(
                    pred_depth[b, s].detach().float().cpu().numpy(),
                    gt_depth[b, s].detach().float().cpu().numpy(),
                    max_depth=self.max_depth,
                    custom_mask=masks[b, s].detach().cpu().numpy(),
                )
                if m["valid_pixels"] < self.min_depth_pixels:
                    continue
                for k in _DEPTH_KEYS:
                    batch_depth[k] += m[k]
                    self._depth[k] += m[k]
                n_frames += 1
                self._depth_frames += 1

        if n_frames == 0:
            return {}
        return {f"metric_{k}": batch_depth[k] / n_frames for k in _DEPTH_KEYS}

    def _update_pose(self, predictions: Mapping, batch: Mapping) -> Dict[str, float]:
        images = batch["images"]
        h, w = images.shape[-2], images.shape[-1]
        pose_enc = predictions["pose_enc"]

        pred_ext, _ = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(h, w)
        )
        pred_ext = pred_ext.detach().float().cpu().numpy()
        gt_ext = batch["extrinsics"].detach().float().cpu().numpy()

        batch_pose = {k: 0.0 for k in _POSE_KEYS}
        n_seqs = 0

        for b in range(pred_ext.shape[0]):
            if pred_ext.shape[1] < 2:
                continue
            gt_tum, gt_ts = extrinsics_w2c_to_tum(gt_ext[b])
            try:
                pm = eval_pose_metrics(pred_ext[b], gt_tum, gt_ts)
            except Exception:
                continue
            for k in _POSE_KEYS:
                batch_pose[k] += pm[k]
                self._pose[k] += pm[k]
            n_seqs += 1
            self._pose_seqs += 1

        if n_seqs == 0:
            return {}
        return {f"metric_{k}": batch_pose[k] / n_seqs for k in _POSE_KEYS}
