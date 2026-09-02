# Dyn-VGGT MPI-Sintel dataset loader (validation / cross-domain eval).
#
# Sintel is the canonical cross-domain dynamic benchmark for this project: a synthetic film
# with large camera motion, animated characters, and dense GT depth + per-frame GT camera. It
# is NOT in the training mix — it is used as the VAL set so the trainer's ATE/depth metrics and
# best.pt selection reflect cross-domain pose robustness (PO val was too in-domain to read).
#
# Disk layout (MonST3R standard, <SINTEL_DIR> = .../sintel/training):
#   final/<seq>/frame_XXXX.png        RGB frames (1024 × 436, uint8)
#   depth/<seq>/frame_XXXX.dpt        dense metric depth (float32, .dpt)
#   camdata_left/<seq>/frame_XXXX.cam intrinsic 3×3 + extrinsic 3×4 (w2c, OpenCV convention)
#   flow/<seq>/frame_XXXX.flo         GT optical flow frame t → t+1 (last frame has none)
#
# Dynamic GT (`motion_mask`), dynamic_source:
#   "none"    — motion_mask absent (ComposedDataset zero-fills). Use when only ATE/depth matter;
#               gate BCE would then be measured against an all-static label (not meaningful).
#   "gt_flow" — per-frame flow-residual mask m*(t) = 1[‖f^gt(t→t+1) − f^ego(t→t+1)‖ > motion_thr]
#               (§3.4 (m_geo)) built from Sintel GT flow (not RAFT) + ego-flow from GT depth+pose. This
#               is the cleanest possible label — the same signal the oracle gate ablation uses.
#               The last frame of each sequence (no GT flow) and invalid-depth pixels are static.
#
# Coordinate convention: Sintel .cam extrinsics are already world-to-cam 3×4 (OpenCV), exactly
# the `extri_opencv` VGGT expects — no inversion needed (unlike TartanAir/Spring/Waymo c2w).

import os
import os.path as osp
import logging
import random

import numpy as np

from pipeline.data.dataset_util import read_image_cv2, threshold_depth_map
from pipeline.data.base_dataset import BaseDataset
from pipeline.data.paths import data_path

from pipeline.data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    read_sintel_depth,
    sintel_cam_read,
    sintel_seq_paths,
    matching_cam_path,
    matching_depth_path,
    load_sintel_rgb_paths,
)
from pipeline.data.motion_mask import read_flo, compute_ego_flow, derive_motion_mask


class SintelDataset(BaseDataset):

    _DYNAMIC_SOURCES = ("none", "gt_flow")

    def __init__(
        self,
        common_conf,
        SINTEL_DIR: str = data_path("eval", "sintel"),
        sequences: list = None,     # default: MonST3R's 14-seq eval split
        min_num_images: int = 16,
        len_train: int = 10000,
        depth_max: float = 100.0,
        dynamic_source: str = "gt_flow",
        motion_thr: float = 2.0,    # flow-residual threshold in native px (matches eval/precompute)
    ):
        super().__init__(common_conf=common_conf)

        if dynamic_source not in self._DYNAMIC_SOURCES:
            raise ValueError(f"dynamic_source must be one of {self._DYNAMIC_SOURCES}, got {dynamic_source!r}")
        self.dynamic_source = dynamic_source
        self.motion_thr = motion_thr

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.SINTEL_DIR = SINTEL_DIR
        self.depth_max = depth_max
        self.len_train = len_train

        if not osp.isdir(osp.join(SINTEL_DIR, "final")):
            raise FileNotFoundError(f"Sintel final/ not found under {SINTEL_DIR}")

        seqs = list(sequences) if sequences is not None else list(SINTEL_EVAL_SEQUENCES)
        if self.debug:
            seqs = seqs[:2]

        # Each Sintel sequence is one video. Cache the sorted RGB frame paths per seq.
        self.data_store = {}   # seq_name -> {rgb_paths, depth_dir, cam_dir, num_frames}
        for seq in seqs:
            rgb_dir, depth_dir, cam_dir = sintel_seq_paths(SINTEL_DIR, seq)
            if not osp.isdir(rgb_dir):
                continue
            rgb_paths = sorted(
                p for p in load_sintel_rgb_paths(SINTEL_DIR, seq)
            )
            if len(rgb_paths) < min_num_images:
                continue
            self.data_store[seq] = {
                "rgb_paths": rgb_paths,
                "depth_dir": depth_dir,
                "cam_dir": cam_dir,
                "num_frames": len(rgb_paths),
            }

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: Sintel loaded {self.sequence_list_len} sequences "
            f"(dynamic_source={self.dynamic_source})"
        )

    @staticmethod
    def _load_cam(cam_dir: str, rgb_path: str):
        """Return (K 3x3, w2c 3x4) for the frame matching rgb_path."""
        K, ext_w2c = sintel_cam_read(matching_cam_path(cam_dir, rgb_path))
        return K.astype(np.float32), ext_w2c.astype(np.float32)

    def _gt_flow_mask(self, store, fid, depth_raw, K, w2c_cur, hw):
        # Per-frame GT-flow residual dynamic mask (§3.4 (m_geo)). Native res. All-static for the last
        # frame (no GT flow) or when dynamic_source != gt_flow. Invalid-depth pixels are static.
        if self.dynamic_source != "gt_flow":
            return np.zeros(hw, dtype=np.float32)
        rgb_paths = store["rgb_paths"]
        if fid + 1 >= len(rgb_paths):
            return np.zeros(hw, dtype=np.float32)
        stem = osp.splitext(osp.basename(rgb_paths[fid]))[0]
        flo_path = osp.join(self.SINTEL_DIR, "flow", osp.basename(osp.dirname(rgb_paths[fid])), f"{stem}.flo")
        if not osp.isfile(flo_path):
            return np.zeros(hw, dtype=np.float32)
        gt_flow = read_flo(flo_path)                                   # (H, W, 2)
        _, w2c_next = self._load_cam(store["cam_dir"], rgb_paths[fid + 1])
        ego = compute_ego_flow(depth_raw.astype(np.float64), K.astype(np.float64), w2c_cur, w2c_next)
        mask = derive_motion_mask(gt_flow, ego, threshold=self.motion_thr)
        mask[depth_raw <= 0] = 0.0        # can't judge invalid depth -> static
        mask[depth_raw >= self.depth_max] = 0.0
        return mask.astype(np.float32)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        store = self.data_store[seq_name]
        rgb_paths = store["rgb_paths"]
        num_frames = store["num_frames"]

        if ids is None:
            if self.get_nearby:
                anchor = np.random.randint(0, num_frames)
                seed_ids = [anchor] * img_per_seq
                ids = self.get_nearby_ids(seed_ids, num_frames, expand_ratio=2.0)
            else:
                ids = np.random.choice(num_frames, img_per_seq, replace=self.allow_duplicate_img)
            ids = np.sort(np.asarray(ids))

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        motion_masks = []
        image_paths, original_sizes = [], []

        for fid in ids:
            fid = int(fid)
            image_path = rgb_paths[fid]
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])

            intri_opencv, extri_opencv = self._load_cam(store["cam_dir"], image_path)

            depth_raw = read_sintel_depth(matching_depth_path(store["depth_dir"], image_path))
            if self.load_depth:
                depth_map = threshold_depth_map(depth_raw, max_depth=self.depth_max)
            else:
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            # Native-res GT-flow dynamic mask (uses RAW depth for ego-flow before thresholding).
            motion_gt = self._gt_flow_mask(
                store, fid, depth_raw, intri_opencv, extri_opencv, tuple(original_size)
            )

            # Capture RNG so the mask replays the SAME geometric transform (mirrors PO/Spring).
            pre_state = np.random.get_state()
            (image_t, depth_t, extri_t, intri_t,
             world_pts, cam_pts, point_mask, _) = self.process_one_image(
                image, depth_map, extri_opencv, intri_opencv,
                original_size, target_image_shape, filepath=image_path,
            )
            post_state = np.random.get_state()
            np.random.set_state(pre_state)
            (_, motion_t, _, _, _, _, _, _) = self.process_one_image(
                image, motion_gt, extri_opencv, intri_opencv,
                original_size, target_image_shape, filepath=image_path,
            )
            np.random.set_state(post_state)
            motion_t = (motion_t > 0.5).astype(np.float32)

            images.append(image_t)
            depths.append(depth_t)
            extrinsics.append(extri_t)
            intrinsics.append(intri_t)
            cam_points.append(cam_pts)
            world_points.append(world_pts)
            point_masks.append(point_mask)
            motion_masks.append(motion_t)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        batch = {
            "seq_name": "sintel_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        if self.dynamic_source == "gt_flow":
            batch["motion_mask"] = motion_masks
        return batch
