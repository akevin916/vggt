# Dyn-VGGT Spring dataset loader.
#
# Spring is a synthetic dynamic benchmark (Blender-rendered) with dense stereo
# disparity, 6-DoF camera poses, and animated characters.  We use it as a
# DYNAMIC-scene dataset for depth/pose supervision.
#
# Dynamic GT (`motion_mask`): Spring ships no per-pixel dynamic segmentation GT, so the only
# geometrically-derivable label is the RAFT flow-residual mask (§3.4 (m_geo), precompute_spring_raft_
# dynmask.py -> dynmask_raft/dyn_{frame_idx:04d}.png). Selected via dynamic_source:
#   "none" — motion_mask absent from get_data() (ComposedDataset zero-fills; NOT valid for a
#            dynamic scene, only kept as a legacy/debug escape hatch).
#   "raft" — load dynmask_raft/dyn_{frame_idx:04d}.png (used for S1/S2 gate + static_photo).
# A frame whose dynmask_raft png is missing falls back to an all-zero (all-static) mask.
#
# Disk layout:  <SPRING_DIR>/train/<NNNN>/
#   frame_left/frame_left_NNNN.png       RGB frames  (1080 × 1920, uint8)
#   disp1_left/disp1_left_NNNN.dsp5      Stereo disparity at 2× resolution (HDF5, float16)
#   cam_data/extrinsics.txt              Per-frame 4×4 cam-to-world matrix (16 values/line)
#   cam_data/intrinsics.txt              Per-frame fx fy cx cy
#   dynmask_raft/dyn_NNNN.png            RAFT flow-residual dynamic mask (uint8 {0,255}, optional)
#
# Depth conversion:
#   Disparity is stored at 2× image resolution (3840 × 2160).
#   depth = fx * baseline / (disp_2x / 2)  where baseline = 0.065 m (Spring stereo rig).
#
# Coordinate convention:
#   Spring extrinsics are cam-to-world (c2w).  We invert to world-to-cam (w2c)
#   3×4, which is the extrinsic format expected by VGGT.

import os
import os.path as osp
import glob
import logging
import random

import cv2
import h5py
import numpy as np

from data.dataset_util import read_image_cv2, threshold_depth_map
from data.base_dataset import BaseDataset
from data.paths import data_path


def _read_dsp5(path: str) -> np.ndarray:
    """Read a Spring .dsp5 disparity file (HDF5 with key 'disparity')."""
    with h5py.File(path, "r") as f:
        return f["disparity"][()].astype(np.float32)


class SpringDataset(BaseDataset):

    BASELINE = 0.065  # stereo baseline in metres (Spring benchmark spec)

    _DYNAMIC_SOURCES = ("none", "raft")

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SPRING_DIR: str = data_path("train", "spring"),
        min_num_images: int = 16,
        len_train: int = 100000,
        depth_max: float = 200.0,
        dynamic_source: str = "none",
        dynamic_max_frac: float = 0.5,
    ):
        super().__init__(common_conf=common_conf)

        if dynamic_source not in self._DYNAMIC_SOURCES:
            raise ValueError(f"dynamic_source must be one of {self._DYNAMIC_SOURCES}, got {dynamic_source!r}")
        self.dynamic_source = dynamic_source
        # RAFT ego-flow on Spring blows up under large camera motion, flagging whole static
        # backgrounds as dynamic. A frame whose mask covers > this fraction of pixels is almost
        # certainly such a false-positive blowout, so revert it to all-static (see _binary_dynamic_mask).
        self.dynamic_max_frac = dynamic_max_frac

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.SPRING_DIR = SPRING_DIR
        self.depth_max = depth_max
        self.len_train = len_train

        split_dir = osp.join(SPRING_DIR, split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(f"Spring {split} directory not found: {split_dir}")

        self.data_store = {}

        for seq in sorted(os.listdir(split_dir)):
            seq_dir = osp.join(split_dir, seq)
            frame_dir = osp.join(seq_dir, "frame_left")
            cam_dir = osp.join(seq_dir, "cam_data")
            if not (osp.isdir(frame_dir) and osp.isdir(cam_dir)):
                continue

            frames = sorted(glob.glob(osp.join(frame_dir, "frame_left_*.png")))
            num_frames = len(frames)
            if num_frames < min_num_images:
                continue

            intri_path = osp.join(cam_dir, "intrinsics.txt")
            extri_path = osp.join(cam_dir, "extrinsics.txt")
            if not (osp.isfile(intri_path) and osp.isfile(extri_path)):
                continue

            self.data_store[seq] = {
                "seq_dir": seq_dir,
                "num_frames": num_frames,
                "intri_path": intri_path,
                "extri_path": extri_path,
            }

        self.sequence_list = list(self.data_store.keys())
        if self.debug:
            self.sequence_list = self.sequence_list[:4]
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: Spring loaded {self.sequence_list_len} sequences"
        )

    def _binary_dynamic_mask(self, seq_dir, frame_idx, hw):
        # RAFT flow-residual dynamic mask (§3.4 (m_geo)); precomputed as uint8 {0,255} at native res,
        # 1-based frame_idx to match frame_left_XXXX.png. Returns float32 {0,1} of shape hw;
        # all-zero (all-static) if disabled or the file is missing for this frame.
        if self.dynamic_source != "raft":
            return np.zeros(hw, dtype=np.float32)
        mask_path = osp.join(seq_dir, "dynmask_raft", f"dyn_{frame_idx:04d}.png")
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            return np.zeros(hw, dtype=np.float32)
        if m.ndim == 3:
            m = m.sum(axis=-1)
        m = (m > 0).astype(np.float32)
        # Reject RAFT false-positive blowouts: a frame with > dynamic_max_frac dynamic pixels is
        # treated as unreliable and reverted to all-static (no scene is that dynamic).
        if m.mean() > self.dynamic_max_frac:
            return np.zeros(hw, dtype=np.float32)
        return m

    def _disp_to_depth(self, disp_2x: np.ndarray, fx: float) -> np.ndarray:
        """Convert 2×-resolution disparity to depth at image resolution."""
        disp = disp_2x[::2, ::2]  # downsample to image resolution
        valid = np.isfinite(disp) & (disp > 0.01)
        depth = np.zeros_like(disp)
        depth[valid] = fx * self.BASELINE / disp[valid]
        depth[depth > self.depth_max] = 0.0
        return depth

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
        seq_dir = store["seq_dir"]
        num_frames = store["num_frames"]

        if ids is None:
            if self.get_nearby:
                anchor = np.random.randint(0, num_frames)
                seed_ids = [anchor] * img_per_seq
                ids = self.get_nearby_ids(seed_ids, num_frames, expand_ratio=2.0)
            else:
                ids = np.random.choice(
                    num_frames, img_per_seq, replace=self.allow_duplicate_img
                )
            ids = np.sort(np.asarray(ids))

        # Load camera parameters (all frames at once)
        all_intri = np.loadtxt(store["intri_path"], dtype=np.float64)   # (N, 4): fx fy cx cy
        all_extri = np.loadtxt(store["extri_path"], dtype=np.float64)   # (N, 16): flattened 4×4 c2w

        # Coordinate centering: shift world origin to mean camera position
        c2w_all = [all_extri[int(fid)].reshape(4, 4) for fid in ids]
        mean_t = np.mean([c[:3, 3] for c in c2w_all], axis=0)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        motion_masks = []
        image_paths, original_sizes = [], []

        for fid, c2w_raw in zip(ids, c2w_all):
            fid = int(fid)
            # Frame indices in Spring are 1-based
            frame_idx = fid + 1

            image_path = osp.join(seq_dir, "frame_left", f"frame_left_{frame_idx:04d}.png")
            image = read_image_cv2(image_path)
            if image is None:
                logging.warning(f"Spring: missing image {image_path}, using zeros.")
                image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            original_size = np.array(image.shape[:2])

            if self.load_depth:
                disp_path = osp.join(
                    seq_dir, "disp1_left", f"disp1_left_{frame_idx:04d}.dsp5"
                )
                fx = all_intri[fid, 0]
                if osp.isfile(disp_path):
                    disp_2x = _read_dsp5(disp_path)
                    depth_map = self._disp_to_depth(disp_2x, fx)
                else:
                    depth_map = np.zeros(image.shape[:2], dtype=np.float32)
            else:
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            # Camera: c2w → w2c (VGGT convention)
            c2w = c2w_raw.copy()
            c2w[:3, 3] -= mean_t
            w2c = np.linalg.inv(c2w)
            extri_opencv = w2c[:3].astype(np.float32)  # (3, 4)

            K = np.array(
                [[all_intri[fid, 0], 0.0, all_intri[fid, 2]],
                 [0.0, all_intri[fid, 1], all_intri[fid, 3]],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

            motion_gt = self._binary_dynamic_mask(seq_dir, frame_idx, tuple(original_size))

            # NEW: capture RNG state so the motion-mask transform replays the SAME augmentation.
            pre_state = np.random.get_state()
            (image_t, depth_t, extri_t, intri_t,
             world_pts, cam_pts, point_mask, _) = self.process_one_image(
                image, depth_map, extri_opencv, K,
                original_size, target_image_shape, filepath=image_path,
            )

            # Align motion_mask identically: restore RNG, pass the binary mask as a pseudo-depth
            # through the same geometric pipeline, keep only its spatial output (mirrors
            # PointOdysseyDataset.get_data).
            post_state = np.random.get_state()
            np.random.set_state(pre_state)
            (_, motion_t, _, _, _, _, _, _) = self.process_one_image(
                image, motion_gt, extri_opencv, K,
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
            "seq_name": "spring_" + seq_name,
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
        # RAFT flow-residual dynamic mask (§3.4 (m_geo)) when dynamic_source="raft"; omitted for "none"
        # so ComposedDataset's zero-fill fallback applies.
        if self.dynamic_source == "raft":
            batch["motion_mask"] = motion_masks
        return batch
