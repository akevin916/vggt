# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# NEW: Dyn-VGGT PointOdyssey dataset (I-1). PointOdyssey is the cornerstone dynamic training set:
#      it provides dense depth, per-frame camera (extrinsics/intrinsics), and dense dynamic instance
#      masks — so it supplies ground-truth `motion_mask` for L_motion direct supervision (docs §5.1).
#      Adapted from reference/monst3r/dust3r/datasets/pointodyssey.py to the VGGT BaseDataset interface.

import os
import os.path as osp
import glob
import logging
import random

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


class PointOdysseyDataset(BaseDataset):
    # NEW: returns the standard VGGT fields plus a binary `motion_mask` (1=dynamic foreground).
    def __init__(
        self,
        common_conf,
        split: str = "train",
        PO_DIR: str = "/media/cvml-75/ssd2t1/data/point_odyssey",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        depth_max: float = 1000.0,   # PointOdyssey depth png is uint16 normalised to [0, depth_max] meters
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if PO_DIR is None:
            raise ValueError("PO_DIR (PointOdyssey root) must be specified.")
        self.PO_DIR = PO_DIR
        self.depth_max = depth_max

        if split == "train":
            self.len_train = len_train
            split_dir = "train"
        elif split == "test":
            self.len_train = len_test
            split_dir = "test"
        else:
            raise ValueError(f"Invalid split: {split}")

        self.min_num_images = min_num_images
        self.invalid_sequence = []

        # Scan sequences: each valid sequence has rgbs/, depths/, masks/ and anno.npz.
        self.data_store = {}   # seq_name -> dict(seq_dir, num_frames)
        seq_dirs = sorted(glob.glob(osp.join(self.PO_DIR, split_dir, "*/")))
        if self.debug:
            seq_dirs = seq_dirs[:2]

        total_frame_num = 0
        for seq_dir in seq_dirs:
            seq_name = seq_dir.rstrip("/").split("/")[-1]
            if seq_name in self.invalid_sequence:
                continue
            rgb_dir = osp.join(seq_dir, "rgbs")
            anno_path = osp.join(seq_dir, "anno.npz")
            if not (osp.isdir(rgb_dir) and osp.isfile(anno_path)):
                continue
            num_frames = len(glob.glob(osp.join(rgb_dir, "rgb_*.jpg")))
            if num_frames < min_num_images:
                continue
            self.data_store[seq_name] = {"seq_dir": seq_dir, "num_frames": num_frames}
            total_frame_num += num_frames

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: PointOdyssey size: {self.sequence_list_len} seqs, {total_frame_num} frames")

    def _binary_dynamic_mask(self, mask_path, hw):
        # Heuristic: PointOdyssey masks segment the animated foreground agents (dynamic).
        # Any non-black pixel -> dynamic. Returns float32 {0,1} of shape (H, W).
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            return np.zeros(hw, dtype=np.float32)
        if m.ndim == 3:
            m = m.sum(axis=-1)
        return (m > 0).astype(np.float32)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        store = self.data_store[seq_name]
        seq_dir = store["seq_dir"]
        num_frames = store["num_frames"]

        # Sample temporally-ordered frame ids (video!). Optionally cluster them around an anchor.
        if ids is None:
            if self.get_nearby:
                # Cluster img_per_seq frames around a random anchor (seed list length = img_per_seq
                # so get_nearby_ids returns exactly img_per_seq ids; see BaseDataset.get_nearby_ids).
                anchor = np.random.randint(0, num_frames)
                seed_ids = [anchor] * img_per_seq
                ids = self.get_nearby_ids(seed_ids, num_frames, expand_ratio=2.0)
            else:
                ids = np.random.choice(num_frames, img_per_seq, replace=self.allow_duplicate_img)
            ids = np.sort(np.asarray(ids))  # keep chronological order for temporal attention

        # anno.npz holds per-frame camera (intrinsics 3x3, extrinsics 4x4 world-to-cam = OpenCV extri).
        anno = np.load(osp.join(seq_dir, "anno.npz"), allow_pickle=True)
        all_intri = anno["intrinsics"].astype(np.float32)
        all_extri = anno["extrinsics"].astype(np.float32)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, cam_points, world_points, point_masks = [], [], [], [], []
        extrinsics, intrinsics, motion_masks, image_paths, original_sizes = [], [], [], [], []

        for fid in ids:
            fid = int(fid)
            image_path = osp.join(seq_dir, "rgbs", f"rgb_{fid:05d}.jpg")
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])

            if self.load_depth:
                depth_path = osp.join(seq_dir, "depths", f"depth_{fid:05d}.png")
                depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                depth_map = depth16.astype(np.float32) / 65535.0 * self.depth_max
                depth_map[depth_map >= self.depth_max] = 0.0  # drop sky/invalid
            else:
                depth_map = None

            mask_path = osp.join(seq_dir, "masks", f"mask_{fid:05d}.png")
            motion_gt = self._binary_dynamic_mask(mask_path, tuple(original_size))

            extri_opencv = all_extri[fid][:3, :4]
            intri_opencv = all_intri[fid]

            # NEW: capture RNG state so the motion-mask transform replays the SAME random augmentation.
            pre_state = np.random.get_state()
            # Transform image+depth+camera with the shared geometric pipeline.
            (image_t, depth_t, extri_t, intri_t,
             world_pts, cam_pts, point_mask, _) = self.process_one_image(
                image, depth_map, extri_opencv, intri_opencv,
                original_size, target_image_shape, filepath=image_path,
            )

            # Align motion_mask identically by replaying the same transforms: restore RNG, pass the
            # binary mask as a pseudo-depth (nearest-interp), keep only its spatial output.
            post_state = np.random.get_state()
            np.random.set_state(pre_state)
            (_, motion_t, _, _, _, _, _, _) = self.process_one_image(
                image, motion_gt, extri_opencv, intri_opencv,
                original_size, target_image_shape, filepath=image_path,
            )
            np.random.set_state(post_state)  # do not perturb the global RNG stream
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

        set_name = "pointodyssey"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "motion_mask": motion_masks,   # NEW: GT dynamic mask for L_motion (docs §5.1)
            "original_sizes": original_sizes,
        }
        return batch
