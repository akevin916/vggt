# Dyn-VGGT TartanAir V1 dataset loader.
#
# TartanAir V1 is a synthetic indoor/outdoor benchmark with dense depth maps and exact
# 6-DoF camera poses.  We use it as a STATIC negative-example dataset for S1/S2 mixed
# training: every scene has camera-only motion, so `motion_mask` is always an all-zero
# array (m=0 everywhere) — every pixel is GT-static. This lets L_gate / static_photo (and
# anything else keyed on batch["motion_mask"]) train directly on TartanAir, not just via
# ComposedDataset's zero-fill fallback for datasets that omit the key entirely.
#
# Disk layout:  <TARTANAIR_DIR>/train/<env>/<difficulty>/<traj>/
#   image_left/<FFFFF>_left.png         RGB frames  (640 × 640, uint8)
#   depth_left/<FFFFF>_left_depth.npy   float32 depth in metres, same resolution
#   pose_left.txt                        per-frame cam pose, one line per frame:
#                                         x y z qx qy qz qw  (TartanAir NED convention)
#
# Camera intrinsics (V1, all envs / difficulties):
#   fx = fy = 320   cx = cy = 320   W = H = 640
#
# Coordinate convention:
#   TartanAir poses are cam-to-world in a NED frame (x=forward, y=left, z=up).
#   We apply the same axis permutation as MonST3R's TartanAirDUSt3R loader to convert
#   to the standard OpenCV cam-to-world (x=right, y=down, z=forward), then invert to
#   obtain the world-to-cam extrinsic expected by VGGT.

import os
import os.path as osp
import glob
import logging
import random

import numpy as np

from data.dataset_util import read_image_cv2, threshold_depth_map
from data.base_dataset import BaseDataset


class TartanAirDataset(BaseDataset):

    # Fixed intrinsics (V1, all sequences)
    _FX = _FY = 320.0
    _CX = _CY = 320.0

    def __init__(
        self,
        common_conf,
        TARTANAIR_DIR: str = "/media/cvml-75/ssd2t1/data/tartanair",
        min_num_images: int = 16,
        len_train: int = 100000,
        depth_max: float = 1000.0,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.TARTANAIR_DIR = TARTANAIR_DIR
        self.depth_max = depth_max
        self.len_train = len_train

        # ── Scan sequences ──────────────────────────────────────────────────────────
        # Each (env, difficulty, traj) triple is treated as an independent sequence.
        self.data_store = {}   # seq_name -> {seq_dir, num_frames, pose_path}

        train_dir = osp.join(TARTANAIR_DIR, "train")
        if not osp.isdir(train_dir):
            raise FileNotFoundError(f"TartanAir train directory not found: {train_dir}")

        for env in sorted(os.listdir(train_dir)):
            env_dir = osp.join(train_dir, env)
            if not osp.isdir(env_dir):
                continue
            for diff in sorted(os.listdir(env_dir)):
                diff_dir = osp.join(env_dir, diff)
                if not osp.isdir(diff_dir):
                    continue
                for traj in sorted(os.listdir(diff_dir)):
                    traj_dir = osp.join(diff_dir, traj)
                    if not osp.isdir(traj_dir):
                        continue

                    img_dir = osp.join(traj_dir, "image_left")
                    pose_path = osp.join(traj_dir, "pose_left.txt")
                    if not (osp.isdir(img_dir) and osp.isfile(pose_path)):
                        continue

                    frames = sorted(glob.glob(osp.join(img_dir, "*_left.png")))
                    num_frames = len(frames)
                    if num_frames < min_num_images:
                        continue

                    seq_name = f"{env}__{diff}__{traj}"
                    self.data_store[seq_name] = {
                        "seq_dir": traj_dir,
                        "num_frames": num_frames,
                        "pose_path": pose_path,
                    }

        self.sequence_list = list(self.data_store.keys())
        if self.debug:
            self.sequence_list = self.sequence_list[:4]
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: TartanAir loaded {self.sequence_list_len} sequences"
        )

    # ── Coordinate helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _pose_to_extri(pose_line: np.ndarray) -> np.ndarray:
        """Convert a TartanAir pose line [x,y,z,qx,qy,qz,qw] → world-to-cam extri (3×4).

        MonST3R axis permutation:
          new_z = old_x (forward),  new_x = old_y (left),  new_y = old_z (up)
          Same remap for quaternion components.
        This converts the NED cam-to-world pose to an OpenCV cam-to-world, then inverts
        to obtain the world-to-cam extrinsic expected by VGGT.
        """
        z, x, y = pose_line[:3]          # NED xyz → OpenCV xyz remapping
        qz, qx, qy, qw = pose_line[3:]   # NED qxqyqzqw → OpenCV qxqyqzqw remapping

        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 0] = 1 - 2*qy*qy - 2*qz*qz
        c2w[0, 1] = 2*qx*qy - 2*qz*qw
        c2w[0, 2] = 2*qx*qz + 2*qy*qw
        c2w[1, 0] = 2*qx*qy + 2*qz*qw
        c2w[1, 1] = 1 - 2*qx*qx - 2*qz*qz
        c2w[1, 2] = 2*qy*qz - 2*qx*qw
        c2w[2, 0] = 2*qx*qz - 2*qy*qw
        c2w[2, 1] = 2*qy*qz + 2*qx*qw
        c2w[2, 2] = 1 - 2*qx*qx - 2*qy*qy
        c2w[:3, 3] = [x, y, z]

        return np.linalg.inv(c2w)[:3].astype(np.float32)   # (3, 4)

    # ── Main data loading ────────────────────────────────────────────────────────────

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

        # Sample temporally-ordered frame ids
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

        # Load all poses at once (cheap; files are small)
        all_poses = np.loadtxt(store["pose_path"], dtype=np.float64)   # (N, 7)

        intri_template = np.array(
            [[self._FX, 0.0, self._CX],
             [0.0, self._FY, self._CY],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        motion_masks = []
        image_paths, original_sizes = [], []

        for fid in ids:
            fid = int(fid)
            image_path = osp.join(
                seq_dir, "image_left", f"{fid:06d}_left.png"
            )
            image = read_image_cv2(image_path)
            if image is None:
                logging.warning(f"TartanAir: missing image {image_path}, using zeros.")
                image = np.zeros((640, 640, 3), dtype=np.uint8)
            original_size = np.array(image.shape[:2])

            if self.load_depth:
                depth_path = osp.join(
                    seq_dir, "depth_left", f"{fid:06d}_left_depth.npy"
                )
                depth_map = np.load(depth_path).astype(np.float32)
                depth_map = threshold_depth_map(depth_map, max_depth=self.depth_max)
            else:
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            extri_opencv = self._pose_to_extri(all_poses[fid])
            intri_opencv = intri_template.copy()

            (image_t, depth_t, extri_t, intri_t,
             world_pts, cam_pts, point_mask, _) = self.process_one_image(
                image, depth_map, extri_opencv, intri_opencv,
                original_size, target_image_shape, filepath=image_path,
            )

            images.append(image_t)
            depths.append(depth_t)
            extrinsics.append(extri_t)
            intrinsics.append(intri_t)
            cam_points.append(cam_pts)
            world_points.append(world_pts)
            point_masks.append(point_mask)
            motion_masks.append(np.zeros(depth_t.shape, dtype=np.float32))  # camera-only motion: all-static
            image_paths.append(image_path)
            original_sizes.append(original_size)

        return {
            "seq_name": "tartanair_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "motion_mask": motion_masks,   # all-zero: every scene has camera-only motion
            "original_sizes": original_sizes,
        }
