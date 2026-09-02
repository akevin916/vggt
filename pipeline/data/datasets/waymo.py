# Dyn-VGGT Waymo dataset loader.
#
# Uses the pre-processed Waymo Open Dataset produced by MonST3R's
# `datasets_preprocess/preprocess_waymo.py`.  The preprocessing script converts
# each segment's raw tfrecord into a flat directory of per-frame trios:
#
#   <WAYMO_DIR>/segment-<ID>.tfrecord/
#       <FFFFF>_<C>.jpg    RGB frame (H × W, uint8)
#       <FFFFF>_<C>.exr    Sparse LiDAR depth projected onto the camera plane (float32)
#       <FFFFF>_<C>.npz    Camera parameters:
#                            intrinsics  (3, 3)  – standard pinhole K (already undistorted)
#                            cam2world   (4, 4)  – camera-to-world in Waymo global frame
#                            distortion  (5,)    – stored but not applied here
#
# where FFFFF is the zero-padded frame index and C ∈ {1,2,3,4,5} is the camera ID
# (1 = front, 2 = front-left, 3 = front-right, 4 = side-left, 5 = side-right).
#
# Each (segment, camera) pair is treated as an independent video sequence to ensure
# temporal coherence: frames from a single camera are captured at a fixed viewpoint
# trajectory, making them suitable for temporal-attention training.
#
# Coordinate note:
#   Waymo's cam2world translations are in a global coordinate frame and can be large
#   (O(10^3–10^4) metres).  To avoid float32 precision loss we shift every sequence so
#   that the mean camera position of the sampled frames is at the origin.  This is a
#   pure rigid-body shift of the world coordinate system and does not affect any
#   relative geometry (depth, optical flow, reprojection).
#
# Dynamic GT (`motion_mask`):
#   Waymo ships no per-pixel dynamic segmentation GT, so the only geometrically-derivable
#   label is the RAFT flow-residual mask (§3.4 (m_geo), preprocess/waymo_raft_dynmask.py ->
#   <seg>/dynmask_raft/dyn_{fid:05d}_{cam_id}.png, one dir per segment shared across cameras,
#   with a per-camera `.done_cam{cam_id}` completion flag). Selected via dynamic_source:
#     "none" — motion_mask absent from get_data() (ComposedDataset zero-fills; NOT a valid
#              label for real dynamic scenes, kept only as a legacy/debug escape hatch).
#     "raft" — load the RAFT mask. Because precompute is a long offline job that is only
#              PARTIALLY done, in this mode a (segment, camera) sequence is ENROLLED ONLY IF
#              its `.done_cam{cam_id}` flag exists — uncomputed segments are skipped entirely
#              rather than fed as (wrong) all-static labels. A per-frame png that is still
#              missing inside an enrolled sequence falls back to all-zero (all-static).

import os
import os.path as osp
import logging
import random
from collections import defaultdict

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

from pipeline.data.dataset_util import read_image_cv2, threshold_depth_map
from pipeline.data.base_dataset import BaseDataset
from pipeline.data.paths import data_path


def _read_waymo_depth(path: str) -> np.ndarray:
    """Read a single-channel float32 EXR depth map (Waymo preprocessed format)."""
    d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if d is None:
        return None
    # Single-channel EXR → already (H, W); multi-channel → take channel 0
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d[d > 1e9] = 0.0
    return d


class WaymoDataset(BaseDataset):

    _DYNAMIC_SOURCES = ("none", "raft")

    def __init__(
        self,
        common_conf,
        WAYMO_DIR: str = data_path("train", "waymo_processed"),
        cameras: list = None,      # which cameras to include; None → all five (1–5)
        min_num_images: int = 16,
        len_train: int = 100000,
        depth_max: float = 80.0,   # LiDAR valid range; beyond this set to 0
        dynamic_source: str = "none",
        dynamic_max_frac: float = 0.5,
    ):
        super().__init__(common_conf=common_conf)

        if dynamic_source not in self._DYNAMIC_SOURCES:
            raise ValueError(f"dynamic_source must be one of {self._DYNAMIC_SOURCES}, got {dynamic_source!r}")
        self.dynamic_source = dynamic_source
        # Safeguard against RAFT false-positive blowouts: a frame whose mask covers > this
        # fraction of pixels is reverted to all-static. Waymo's sparse-LiDAR masks sit at ~5%
        # so this never triggers in practice, but it keeps the raft path uniform with Spring.
        self.dynamic_max_frac = dynamic_max_frac

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.WAYMO_DIR = WAYMO_DIR
        self.cameras = cameras if cameras is not None else [1, 2, 3, 4, 5]
        self.depth_max = depth_max
        self.len_train = len_train

        # ── Scan sequences ───────────────────────────────────────────────────────────
        # Each (segment_dir, camera_id) pair with enough frames is one sequence.
        self.data_store = {}    # seq_name -> {seg_dir, cam_id, frame_ids}

        if not osp.isdir(WAYMO_DIR):
            raise FileNotFoundError(
                f"Waymo processed directory not found: {WAYMO_DIR}"
            )

        seg_dirs = sorted(
            d for d in (osp.join(WAYMO_DIR, n) for n in os.listdir(WAYMO_DIR))
            if osp.isdir(d)
        )
        if self.debug:
            seg_dirs = seg_dirs[:4]

        for seg_dir in seg_dirs:
            seg_name = osp.basename(seg_dir)
            # Group frame indices by camera
            cam_frames: dict[int, list[int]] = defaultdict(list)
            for fname in os.listdir(seg_dir):
                if not fname.endswith(".jpg"):
                    continue
                stem = fname[:-4]          # e.g. "00042_3"
                parts = stem.split("_")
                if len(parts) != 2:
                    continue
                try:
                    frame_idx = int(parts[0])
                    cam_id = int(parts[1])
                except ValueError:
                    continue
                if cam_id in self.cameras:
                    npz = osp.join(seg_dir, f"{frame_idx:05d}_{cam_id}.npz")
                    if not osp.isfile(npz):
                        continue
                    cam_frames[cam_id].append(frame_idx)

            for cam_id, frame_ids in cam_frames.items():
                frame_ids = sorted(frame_ids)
                if len(frame_ids) < min_num_images:
                    continue
                # In raft mode, enroll only sequences whose dynmask precompute has completed
                # (partial offline job); otherwise we'd feed uncomputed dynamic scenes as
                # all-static labels. See class header.
                if self.dynamic_source == "raft":
                    done_flag = osp.join(seg_dir, "dynmask_raft", f".done_cam{cam_id}")
                    if not osp.isfile(done_flag):
                        continue
                seq_name = f"{seg_name}__cam{cam_id}"
                self.data_store[seq_name] = {
                    "seg_dir": seg_dir,
                    "cam_id": cam_id,
                    "frame_ids": np.array(frame_ids, dtype=np.int32),
                }

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: Waymo loaded {self.sequence_list_len} sequences "
            f"(cameras={self.cameras})"
        )

    def _binary_dynamic_mask(self, seg_dir, fid, cam_id, hw):
        # RAFT flow-residual dynamic mask (§3.4 (m_geo)), uint8 {0,255} at native res, one dir per
        # segment shared across cameras: dynmask_raft/dyn_{fid:05d}_{cam_id}.png. Returns float32
        # {0,1} of shape hw; all-zero (all-static) if disabled or the per-frame png is missing.
        if self.dynamic_source != "raft":
            return np.zeros(hw, dtype=np.float32)
        mask_path = osp.join(seg_dir, "dynmask_raft", f"dyn_{fid:05d}_{cam_id}.png")
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            return np.zeros(hw, dtype=np.float32)
        if m.ndim == 3:
            m = m.sum(axis=-1)
        m = (m > 0).astype(np.float32)
        # Reject RAFT false-positive blowouts (see __init__): revert over-dynamic frames to static.
        if m.mean() > self.dynamic_max_frac:
            return np.zeros(hw, dtype=np.float32)
        return m

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
        seg_dir = store["seg_dir"]
        cam_id = store["cam_id"]
        frame_ids = store["frame_ids"]   # sorted array of available frame indices
        num_frames = len(frame_ids)

        # Sample temporally-ordered frame ids (indices into frame_ids array)
        if ids is None:
            if self.get_nearby:
                anchor = np.random.randint(0, num_frames)
                seed_ids = [anchor] * img_per_seq
                local_ids = self.get_nearby_ids(seed_ids, num_frames, expand_ratio=2.0)
            else:
                local_ids = np.random.choice(
                    num_frames, img_per_seq, replace=self.allow_duplicate_img
                )
            local_ids = np.sort(np.asarray(local_ids))
        else:
            local_ids = np.sort(np.asarray(ids))

        # Translate local indices → actual Waymo frame numbers
        selected_frame_ids = frame_ids[local_ids]

        # ── Pre-load cam2world to compute per-sequence coordinate centering ─────────
        # Subtracting the mean camera position keeps world coordinates near the origin,
        # preventing float32 precision loss from Waymo's large global translations.
        cam2world_all = []
        for fid in selected_frame_ids:
            npz_path = osp.join(seg_dir, f"{fid:05d}_{cam_id}.npz")
            cam_data = np.load(npz_path)
            cam2world_all.append(cam_data["cam2world"].astype(np.float64))

        mean_t = np.mean([c[:3, 3] for c in cam2world_all], axis=0)   # (3,)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        motion_masks = []
        image_paths, original_sizes = [], []

        for fid, c2w_raw in zip(selected_frame_ids, cam2world_all):
            img_path = osp.join(seg_dir, f"{fid:05d}_{cam_id}.jpg")
            exr_path = osp.join(seg_dir, f"{fid:05d}_{cam_id}.exr")
            npz_path = osp.join(seg_dir, f"{fid:05d}_{cam_id}.npz")

            # RGB
            image = read_image_cv2(img_path)
            if image is None:
                logging.warning(f"Waymo: missing image {img_path}, using zeros.")
                # best-effort: guess shape from intrinsics
                cam_data = np.load(npz_path)
                K = cam_data["intrinsics"].astype(np.float32)
                H = int(round(2 * K[1, 2]))
                W = int(round(2 * K[0, 2]))
                image = np.zeros((H, W, 3), dtype=np.uint8)
            original_size = np.array(image.shape[:2])

            # Depth (sparse LiDAR projected to image plane)
            if self.load_depth:
                depth_map = _read_waymo_depth(exr_path)
                if depth_map is None:
                    depth_map = np.zeros(image.shape[:2], dtype=np.float32)
                else:
                    depth_map[depth_map > self.depth_max] = 0.0
            else:
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            # Camera parameters — apply coordinate centering to translation
            cam_data = np.load(npz_path)
            K = cam_data["intrinsics"].astype(np.float32)   # (3, 3)
            c2w = c2w_raw.copy()
            c2w[:3, 3] -= mean_t                             # shift to local frame

            extri_opencv = np.linalg.inv(c2w)[:3].astype(np.float32)   # (3, 4)

            motion_gt = self._binary_dynamic_mask(seg_dir, int(fid), cam_id, tuple(original_size))

            # NEW: capture RNG state so the motion-mask transform replays the SAME augmentation.
            pre_state = np.random.get_state()
            (image_t, depth_t, extri_t, intri_t,
             world_pts, cam_pts, point_mask, _) = self.process_one_image(
                image, depth_map, extri_opencv, K,
                original_size, target_image_shape, filepath=img_path,
            )

            # Align motion_mask identically: restore RNG, pass the binary mask as a pseudo-depth
            # through the same geometric pipeline, keep only its spatial output (mirrors
            # PointOdysseyDataset.get_data).
            post_state = np.random.get_state()
            np.random.set_state(pre_state)
            (_, motion_t, _, _, _, _, _, _) = self.process_one_image(
                image, motion_gt, extri_opencv, K,
                original_size, target_image_shape, filepath=img_path,
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
            image_paths.append(img_path)
            original_sizes.append(original_size)

        batch = {
            "seq_name": "waymo_" + seq_name,
            "ids": local_ids,
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
