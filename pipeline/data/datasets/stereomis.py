# Dyn-VGGT StereoMIS dataset loader (camera-pose only).
#
# StereoMIS is in-vivo porcine laparoscopy with da Vinci kinematics as pose. Reads the layout
# produced by the 2026-10-14 conversion (docs/plan/StereoMIS_mask.md §1.1):
#
#   <STEREOMIS_DIR>/<split>/<seq>/
#     image_left/{fid:06d}.png     RGB 640x512 (left eye, NOT rectified / undistorted)
#     mask_left/{fid:06d}.png      255 = usable, 0 = instrument or whole-frame unusable
#     cam_data/frames.txt          N video frame numbers (1-based)
#     cam_data/extrinsics.txt      N x 12, world-to-cam 3x4 row-major, METRES
#     cam_data/intrinsics.txt      one line: fx fy cx cy
#
# THERE IS NO DEPTH. Two things in the training pipeline read depth-derived points even when
# only the camera loss is on, so this loader supplies a PSEUDO depth:
#
#   * loss.compute_camera_loss drops a whole sample unless frame 0 has > 100 valid points;
#   * normalization.normalize_camera_extrinsics_and_points_batch divides translation by the
#     mean distance of the valid world points (clamped to >= 1e-6 -- with no points the
#     translation target would be blown up by 1e6).
#
# The pseudo depth is a constant `pseudo_depth` (metres) on every pixel where mask_left is 255
# and 0 elsewhere. Consequences, stated so nobody mistakes it for a label:
#   - translation targets are divided by a FIXED length, so the model regresses translation in
#     units of `pseudo_depth` (metric-consistent across samples) instead of per-sample scene
#     scale. Rotation and focal are unaffected.
#   - point/depth losses must stay off (`loss.depth: null`, no flow_geom); they would train on
#     a fake plane.
#   - the instrument mask doubles as point validity, so a frame that is mostly instrument/
#     unusable cannot anchor a sample.
#
# `ids` index frames.txt, not frame numbers. Poses are only available on frames that have a
# mask (train: every 2nd video frame), so a gap of g positions = 2g video frames.

import logging
import os
import os.path as osp
import random

import cv2
import numpy as np

from pipeline.data.base_dataset import BaseDataset
from pipeline.data.dataset_util import read_image_cv2
from pipeline.data.paths import data_path


class StereoMisDataset(BaseDataset):

    def __init__(
        self,
        common_conf,
        split: str = "train",
        STEREOMIS_DIR: str = data_path("train", "stereomis"),
        sequences: list | None = None,       # None = every sequence directory in the split
        min_num_images: int = 16,
        len_train: int = 10000,
        pseudo_depth: float = 0.07,          # METRES; see header. StereoMIS scene depth ~50-100 mm
        min_anchor_valid_frac: float = 0.5,  # share of mask==255 pixels the anchor frame needs
        nearby_expand_range: int | None = None,
        contiguous_stride: int | None = None,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.STEREOMIS_DIR = STEREOMIS_DIR
        self.len_train = len_train
        self.pseudo_depth = pseudo_depth
        self.min_anchor_valid_frac = min_anchor_valid_frac
        self.nearby_expand_range = nearby_expand_range
        # Same semantics as ScaredDataset: an int fixes the stride, a list draws one per clip.
        if contiguous_stride is None or isinstance(contiguous_stride, int):
            self.contiguous_strides = [contiguous_stride] if contiguous_stride else None
        else:
            self.contiguous_strides = [int(s) for s in contiguous_stride]

        split_dir = osp.join(STEREOMIS_DIR, split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(f"StereoMIS split directory not found: {split_dir}")

        names = sorted(os.listdir(split_dir)) if sequences is None else list(sequences)
        self.data_store = {}
        for seq in names:
            seq_dir = osp.join(split_dir, seq)
            cam_dir = osp.join(seq_dir, "cam_data")
            if not osp.isdir(osp.join(seq_dir, "image_left")) or not osp.isdir(cam_dir):
                if sequences is not None:
                    raise FileNotFoundError(f"StereoMIS sequence not found: {seq_dir}")
                continue

            fids = np.loadtxt(osp.join(cam_dir, "frames.txt"), dtype=np.int64, ndmin=1)
            if len(fids) < min_num_images:
                continue
            extri = np.loadtxt(osp.join(cam_dir, "extrinsics.txt")).reshape(-1, 3, 4)
            fx, fy, cx, cy = np.loadtxt(osp.join(cam_dir, "intrinsics.txt"))
            anchors = np.nonzero(self._mask_valid_frac(seq_dir, fids) >= min_anchor_valid_frac)[0]
            if len(anchors) == 0:
                logging.warning(f"StereoMIS: {seq} has no usable anchor frame; skipping.")
                continue

            self.data_store[seq] = dict(
                seq_dir=seq_dir,
                fids=fids,
                extri=extri.astype(np.float32),
                intri=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32),
                num_frames=len(fids),
                anchors=anchors,
            )

        self.sequence_list = list(self.data_store.keys())
        if self.debug:
            self.sequence_list = self.sequence_list[:4]
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: StereoMIS loaded {self.sequence_list_len} sequences "
                     f"from split '{split}'")

    @staticmethod
    def _mask_valid_frac(seq_dir, fids):
        """Fraction of mask==255 pixels per frame. Scanned once, cached beside cam_data."""
        cache = osp.join(seq_dir, "cam_data", "mask_valid_frac.txt")
        if osp.isfile(cache):
            vf = np.loadtxt(cache, ndmin=1)
            if len(vf) == len(fids):
                return vf
        vf = np.zeros(len(fids), dtype=np.float32)
        for k, fid in enumerate(fids):
            m = cv2.imread(osp.join(seq_dir, "mask_left", f"{int(fid):06d}.png"),
                           cv2.IMREAD_UNCHANGED)
            vf[k] = float((m == 255).mean()) if m is not None else 0.0
        try:
            np.savetxt(cache, vf, fmt="%.4f")
        except OSError:
            logging.warning(f"StereoMIS: cannot write {cache}; mask scan repeats next start.")
        return vf

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
        seq_dir, fids = store["seq_dir"], store["fids"]
        num_frames = store["num_frames"]

        if ids is None:
            if self.get_nearby and self.contiguous_strides:
                fits = [s for s in self.contiguous_strides if s * (img_per_seq - 1) < num_frames]
                stride = int(np.random.choice(fits)) if fits else 1
                span = stride * (img_per_seq - 1)
                pool = store["anchors"][store["anchors"] + span < num_frames]
                anchor = int(np.random.choice(pool if len(pool) else store["anchors"]))
                start = max(min(anchor, num_frames - 1 - span), 0)
                ids = start + stride * np.arange(img_per_seq)
            elif self.get_nearby:
                anchor = int(np.random.choice(store["anchors"]))
                if self.nearby_expand_range is None:
                    ids = self.get_nearby_ids([anchor] * img_per_seq, num_frames,
                                              expand_ratio=2.0)
                else:
                    ids = self.get_nearby_ids([anchor] * img_per_seq, num_frames,
                                              expand_range=self.nearby_expand_range)
            else:
                ids = np.random.choice(num_frames, img_per_seq,
                                       replace=self.allow_duplicate_img)
            ids = np.sort(np.asarray(ids))

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        motion_masks, image_paths, original_sizes = [], [], []

        for i in ids:
            i = int(i)
            fid = int(fids[i])
            image_path = osp.join(seq_dir, "image_left", f"{fid:06d}.png")
            image = read_image_cv2(image_path)
            if image is None:
                logging.warning(f"StereoMIS: missing image {image_path}, using zeros.")
                image = np.zeros((512, 640, 3), dtype=np.uint8)
            original_size = np.array(image.shape[:2])

            # pseudo depth: constant where the instrument mask says "usable" (see header)
            m = cv2.imread(osp.join(seq_dir, "mask_left", f"{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
            if m is None or m.shape != image.shape[:2]:
                m = np.zeros(image.shape[:2], dtype=np.uint8)
            depth_map = np.where(m == 255, self.pseudo_depth, 0.0).astype(np.float32)

            (image_t, depth_t, extri_t, intri_t,
             world_pts, cam_pts, point_mask, _) = self.process_one_image(
                image, depth_map, store["extri"][i].copy(), store["intri"].copy(),
                original_size, target_image_shape, filepath=image_path,
            )

            images.append(image_t)
            depths.append(depth_t)
            extrinsics.append(extri_t)
            intrinsics.append(intri_t)
            cam_points.append(cam_pts)
            world_points.append(world_pts)
            point_masks.append(point_mask)
            motion_masks.append(np.zeros(depth_t.shape, dtype=np.float32))  # placeholder, NOT a label
            image_paths.append(image_path)
            original_sizes.append(original_size)

        return {
            "seq_name": "stereomis_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "motion_mask": motion_masks,
            "original_sizes": original_sizes,
        }
