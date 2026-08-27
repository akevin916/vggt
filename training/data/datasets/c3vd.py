# Dyn-VGGT C3VD dataset loader.
#
# C3VD is COLONOSCOPY footage (real Olympus CF-HQ190L, silicone colon phantoms) registered
# against a CT-derived mesh, so every frame carries a GT depth map and a GT camera pose.
# This reads ONLY the converted layout written by data/preprocess/c3vd_convert.py; the raw
# release's quirks -- fisheye intrinsics, transposed pose.txt, two distinct invalid depth
# codes -- are absorbed at conversion time and never appear here.
#
# Deliberately a separate file from scared.py rather than a subclass. The layouts match
# today by construction, but the two datasets disagree on almost everything that matters
# (depth density, depth range, split semantics, how anchors should be drawn), and a shared
# base would make every future C3VD-only decision a SCARED regression risk.
#
# Disk layout:  <C3VD_DIR>/<split>/<anatomy>/t<tex>_<vid>/
#   image_left/{fid:06d}.png     RGB, undistorted to a PINHOLE camera (default 1024x1024)
#   depth_left/{fid:06d}.png     uint16, /100 -> MILLIMETRES, 0 = invalid
#   cam_data/intrinsics.txt      one line: fx fy cx cy  (the pinhole the conversion resampled
#                                onto -- NOT a calibration of the physical scope)
#   cam_data/extrinsics.txt      N x 12, world-to-cam 3x4 row-major
#   cam_data/frames.txt          N original frame indices (== 0..N-1; C3VD is gapless)
#   cam_data/valid_frac.txt      N valid-depth fractions
#   meta.json                    fov_deg, K, source sequence, units
#
# Five things worth knowing before changing anything here:
#
#  1. UNITS ARE MILLIMETRES and the release CLAMPS AT 100 mm. trainer.py normalises each
#     sample to unit scale, so absolute units do not reach the model -- but `depth_max` is
#     applied BEFORE that, so it must be in mm. 100.0 is the conversion ceiling, not a
#     tuning knob: anything beyond it was already discarded as invalid upstream.
#
#  2. EXTRINSICS ARE ALREADY WORLD-TO-CAM. No inversion, no axis permutation.
#
#  3. DEPTH IS DENSE, unlike SCARED. Typical valid fraction is the whole undistorted disc,
#     the invalid part being the fisheye corners the pinhole grid cannot fill. So
#     `min_anchor_valid_frac` is a guard against a corrupt frame, not the sampling strategy
#     it has to be on SCARED.
#
#  4. SPLIT IS BY WHOLE SEQUENCE, held out by texture (see c3vd_convert.SPLIT_MAP). Adjacent
#     colonoscopy frames are near-duplicates, so a frame-level split would leak outright.
#     test additionally contains the only descending-colon sequence -> unseen anatomy too.
#
#  5. NO DYNAMIC ANNOTATION, and unlike SCARED the scene really is rigid: a silicone phantom
#     with a moving camera. All-zero motion_mask is therefore closer to true here than
#     anywhere else in the repo -- but it is still a placeholder to keep the key present,
#     not a label, and L_gate has nothing to learn from a scene with no moving parts.
#
# Sampling: the endoscope advances slowly along the lumen, so consecutive frames carry very
# little baseline. `nearby_expand_range` widens the window a sample is drawn from; measure
# GT displacement per gap on the converted data before picking a value, the way SCARED's
# note records it.

import logging
import os
import os.path as osp
import random

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2, threshold_depth_map
from data.paths import data_path

DEPTH_SCALE = 100.0  # uint16 counts per mm, set by c3vd_convert.py


class C3vdDataset(BaseDataset):

    def __init__(
        self,
        common_conf,
        split: str = "train",
        C3VD_DIR: str = data_path("train", "c3vd"),
        min_num_images: int = 16,
        len_train: int = 10000,
        depth_max: float = 100.0,           # MILLIMETRES (see note 1)
        dynamic_source: str = "none",
        min_anchor_valid_frac: float = 0.30,
        nearby_expand_range: int | None = None,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.C3VD_DIR = C3VD_DIR
        self.depth_max = depth_max
        self.len_train = len_train
        self.min_anchor_valid_frac = min_anchor_valid_frac
        self.nearby_expand_range = nearby_expand_range

        if dynamic_source not in ("none",):
            raise ValueError(
                f"C3VD has no dynamic annotation (rigid phantom, moving camera); "
                f"dynamic_source must be 'none', got {dynamic_source!r}."
            )
        self.dynamic_source = dynamic_source

        split_dir = osp.join(C3VD_DIR, split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(
                f"C3VD split directory not found: {split_dir}. Run "
                f"data/preprocess/c3vd_convert.py first."
            )

        self.data_store = {}
        for ana in sorted(os.listdir(split_dir)):
            ana_dir = osp.join(split_dir, ana)
            if not osp.isdir(ana_dir):
                continue
            for vid in sorted(os.listdir(ana_dir)):
                seq_dir = osp.join(ana_dir, vid)
                cam_dir = osp.join(seq_dir, "cam_data")
                if not osp.isdir(osp.join(seq_dir, "image_left")) or not osp.isdir(cam_dir):
                    continue

                fids = np.loadtxt(osp.join(cam_dir, "frames.txt"), dtype=np.int64, ndmin=1)
                if len(fids) < min_num_images:
                    logging.info(f"C3VD: {ana}/{vid} has {len(fids)} frames "
                                 f"(< {min_num_images}); skipping.")
                    continue
                extri = np.loadtxt(osp.join(cam_dir, "extrinsics.txt")).reshape(-1, 3, 4)
                vf = np.loadtxt(osp.join(cam_dir, "valid_frac.txt"), ndmin=1)
                fx, fy, cx, cy = np.loadtxt(osp.join(cam_dir, "intrinsics.txt"))

                anchors = np.nonzero(vf > self.min_anchor_valid_frac)[0]
                if len(anchors) == 0:
                    logging.warning(f"C3VD: {ana}/{vid} has no frame above the anchor "
                                    f"valid-depth threshold; skipping.")
                    continue

                self.data_store[f"{ana}__{vid}"] = dict(
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
        if self.sequence_list_len == 0:
            raise RuntimeError(f"C3VD: no usable sequence under {split_dir}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: C3VD loaded {self.sequence_list_len} sequences "
                     f"from split '{split}'")

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
            if self.get_nearby:
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
                raise FileNotFoundError(f"C3VD: missing image {image_path}")
            original_size = np.array(image.shape[:2])

            if self.load_depth:
                d16 = cv2.imread(osp.join(seq_dir, "depth_left", f"{fid:06d}.png"),
                                 cv2.IMREAD_UNCHANGED)
                depth_map = (d16.astype(np.float32) / DEPTH_SCALE) if d16 is not None \
                    else np.zeros(image.shape[:2], dtype=np.float32)
                depth_map = threshold_depth_map(depth_map, max_depth=self.depth_max)
            else:
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            extri_opencv = store["extri"][i]        # already world-to-cam (see note 2)
            intri_opencv = store["intri"].copy()

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
            motion_masks.append(np.zeros(depth_t.shape, dtype=np.float32))  # placeholder (note 5)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        return {
            "seq_name": "c3vd_" + seq_name,
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
