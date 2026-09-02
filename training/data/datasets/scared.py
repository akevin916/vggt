# Dyn-VGGT SCARED dataset loader.
#
# SCARED is a stereo ENDOSCOPY dataset (porcine abdominal anatomy, surgical instruments).
# Everything here reads the CONVERTED layout produced by data/preprocess/scared_convert.py;
# the raw release's quirks (BGR-swapped tiffs, dataset 8/9 off-by-one, stacked left/right
# point maps) were absorbed at conversion time. Full spec: docs/topics/scared_dataset.md.
#
# Disk layout:  <SCARED_DIR>/<split>/dataset{n}/keyframe{m}/
#   image_left/{fid:06d}.png     RGB 1280x1024 uint8
#   depth_left/{fid:06d}.png     uint16, /100 -> MILLIMETRES, 0 = invalid
#   cam_data/intrinsics.txt      one line: fx fy cx cy (constant within a keyframe)
#   cam_data/extrinsics.txt      N x 12, world-to-cam 3x4 row-major
#   cam_data/frames.txt          N original frame indices (file names)
#   cam_data/valid_frac.txt      N valid-depth fractions
#
# Three things that differ from the other loaders:
#
#  1. UNITS ARE MILLIMETRES, and stay that way. trainer.py normalises each sample to unit
#     scale before the model sees it, so absolute units are irrelevant -- but `depth_max` is
#     applied BEFORE that normalisation and must therefore be in mm (655.35 = the conversion
#     ceiling; every keyframe's p99.9 is <= 165).
#
#  2. EXTRINSICS ARE ALREADY WORLD-TO-CAM. No inversion, no axis permutation -- unlike
#     TartanAir/Spring/Waymo, which all store cam-to-world.
#
#  3. `ids` INDEX THE FRAME LIST, NOT THE FRAME NUMBER. val keyframes start at fid=2 and the
#     test split is a sparse sampling (stride 8-38), so fid != position. Every id here is a
#     position into frames.txt; file names come from frames.txt[id].
#
# Sampling: depth is sparse (4-73% valid) and some frames are near-empty. loss.py:181 derives
# a whole sample's camera-loss validity from its FIRST frame alone, so an empty anchor
# silently discards that sample's pose supervision. Anchors are therefore drawn only from
# frames with valid_frac > `min_anchor_valid_frac`.
#
# motion_mask: SCARED has NO dynamic annotation, and the RAFT flow-residual recipe used by
# Spring/Waymo was measured NOT to work here (docs/topics/scared_dataset.md). `dynamic_source` is
# therefore "none" by default, which returns all-zero masks. NOTE that all-zero means
# "everything is GT-static", which is a LIE for endoscopy (deforming tissue, moving
# instruments) -- it is a placeholder to keep the key present, not a label. Do not enable
# L_gate on this dataset while it holds.

import logging
import os
import os.path as osp
import random

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2, threshold_depth_map
from data.paths import data_path

DEPTH_SCALE = 100.0  # uint16 counts per mm


class ScaredDataset(BaseDataset):

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SCARED_DIR: str = data_path("train", "scared"),
        min_num_images: int = 16,
        len_train: int = 10000,
        depth_max: float = 655.35,          # MILLIMETRES (see note 1)
        dynamic_source: str = "none",
        min_anchor_valid_frac: float = 0.05,
        nearby_expand_range: int | None = None,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.SCARED_DIR = SCARED_DIR
        self.depth_max = depth_max
        self.len_train = len_train
        self.min_anchor_valid_frac = min_anchor_valid_frac
        # Half-width of the get_nearby sampling window, in frames. None keeps the inherited
        # expand_ratio=2.0 behaviour (window = +-2*img_per_seq, i.e. +-24 for a 12-view
        # sample). Measured GT camera displacement on the train split, median over the 22
        # keyframes: gap 1 = 0.26 mm, 24 = 4.40, 48 = 6.91, 120 = 12.40, 240 = 19.26,
        # 480 = 21.72 -- the endoscope loiters and doubles back, so displacement SATURATES
        # past ~240 and a wider window buys baseline at the cost of view overlap.
        self.nearby_expand_range = nearby_expand_range

        if dynamic_source not in ("none",):
            raise ValueError(
                f"SCARED has no dynamic annotation; dynamic_source must be 'none', got "
                f"{dynamic_source!r}. See docs/topics/scared_dataset.md."
            )
        self.dynamic_source = dynamic_source

        split_dir = osp.join(SCARED_DIR, split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(f"SCARED split directory not found: {split_dir}")

        self.data_store = {}
        for ds in sorted(os.listdir(split_dir)):
            ds_dir = osp.join(split_dir, ds)
            if not osp.isdir(ds_dir):
                continue
            for kf in sorted(os.listdir(ds_dir)):
                seq_dir = osp.join(ds_dir, kf)
                cam_dir = osp.join(seq_dir, "cam_data")
                if not osp.isdir(osp.join(seq_dir, "image_left")) or not osp.isdir(cam_dir):
                    continue

                fids = np.loadtxt(osp.join(cam_dir, "frames.txt"), dtype=np.int64, ndmin=1)
                if len(fids) < min_num_images:
                    continue
                extri = np.loadtxt(osp.join(cam_dir, "extrinsics.txt")).reshape(-1, 3, 4)
                vf = np.loadtxt(osp.join(cam_dir, "valid_frac.txt"), ndmin=1)
                fx, fy, cx, cy = np.loadtxt(osp.join(cam_dir, "intrinsics.txt"))

                anchors = np.nonzero(vf > self.min_anchor_valid_frac)[0]
                if len(anchors) == 0:
                    logging.warning(f"SCARED: {ds}/{kf} has no frame above the anchor "
                                    f"valid-depth threshold; skipping.")
                    continue

                self.data_store[f"{ds}__{kf}"] = dict(
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
        logging.info(f"{status}: SCARED loaded {self.sequence_list_len} sequences "
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
                # anchor from the dense-depth pool only (see module header)
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
                logging.warning(f"SCARED: missing image {image_path}, using zeros.")
                image = np.zeros((1024, 1280, 3), dtype=np.uint8)
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
            motion_masks.append(np.zeros(depth_t.shape, dtype=np.float32))  # placeholder, NOT a label
            image_paths.append(image_path)
            original_sizes.append(original_size)

        return {
            "seq_name": "scared_" + seq_name,
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
