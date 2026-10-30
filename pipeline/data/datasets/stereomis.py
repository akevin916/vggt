# Dyn-VGGT StereoMIS dataset loader (方向 3: instrument-mask training, planning/方向3_StereoMIS器械遮罩訓練.md).
#
# StereoMIS is a stereo ENDOSCOPY dataset (porcine, da Vinci kinematics as pose GT) with a per-
# frame instrument mask. It has NO depth, so every sample comes back with all-False point_masks;
# the trainer's `data.no_depth` path (scale from the GT trajectory, camera loss forced on) is what
# makes it trainable. Do not mix it into a depth-supervised config.
#
# Disk layout:  <ROOT>/<split>/<seq>/          (converted by the 方向 2 pipeline)
#   image_left/{fid:06d}.png     RGB 640x512 uint8
#   mask_left/{fid:06d}.png      uint8, 255 = valid, 0 = instrument
#   cam_data/intrinsics.txt      one line: fx fy cx cy
#   cam_data/extrinsics.txt      N x 12, world-to-cam 3x4 row-major, METRES
#   cam_data/frames.txt          N frame ids (stride 2 in the source video, P1 has two gaps)
#   cam_data/valid_frac.txt      N flags, 1 = usable, 0 = instrument covers >= 90% of the frame
#
# `ids` are ROW indices into frames.txt (one row = one 27 fps step), like ScaredDataset, so the
# temporal RoPE and camera_smooth's dt see the same spacing eval sees on contiguous frames.
#
# motion_mask carries the INSTRUMENT pixel mask (1 = instrument), warped through the exact
# augmentation the image got. The trainer pools it to patch flags for train_mask; nothing else
# reads it in the 方向 3 configs (gate / static_photo / ego_flow are all off).

import logging
import os
import os.path as osp
import random

import cv2
import numpy as np

from pipeline.data.base_dataset import BaseDataset
from pipeline.data.dataset_util import read_image_cv2
from pipeline.data.paths import data_path


class StereoMISDataset(BaseDataset):

    def __init__(
        self,
        common_conf,
        split: str = "train",
        ROOT: str = data_path("train", "stereomis"),
        seqs=None,                 # None = every sequence under <ROOT>/<split>
        len_train: int = 10000,
        min_num_images: int = 16,
    ):
        super().__init__(common_conf=common_conf)

        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.len_train = len_train

        split_dir = osp.join(ROOT, split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(f"StereoMIS split directory not found: {split_dir}")
        names = [str(s) for s in seqs] if seqs else sorted(os.listdir(split_dir))

        self.data_store = {}
        for seq in names:
            seq_dir = osp.join(split_dir, seq)
            cam_dir = osp.join(seq_dir, "cam_data")
            fids = np.loadtxt(osp.join(cam_dir, "frames.txt"), dtype=np.int64, ndmin=1)
            vf_path = osp.join(cam_dir, "valid_frac.txt")
            if not osp.isfile(vf_path):
                raise FileNotFoundError(f"{vf_path} missing: run the 方向 3 prep (usable-frame flags)")
            usable = np.nonzero(np.loadtxt(vf_path, ndmin=1) > 0.5)[0]
            if len(usable) < min_num_images:
                logging.warning(f"StereoMIS: {seq} has {len(usable)} usable frames; skipping.")
                continue
            extri = np.loadtxt(osp.join(cam_dir, "extrinsics.txt")).reshape(-1, 3, 4).astype(np.float32)
            fx, fy, cx, cy = np.loadtxt(osp.join(cam_dir, "intrinsics.txt"))
            C = -np.einsum("nji,nj->ni", extri[:, :, :3].astype(np.float64), extri[:, :, 3])
            step = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
            self.data_store[seq] = dict(
                seq_dir=seq_dir, fids=fids, extri=extri, usable=usable,
                intri=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32),
                traj_step=step,   # median adjacent camera displacement, metres (s_gt floor)
            )

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise RuntimeError(f"StereoMIS: no usable sequence in {split_dir} ({names})")
        logging.info(f"{'Training' if self.training else 'Testing'}: StereoMIS loaded "
                     f"{self.sequence_list_len} sequences from '{split}': {self.sequence_list}")

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None,
                 aspect_ratio: float = 1.0) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        store = self.data_store[seq_name]
        seq_dir, fids, usable = store["seq_dir"], store["fids"], store["usable"]

        if ids is None:
            # Sample POSITIONS in the usable-row list, so a frame the instrument covers >= 90%
            # never enters a clip (not only never anchors it).
            if self.get_nearby:
                anchor = random.randint(0, len(usable) - 1)
                pos = self.get_nearby_ids([anchor] * img_per_seq, len(usable), expand_ratio=2.0)
            else:
                pos = np.random.choice(len(usable), img_per_seq, replace=self.allow_duplicate_img)
            ids = usable[np.asarray(pos)]
        ids = np.sort(np.asarray(ids))

        target_image_shape = self.get_target_shape(aspect_ratio)
        out = {k: [] for k in ("images", "depths", "extrinsics", "intrinsics", "cam_points",
                               "world_points", "point_masks", "motion_mask", "original_sizes")}
        for i in ids:
            i = int(i)
            fid = int(fids[i])
            image_path = osp.join(seq_dir, "image_left", f"{fid:06d}.png")
            image = read_image_cv2(image_path)
            m = cv2.imread(osp.join(seq_dir, "mask_left", f"{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
            if image is None or m is None:
                raise FileNotFoundError(f"StereoMIS: missing image/mask for {seq_name} fid {fid}")
            original_size = np.array(image.shape[:2])
            inst = (m == 0).astype(np.float32)
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)
            extri, intri = store["extri"][i], store["intri"].copy()

            # Replay the RNG so the instrument mask gets the image's exact crop/scale (spring.py).
            pre_state = np.random.get_state()
            (image_t, depth_t, extri_t, intri_t,
             world_pts, cam_pts, point_mask, _) = self.process_one_image(
                image, depth_map, extri, intri, original_size, target_image_shape,
                filepath=image_path)
            post_state = np.random.get_state()
            np.random.set_state(pre_state)
            (_, inst_t, _, _, _, _, _, _) = self.process_one_image(
                image, inst, extri, intri, original_size, target_image_shape, filepath=image_path)
            np.random.set_state(post_state)

            out["images"].append(image_t)
            out["depths"].append(depth_t)
            out["extrinsics"].append(extri_t)
            out["intrinsics"].append(intri_t)
            out["cam_points"].append(cam_pts)
            out["world_points"].append(world_pts)
            out["point_masks"].append(point_mask)
            out["motion_mask"].append((inst_t > 0.5).astype(np.float32))
            out["original_sizes"].append(original_size)

        out.update(seq_name="stereomis_" + seq_name, ids=ids, frame_num=len(ids),
                   traj_step=store["traj_step"])
        return out
