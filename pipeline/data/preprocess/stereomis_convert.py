#!/usr/bin/env python3
"""Convert StereoMIS's raw release into the SCARED-style layout the VGGT dataloaders expect.

Raw release (``<raw_dir>/<seq>/``): one stereo video (1280x2048, left view stacked on top
of right), ``groundtruth.txt`` (one camera pose per video frame), ``StereoCalibration.ini``
and ``masks/{NNNNNN}l.png``. There is NO depth.

Output (``<out_dir>/``, mirrors data/train/scared):
  split/{train,test}.txt                  ``<seq>\\t<fid>\\tl`` per converted frame
  <split>/<seq>/image_left/{fid:06d}.png  RGB 640x512 uint8 (left view, INTER_AREA /2)
  <split>/<seq>/mask_left/{fid:06d}.png   uint8 640x512, 255 = valid, 0 = excluded
                                          (instrument, or a whole unusable frame)
  <split>/<seq>/cam_data/intrinsics.txt   one line: fx fy cx cy (at 640x512)
  <split>/<seq>/cam_data/extrinsics.txt   N x 12, world-to-cam 3x4 row-major
  <split>/<seq>/cam_data/frames.txt       N frame ids (= file names)
  <split>/<seq>/meta.json
  <split>/<seq>/rgb.mp4, groundtruth.txt, StereoCalibration.ini   hard links to the raw files

Decisions absorbed here, so nothing downstream has to remember them:

  * FRAME IDS ARE 1-BASED. Mask names run 1..N for an N-frame video (P2_8: 19052 frames,
    masks 000001..019052), so ``fid`` = mask number and the video frame is decoded at
    0-based index fid-1. Its pose is groundtruth row fid-1+pose_offset.
  * THE FRAME LIST IS THE MASK LIST. The release's masks already encode the benchmark's
    frame selection: step 2 on P1/P2_0-P2_5 and step 1 on P2_6-P2_8 (sequences.txt), and
    on P1 only the train_split.csv ranges (P1's last range 16801-19140 has no masks and is
    not converted). Converting exactly the masked frames keeps image, mask and pose aligned
    by construction.
  * SPLIT: train = P1, P2_0..P2_5 (step 2); test = P2_6..P2_8 (step 1). There is no val.
    P3 is listed in sequences.txt but absent from the release.
  * POSES are the LEFT camera's cam-to-world, TUM format ``id tx ty tz qx qy qz qw``,
    metres; inverted here to world-to-cam. Source: the authors' own code
    (aimi-lab/robust-pose-estimator: core/utils/trajectory.py, scripts/infer_trajectory.py
    use groundtruth.txt directly as the left-eye trajectory, no hand-eye transform). The
    release itself does not document this.
  * POSE/VIDEO OFFSET (--pose_offset): video frame k pairs with groundtruth row k+offset.
    The authors evaluate with offset=-4; P1 has exactly 4 more video frames than poses
    (19180 vs 19176). Frames whose row falls outside groundtruth are left out of
    frames.txt/extrinsics.txt (their images stay on disk). Images do not depend on the
    offset: change it with --poses_only.
  * LEFT = TOP HALF, per the authors' dataset/video_dataset.py (_split_stereo_img).
  * UNDECODABLE TAIL: up to MAX_TAIL_DROP frames at the end of a video may fail to decode
    (P2_5's last frame does); they are dropped and listed in meta.json.
  * No undistortion: the radial terms are ~0.004/0.015 at 1280x1024.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import os.path as osp
import shutil
import time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.data.paths import data_path

TRAIN_SEQS = ["P1", "P2_0", "P2_1", "P2_2", "P2_3", "P2_4", "P2_5"]
TEST_SEQS = ["P2_6", "P2_7", "P2_8"]
H_EYE, W_EYE = 1024, 1280
H_OUT, W_OUT = 512, 640
SCALE = W_OUT / W_EYE
MAX_TAIL_DROP = 3   # undecodable frames tolerated at the very end of a video


def seq_video(seq_dir: str) -> str:
    vids = [f for f in os.listdir(seq_dir) if f.endswith(".mp4")]
    assert len(vids) == 1, f"{seq_dir}: expected one mp4, found {vids}"
    return osp.join(seq_dir, vids[0])


def mask_fids(seq_dir: str) -> np.ndarray:
    names = [f for f in os.listdir(osp.join(seq_dir, "masks")) if f.endswith("l.png")]
    return np.array(sorted(int(f[:6]) for f in names), dtype=np.int64)


def left_intrinsics(seq_dir: str) -> np.ndarray:
    ini = configparser.ConfigParser()
    ini.read(osp.join(seq_dir, "StereoCalibration.ini"))
    L = ini["StereoLeft"]
    assert (int(L["res_x"]), int(L["res_y"])) == (W_EYE, H_EYE)
    fx, fy = float(L["fc_x"]) * SCALE, float(L["fc_y"]) * SCALE
    # pixel-centre convention, matching cv2.resize's sampling grid
    cx = (float(L["cc_x"]) + 0.5) * SCALE - 0.5
    cy = (float(L["cc_y"]) + 0.5) * SCALE - 0.5
    return np.array([fx, fy, cx, cy])


def w2c_from_groundtruth(seq_dir: str, fids: np.ndarray, offset: int):
    """(fids that have a pose, their world-to-cam 3x4 flattened to N x 12)."""
    gt = np.loadtxt(osp.join(seq_dir, "groundtruth.txt"))
    assert np.array_equal(gt[:, 0], np.arange(len(gt))), "groundtruth ids not 0..N-1"
    row = fids - 1 + offset
    keep = (row >= 0) & (row < len(gt))
    rows = gt[row[keep]]
    c2w = np.tile(np.eye(4), (len(rows), 1, 1))
    c2w[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()   # scipy order: x y z w
    c2w[:, :3, 3] = rows[:, 1:4]
    w2c = np.linalg.inv(c2w)
    return fids[keep], w2c[:, :3, :].reshape(len(rows), 12)


def write_cam_data(src: str, dst: str, fids: np.ndarray, offset: int) -> int:
    K = left_intrinsics(src)
    posed, extri = w2c_from_groundtruth(src, fids, offset)
    os.makedirs(osp.join(dst, "cam_data"), exist_ok=True)
    np.savetxt(osp.join(dst, "cam_data", "intrinsics.txt"), K[None], fmt="%.6f")
    np.savetxt(osp.join(dst, "cam_data", "extrinsics.txt"), extri, fmt="%.9e")
    np.savetxt(osp.join(dst, "cam_data", "frames.txt"), posed, fmt="%d")
    meta_path = osp.join(dst, "meta.json")
    if osp.exists(meta_path):                 # --poses_only: keep meta in sync
        with open(meta_path) as f:
            meta = json.load(f)
        meta.update(pose_offset=offset, num_posed_frames=int(len(posed)))
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    return len(posed)


def convert_seq(args) -> dict:
    seq, split, raw_dir, out_dir, poses_only, max_frames, offset = args
    t0 = time.time()
    src = osp.join(raw_dir, seq)
    dst = osp.join(out_dir, split, seq)
    fids = mask_fids(src)
    if max_frames:
        fids = fids[:max_frames]

    if poses_only:   # only frames that were actually converted
        on_disk = set(int(f[:6]) for f in os.listdir(osp.join(dst, "image_left")))
        n_posed = write_cam_data(src, dst, fids[np.isin(fids, list(on_disk))], offset)
        return {"seq": seq, "n": n_posed, "sec": time.time() - t0}

    for d in ("image_left", "mask_left"):
        os.makedirs(osp.join(dst, d), exist_ok=True)
    video = seq_video(src)
    for name, s in (("rgb.mp4", video), ("groundtruth.txt", osp.join(src, "groundtruth.txt")),
                    ("StereoCalibration.ini", osp.join(src, "StereoCalibration.ini"))):
        d = osp.join(dst, name)
        if not osp.exists(d):
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)

    want = set(fids.tolist())
    last = int(fids.max())
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    written, idx = [], 0                      # idx = 0-based video frame index
    while idx < last:
        ok = cap.grab()
        if not ok:
            break
        fid = idx + 1
        if fid in want:
            ok, fr = cap.retrieve()
            assert ok and fr.shape[:2] == (2 * H_EYE, W_EYE), f"{seq} frame {idx}: {fr.shape}"
            left = cv2.resize(fr[:H_EYE], (W_OUT, H_OUT), interpolation=cv2.INTER_AREA)
            cv2.imwrite(osp.join(dst, "image_left", f"{fid:06d}.png"), left)
            m = cv2.imread(osp.join(src, "masks", f"{fid:06d}l.png"), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(osp.join(dst, "mask_left", f"{fid:06d}.png"), m)
            written.append(fid)
        idx += 1
    # A few trailing frames can be undecodable (P2_5: the last HEVC frame, "Could not find
    # ref with POC"). Tolerate a short tail; anything else is a real failure.
    missing = sorted(want - set(written))
    if missing and (len(missing) > MAX_TAIL_DROP or missing[0] < fids[-MAX_TAIL_DROP]):
        raise RuntimeError(f"{seq}: {len(missing)} frames not decoded, first {missing[:5]} "
                           f"(video ended at index {idx})")
    fids = np.array(written, dtype=np.int64)
    n_posed = write_cam_data(src, dst, fids, offset)

    meta = {
        "seq": seq, "split": split, "num_frames": int(len(fids)),
        "num_posed_frames": n_posed, "pose_offset": offset,
        "dropped_undecodable_fids": missing,
        "fid_first": int(fids[0]), "fid_last": int(fids[-1]),
        "fid_convention": "1-based; video index = fid-1; groundtruth row = fid-1+pose_offset",
        "source_video": osp.basename(video), "source_fps": fps,
        "image_size_hw": [H_OUT, W_OUT], "resize_from_hw": [H_EYE, W_EYE],
        "intrinsics_fx_fy_cx_cy": left_intrinsics(src).tolist(),
        "extrinsics": "world-to-cam of the LEFT eye, inverted from groundtruth.txt "
                      "(TUM cam-to-world, metres)",
        "mask": "255 = valid, 0 = excluded",
        "depth": None,
    }
    with open(osp.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {"seq": seq, "n": len(written), "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default=data_path("train", "StereoMIS"))
    ap.add_argument("--out_dir", default=data_path("train", "stereomis"))
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of sequences (default: all)")
    ap.add_argument("--poses_only", action="store_true",
                    help="rewrite cam_data/ only (e.g. after changing the pose convention)")
    ap.add_argument("--max_frames", type=int, default=0, help="smoke test: first N frames only")
    ap.add_argument("--pose_offset", type=int, default=0,
                    help="video frame k <-> groundtruth row k+offset (authors evaluate with -4)")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    assert osp.realpath(args.raw_dir) != osp.realpath(args.out_dir)
    plan = [(s, "train") for s in TRAIN_SEQS] + [(s, "test") for s in TEST_SEQS]
    if args.seqs:
        plan = [p for p in plan if p[0] in args.seqs]

    jobs = [(s, sp, args.raw_dir, args.out_dir, args.poses_only, args.max_frames,
             args.pose_offset) for s, sp in plan]
    with Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(convert_seq, jobs):
            print(f"[done] {r['seq']}: {r['n']} frames in {r['sec']:.0f}s", flush=True)

    os.makedirs(osp.join(args.out_dir, "split"), exist_ok=True)
    for split in ("train", "test"):
        lines = []
        # every converted sequence on disk, not only this run's --seqs subset
        for seq in (TRAIN_SEQS if split == "train" else TEST_SEQS):
            if not osp.exists(osp.join(args.out_dir, split, seq, "cam_data", "frames.txt")):
                continue
            fids = np.loadtxt(osp.join(args.out_dir, split, seq, "cam_data", "frames.txt"),
                              dtype=np.int64, ndmin=1)
            lines += [f"{seq}\t{fid}\tl" for fid in fids]
        if lines:
            with open(osp.join(args.out_dir, "split", f"{split}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
