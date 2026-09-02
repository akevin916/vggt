#!/usr/bin/env python3
"""Convert SCARED's raw release into the layout the VGGT dataloaders expect.

What the raw release gives us and what has to change:

  * RGB is locked inside ``data/rgb.mp4`` (1280x2048, left view stacked on top of right).
    We decode it once, sequentially, and write the left half per frame.
  * Geometry is ``data/scene_pointsNNNNNN.tiff`` -- a 2048x1280x3 float32 XYZ point map,
    31.5 MB/frame. VGGT only needs metric depth: X and Y are recoverable from Z and K
    (verified, 0.395 px median reprojection error), and the bottom half is the right
    camera, which a monocular multi-view model never reads. Keeping only the left Z as
    uint16 takes 31.5 MB/frame down to ~1.2 MB.
    NOTE: these TIFFs must be read with tifffile. ``cv2.imread`` silently returns the
    channels in BGR order, which turns Z into X.
  * Per-frame camera pose lives in one JSON per frame inside ``data/frame_data.tar.gz``.
    ``camera-pose`` is already world-to-cam (confirmed: treating it as c2w scatters a
    static scene by 41.6 mm between first and last frame, inverting it holds to 0.59 mm),
    so it is exactly VGGT's ``extri_opencv`` and needs no inversion. We collapse the
    per-frame JSONs into one extrinsics.txt; KL/DL are constant within a keyframe
    (verified bit-identical across every frame), so intrinsics are stored once.

  * DATASETS 8 AND 9 ARE OFF BY ONE. Their directories are keyframe_0..3 while every
    split file is 1-based, so ``dataset8/keyframe1`` in split/*.txt is the directory
    ``dataset_8/keyframe_0``. Taking the name literally loads a different keyframe of the
    same patient and indexes past the end of the sequence. This script resolves it once,
    at conversion time, and names the output after the split convention -- so nothing
    downstream ever has to remember the exception.

Frame selection (option B):
  train -- every frame of the sequence. The split drops only 217 of 15568 frames, and
           keeping them all makes the output a gapless 0..N-1 that the loader can index
           directly, the way TartanAir does.
  val/test -- only the frames the split lists. The remaining frames of those keyframes
           belong to held-out scenes; training on them would leak. test is deliberately
           left non-contiguous (stride 8-38): that sparsity is the benchmark's design,
           so filenames carry the ORIGINAL frame index and callers must not run
           get_nearby over it as if it were a video.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import os.path as osp
import re
import shutil
import tarfile
import time

import cv2
import numpy as np
import tifffile

from pipeline.data.paths import data_path

DEPTH_SCALE = 100.0                      # uint16 counts per mm -> 0.01 mm resolution
DEPTH_MAX_MM = 65535 / DEPTH_SCALE       # 655.35 mm; beyond this we mark invalid
LEFT_ROWS = 1024                         # top half of the stacked frame is the left view


def resolve_source(split_name: str) -> str:
    """split/*.txt name -> actual directory, absorbing the dataset 8/9 off-by-one."""
    m = re.match(r"dataset(\d+)/keyframe(\d+)$", split_name)
    if not m:
        raise ValueError(f"unparseable split entry: {split_name}")
    ds, kf = m.group(1), int(m.group(2))
    if ds in ("8", "9"):
        kf -= 1
    return f"dataset_{ds}/keyframe_{kf}"


def load_split(root: str) -> dict:
    """split name -> (split, sorted frame ids listed)."""
    out = {}
    for s in ("train", "val", "test"):
        for line in open(osp.join(root, "split", f"{s}.txt")):
            parts = line.split()
            if len(parts) < 2:
                continue
            out.setdefault(parts[0], (s, []))[1].append(int(parts[1]))
    return {k: (s, sorted(set(v))) for k, (s, v) in out.items()}


def read_poses(src_dir: str) -> dict:
    """frame id -> 3x4 world-to-cam, plus the (constant) intrinsics and calibration."""
    poses, calib = {}, None
    with tarfile.open(osp.join(src_dir, "data", "frame_data.tar.gz")) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            fid = int(re.search(r"(\d{6})\.json$", m.name).group(1))
            d = json.load(io.BytesIO(t.extractfile(m).read()))
            poses[fid] = np.asarray(d["camera-pose"], dtype=np.float64)[:3, :4]
            if calib is None:
                calib = d["camera-calibration"]
    return poses, calib


def convert_one(root: str, split_name: str, split: str, ids: list[int], out_root: str,
                copy_video: bool, override_ids: list[int] | None = None,
                out_split: str | None = None, note: str | None = None,
                no_depth: bool = False) -> dict:
    src = osp.join(root, resolve_source(split_name))
    dst = osp.join(out_root, out_split or split, split_name)
    img_dir, dep_dir, cam_dir = (osp.join(dst, d) for d in ("image_left", "depth_left", "cam_data"))
    for d in (img_dir, cam_dir) if no_depth else (img_dir, dep_dir, cam_dir):
        os.makedirs(d, exist_ok=True)

    poses, calib = read_poses(src)
    if no_depth:
        # The pose-only sequences keep scene_points-*.tar.gz packed (~39 GB unpacked for the
        # two of them) because ATE never reads depth. frame_data is then the only frame
        # count we have; it matched scene_points exactly on every keyframe converted so far.
        n_sp = len(poses)
    else:
        n_sp = len([p for p in os.listdir(osp.join(src, "data")) if p.startswith("scene_points")])
        if len(poses) != n_sp:
            raise RuntimeError(f"{split_name}: {len(poses)} poses vs {n_sp} scene_points")

    if override_ids is not None:
        ids = [i for i in override_ids if i < n_sp]
    else:
        ids = list(range(n_sp)) if split == "train" else [i for i in ids if i < n_sp]
    want = set(ids)

    # RGB: one sequential pass over the video, keeping only the frames we need.
    cap = cv2.VideoCapture(osp.join(src, "data", "rgb.mp4"))
    n_written, fid = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in want:
            cv2.imwrite(osp.join(img_dir, f"{fid:06d}.png"), frame[:LEFT_ROWS],
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            n_written += 1
        fid += 1
    cap.release()
    if n_written != len(ids):
        raise RuntimeError(f"{split_name}: wrote {n_written} images for {len(ids)} ids "
                           f"(video had {fid} frames)")

    # Depth: left half, Z channel only, quantised. Out-of-range -> 0, the repo's invalid marker.
    #
    # We also record each frame's valid fraction. SCARED's depth is sparse and some frames are
    # empty outright (dataset_9 runs 25-35% empty), and loss.py:181 derives its per-sample
    # validity from the FIRST sampled frame alone -- so an empty anchor silently drops that
    # whole sample's camera loss. Dropping those frames here would punch holes in the
    # otherwise gapless train sequences, so we keep them and publish the statistic instead,
    # letting a loader bias its sampling without changing the data.
    clipped_total = 0
    valid_frac = []
    for i in ([] if no_depth else ids):
        Z = tifffile.imread(osp.join(src, "data", f"scene_points{i:06d}.tiff"))[:LEFT_ROWS, :, 2]
        Z = np.nan_to_num(Z.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        over = Z > DEPTH_MAX_MM
        clipped_total += int(over.sum())
        Z[over] = 0.0
        Z[Z < 0] = 0.0
        valid_frac.append(float((Z > 0).mean()))
        cv2.imwrite(osp.join(dep_dir, f"{i:06d}.png"),
                    np.clip(Z * DEPTH_SCALE, 0, 65535).astype(np.uint16))

    K = np.asarray(calib["KL"], dtype=np.float64)
    np.savetxt(osp.join(cam_dir, "intrinsics.txt"),
               np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]]), fmt="%.8f",
               header="fx fy cx cy  (constant across frames)")
    np.savetxt(osp.join(cam_dir, "extrinsics.txt"),
               np.stack([poses[i].reshape(-1) for i in ids]), fmt="%.10f",
               header="world-to-cam 3x4, row-major; one line per frame in frames.txt order")
    np.savetxt(osp.join(cam_dir, "frames.txt"), np.array(ids), fmt="%d",
               header="original frame index of each row above / each image file")
    np.savetxt(osp.join(cam_dir, "valid_frac.txt"), np.array(valid_frac), fmt="%.6f",
               header="fraction of pixels with valid depth, same order as frames.txt")
    with open(osp.join(cam_dir, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)

    if copy_video:
        shutil.copy2(osp.join(src, "data", "rgb.mp4"), osp.join(dst, "rgb.mp4"))

    # "contiguous" means consecutive frames, not that the run starts at 0: val is frames
    # 2..285, which is a gapless clip and safe for nearby-frame sampling.
    contiguous = bool(len(ids) > 1 and np.all(np.diff(ids) == 1))
    meta = dict(split_name=split_name, split=split, source=resolve_source(split_name),
                source_frames=n_sp, stored_frames=len(ids), contiguous=contiguous,
                starts_at_zero=bool(ids[0] == 0),
                frame_range=[int(ids[0]), int(ids[-1])], depth_scale=DEPTH_SCALE,
                depth_max_mm=DEPTH_MAX_MM, depth_clipped_px=clipped_total,
                depth_written=not no_depth,
                valid_frac_median=float(np.median(valid_frac)) if valid_frac else None,
                valid_frac_min=float(np.min(valid_frac)) if valid_frac else None,
                n_frames_below_1pct=(int(np.sum(np.asarray(valid_frac) < 0.01))
                                     if valid_frac else None),
                video_copied=bool(copy_video))
    if note:
        meta["note"] = note
    with open(osp.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scared_dir", default=data_path("train", "scared"))
    ap.add_argument("--out_dir", default=None,
                    help="default: <scared_dir>/sample for --keyframes, else <scared_dir>")
    ap.add_argument("--keyframes", nargs="*", default=None,
                    help="split-style names, e.g. dataset2/keyframe1. Default: everything.")
    ap.add_argument("--no_video", action="store_true", help="do not copy rgb.mp4 alongside")
    ap.add_argument("--frames", default=None,
                    help="override the split's frame list with a CONTIGUOUS (stride-1) run: "
                         "'all' or 'a:b' (half-open). For val/test this pulls in frames the "
                         "split held out, so the result is probe/visualisation material only "
                         "-- never training data. Pair it with --out_split.")
    ap.add_argument("--out_split", default=None,
                    help="name of the split subdirectory to write into (default: the "
                         "keyframe's own split), so a --frames re-extraction does not "
                         "overwrite the canonical conversion.")
    ap.add_argument("--no_depth", action="store_true",
                    help="skip scene_points entirely (leaves the tarballs packed) and write "
                         "images + poses only. For the pose-eval sequences, whose ATE never "
                         "reads depth; frame count then comes from frame_data.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip keyframes whose meta.json is already written, so an "
                         "interrupted run resumes instead of redoing everything")
    args = ap.parse_args()

    split_map = load_split(args.scared_dir)
    names = args.keyframes or sorted(split_map)
    out_root = args.out_dir or osp.join(args.scared_dir, "sample" if args.keyframes else "")

    override_ids, note = None, None
    if args.frames:
        if args.frames == "all":
            override_ids = list(range(10**6))  # clipped to n_scene_points inside convert_one
        else:
            a, b = (int(x) for x in args.frames.split(":"))
            override_ids = list(range(a, b))
        note = ("stride-1 re-extraction via --frames; for val/test this includes frames the "
                "official split held out. Probe / visualisation only -- do NOT put this "
                "directory into a training mixture.")

    print(f"{len(names)} keyframes -> {out_root}\n", flush=True)
    t0 = time.time()
    for i, name in enumerate(names, 1):
        if name not in split_map:
            print(f"[{i}/{len(names)}] SKIP unknown: {name}", flush=True)
            continue
        split, ids = split_map[name]
        if args.skip_existing and osp.isfile(
                osp.join(out_root, args.out_split or split, name, "meta.json")):
            print(f"[{i}/{len(names)}] SKIP done: {name}", flush=True)
            continue
        t1 = time.time()
        m = convert_one(args.scared_dir, name, split, ids, out_root, not args.no_video,
                        override_ids=override_ids, out_split=args.out_split, note=note,
                        no_depth=args.no_depth)
        print(f"[{i}/{len(names)}] {split:5s} {name:22s} src={m['source']:22s} "
              f"{m['stored_frames']:5d}/{m['source_frames']:5d} 幀  "
              f"contiguous={str(m['contiguous']):5s} clipped_px={m['depth_clipped_px']:8d}  "
              f"{time.time()-t1:5.1f}s", flush=True)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
