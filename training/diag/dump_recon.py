#!/usr/bin/env python3
"""Reconstruct image folders and dump them in the ``seq.npz`` format.

benchmark/eval_{lesion,gastric}.py dump the arrays their figures need; eval_scared.py does
not -- it only writes metrics json -- so there is nothing on disk for the visualisation
tools to draw. This is the missing dump step: pick a window of frames, run one forward
pass, and write the same ``seq.npz`` (depth / intrinsic / extrinsic / images) that
diag/vis/cloud_polish.py and the timeline already know how to read.

Two sources:

  ``--ckpt``     one forward pass per job. No chunking: a stitched sequence has seams and
                 its geometry would not fuse (see vggt_infer.infer_sequence_chunked), so
                 keep the window small enough to fit instead. Several jobs share one
                 loaded model -- the 9 GB checkpoint is read once, not once per job.
  ``--gt``       SCARED ground-truth depth + extrinsics straight off disk. No GPU, no
                 model. This is the ceiling: whatever the GT cloud cannot show, no
                 checkpoint can be blamed for missing. Note GT depth is only ~50% dense,
                 so the reference is legitimately holey.
  ``--monst3r``  MonST3R pairwise inference + global alignment, via the runner in
                 benchmark/eval_monst3r_lesion.py. Exists because the two MonST3R
                 benchmark scripts are bound to the lesion/gastric folder layouts and
                 there is no way to point either at a SCARED sequence -- but a comparison
                 figure needs all arms on the same frames, MonST3R included.

Jobs are ``--job image_dir,start,count,stride,exp`` and may be repeated.

Usage (from training/):
  python diag/dump_recon.py --ckpt logs/scared_cam_b16/ckpts/best_ate.pt \
      --job ../data/train/scared/val/dataset2/keyframe3/image_left,0,48,2,scared_ds2kf3_scared_cam_b16 \
      --job ../data/train/scared/val/dataset3/keyframe3/image_left,0,48,2,scared_ds3kf3_scared_cam_b16

  python diag/dump_recon.py --gt \
      --job ../data/train/scared/val/dataset5/keyframe3,0,48,2,scared_ds5kf3_GT
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_utils.media_io import write_video
from eval_utils.paths import output_dir_for_exp
from vggt.utils.load_fn import load_and_preprocess_images

TOOL = "dump_recon"

DEPTH_SCALE = 100.0        # SCARED uint16 counts per mm (benchmark/eval_scared.py)
TARGET_W = 518


def parse_job(s: str):
    parts = s.split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--job wants image_dir,start,count,stride,exp -- got {s!r}")
    d, start, count, stride, exp = parts
    return dict(dir=d, start=int(start), count=int(count), stride=int(stride), exp=exp)


def select_frames(image_dir, job):
    names = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return names[job["start"]::job["stride"]][:job["count"]]


def preprocessed(paths):
    """The exact tensor the model sees, as [S,H,W,3] float in [0,1]."""
    t = load_and_preprocess_images(paths, mode="crop")
    return t.permute(0, 2, 3, 1).numpy().astype(np.float32)


# ---------------------------------------------------------------------------

def run_model_job(model, job, args):
    from eval_utils.vggt_infer import infer_sequence

    names = select_frames(job["dir"], job)
    if not names:
        raise SystemExit(f"no frames selected for {job['exp']}")
    paths = [os.path.join(job["dir"], n) for n in names]
    print(f"  {len(paths)} frames: {names[0]} .. {names[-1]} (stride {job['stride']})")

    pred = infer_sequence(model, paths, device=args.device)
    if "depth" not in pred:
        raise SystemExit("checkpoint has no depth head; nothing to reconstruct")
    payload = {k: v for k, v in pred.items() if isinstance(v, np.ndarray)}
    return payload, preprocessed(paths), names


def run_monst3r_job(model, job, args):
    """MonST3R on an arbitrary image folder, in the same seq.npz format as the rest.

    Its depth/K/E come out of the global alignment in MonST3R's own 512-px preprocessing,
    not the 518 crop the VGGT arms use -- which is fine, because each panel is rendered
    from its own seq.npz and the two are never mixed inside one cloud.
    """
    from benchmark.eval_monst3r_lesion import run_monst3r_sequence, scene_to_arrays

    names = select_frames(job["dir"], job)
    if not names:
        raise SystemExit(f"no frames selected for {job['exp']}")
    paths = [os.path.join(job["dir"], n) for n in names]
    print(f"  {len(paths)} frames: {names[0]} .. {names[-1]} (stride {job['stride']}), "
          f"niter={args.niter}, scene_graph={args.scene_graph}")

    scene, imgs_raw = run_monst3r_sequence(model, paths, args.device, args.niter,
                                           args.schedule, args.lr,
                                           scene_graph=args.scene_graph,
                                           video_opts=not args.no_video_opts)
    depths, Ks, Es, imgs_np = scene_to_arrays(scene, imgs_raw, args.device)
    payload = dict(depth=np.stack(depths).astype(np.float32),
                   intrinsic=np.stack([K.cpu().numpy() for K in Ks]).astype(np.float32),
                   extrinsic=np.stack([E.cpu().numpy() for E in Es]).astype(np.float32))
    return payload, imgs_np, names


def run_gt_job(job, args):
    """SCARED GT: ``<seq_dir>/{image_left,depth_left,cam_data}``.

    GT depth lives at the original 1280x1024 with the calibration's KL; the model's own
    dumps live on the 518-wide preprocessed grid. Both are resampled onto that same grid
    here so a GT cloud and a predicted cloud are directly comparable -- and so the
    intrinsic that goes into the npz is the one that matches the stored depth.
    """
    seq_dir = job["dir"]
    img_dir = os.path.join(seq_dir, "image_left")
    names = select_frames(img_dir, job)
    paths = [os.path.join(img_dir, n) for n in names]
    imgs = preprocessed(paths)                          # [S,H,W,3]
    H, W = imgs.shape[1:3]
    print(f"  {len(paths)} frames: {names[0]} .. {names[-1]} (stride {job['stride']}) -> {H}x{W}")

    with open(os.path.join(seq_dir, "cam_data", "calibration.json")) as f:
        K0 = np.array(json.load(f)["KL"], dtype=np.float64)
    fids = np.loadtxt(os.path.join(seq_dir, "cam_data", "frames.txt"), dtype=np.int64, ndmin=1)
    E_all = np.loadtxt(os.path.join(seq_dir, "cam_data", "extrinsics.txt")).reshape(-1, 3, 4)
    pos = {f"{int(f):06d}.png": i for i, f in enumerate(fids)}

    depths, extr = [], []
    for n in names:
        d = cv2.imread(os.path.join(seq_dir, "depth_left", n), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise SystemExit(f"missing GT depth for {n}")
        oh, ow = d.shape[:2]
        # nearest, not linear: interpolating across a depth edge invents surface that is
        # not there, which is exactly the artefact the cull stage exists to remove.
        depths.append(cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                      / DEPTH_SCALE)
        extr.append(E_all[pos[n]])

    sx, sy = W / ow, H / oh
    K = K0.copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    depth = np.stack(depths)
    valid = float((depth > 0).mean())
    print(f"  GT depth valid fraction {valid:.2f}")
    payload = dict(depth=depth,
                   intrinsic=np.repeat(K[None].astype(np.float32), len(names), 0),
                   extrinsic=np.stack(extr).astype(np.float32),
                   gt_valid_frac=np.float32(valid))
    return payload, imgs, names


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", type=parse_job, action="append", required=True,
                    help="image_dir,start,count,stride,exp (repeatable)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gt", action="store_true", help="SCARED ground truth instead of a model")
    ap.add_argument("--monst3r", action="store_true", help="MonST3R instead of a VGGT ckpt")
    ap.add_argument("--monst3r_ckpt", default=None,
                    help="default: whatever benchmark/eval_monst3r_lesion.py uses")
    ap.add_argument("--niter", type=int, default=300, help="--monst3r only")
    ap.add_argument("--schedule", default="linear", choices=["cosine", "linear"])
    ap.add_argument("--lr", type=float, default=0.01, help="--monst3r only")
    ap.add_argument("--scene_graph", default="swinstride-5-noncyclic",
                    help="--monst3r only: demo.py's video default. 'complete' is O(N^2) "
                         "pairs and will not fit a 64-frame window")
    ap.add_argument("--no_video_opts", action="store_true",
                    help="--monst3r only: bare DUSt3R-style alignment instead of MonST3R's "
                         "video settings")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if sum([bool(args.ckpt), bool(args.gt), bool(args.monst3r)]) != 1:
        raise SystemExit("pass exactly one of --ckpt / --gt / --monst3r")

    model = None
    if args.ckpt:
        from eval_utils.vggt_infer import load_vggt_for_eval
        model = load_vggt_for_eval(args.ckpt, device=args.device)
    elif args.monst3r:
        from benchmark.eval_monst3r_lesion import DEFAULT_CKPT, load_model
        model = load_model(args.monst3r_ckpt or DEFAULT_CKPT, args.device)

    for job in args.job:
        print(f"[{job['exp']}]")
        try:
            if args.gt:
                payload, imgs, names = run_gt_job(job, args)
            elif args.monst3r:
                payload, imgs, names = run_monst3r_job(model, job, args)
            else:
                payload, imgs, names = run_model_job(model, job, args)
        except Exception as e:                       # one bad job must not lose the batch
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

        out_dir = output_dir_for_exp(job["exp"], TOOL)
        os.makedirs(out_dir, exist_ok=True)
        payload["depth"] = payload["depth"].astype(np.float16)
        payload["images"] = (np.clip(imgs, 0, 1) * 255).astype(np.uint8)
        payload["frames"] = np.array(names)
        np.savez_compressed(os.path.join(out_dir, "seq.npz"), **payload)
        write_video(os.path.join(out_dir, "input.mp4"), imgs, fps=args.fps)
        source = "gt" if args.gt else ("monst3r" if args.monst3r else args.ckpt)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(dict(source=source, **job, frames=names), f, indent=2)
        print(f"  -> {out_dir}/seq.npz")


if __name__ == "__main__":
    main()
