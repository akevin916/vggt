"""Reconstruct the private lesion set and score the stereo pairs by warped PSNR.

Three runs per checkpoint, over each of the lesion folders:

  * ``pair``  -- every (L, R) pair as its own 2-frame forward. The right view is
    synthesised from the left image and compared against the real right image; this is
    the number that reproduces the previous phase's pair-only setting.
  * ``seq``   -- the folder's left images as one sequence, every frame.
  * ``seq2``  -- the same sequence at stride 2, to show what the model does with half
    the temporal density.

PSNR is reported for ``pair`` only. The sequence runs exist for the reconstruction
figures, so they dump arrays and point clouds and no metric.

The set ships **no calibration** -- no intrinsics, no baseline, no ground-truth depth or
pose. Everything here therefore comes from the model's own camera head, and the PSNR is a
self-consistency measure of depth against pose, not metric accuracy. Report it as such.

WHO TO COMPARE AGAINST (decided 2026-08-19, not yet available)
--------------------------------------------------------------
The comparison must be against baselines **fine-tuned on this domain**:

  1. VGGT-1B fine-tuned on endoscopic data   -- not yet trained
  2. MonST3R fine-tuned on endoscopic data   -- not yet trained

Pretrained VGGT-1B is NOT a valid comparison point and its numbers are held back: it has
never seen an endoscope, so any gap it shows is domain gap, not method. Running it only
answers "does fine-tuning help", which nobody is asking. It stays in ``--ckpts`` as a
sanity reference (does the pipeline produce sane geometry at all), and any table built
from it must be labelled as such -- never as the baseline row.

Everything lands in ``outputs/eval_lesion/<exp>/``, one subdirectory per mode and folder,
with the raw float arrays alongside the figures' inputs -- redrawing must never need the
GPU again.

Usage:
  python benchmark/eval_lesion.py \
      --ckpts checkpoints/VGGT-1B.pt logs/scared_cam_b16/ckpts/best_ate.pt
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.paths import data_path
from eval_utils.media_io import write_video
from eval_utils.paths import default_output_dir
from eval_utils.ply_io import write_ply
from eval_utils.vggt_infer import infer_sequence, load_vggt_for_eval
from eval_utils.warp_psnr import score_pair, unproject_to_world
from vggt.utils.load_fn import load_and_preprocess_images

TOOL = "eval_lesion"


def list_folders(root):
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]


def left_frames(folder_dir):
    return sorted(f for f in os.listdir(folder_dir) if f.endswith("_L.png"))


def preprocessed(paths):
    """The exact tensor the model saw, as [S,H,W,3] float in [0,1]. PSNR is computed at
    this resolution -- comparing against the original 512x512 would fold a resize into
    the metric."""
    t = load_and_preprocess_images(paths, mode="crop")
    return t.permute(0, 2, 3, 1).numpy().astype(np.float32)


def save_cloud(out_stem, depth, intrinsic, extrinsic, images, max_points, depth_pct):
    """Point cloud by unprojecting the predicted depth -- not the point head, which the
    fine-tuned checkpoints do not carry. Same arrays the depth figures use, so the cloud
    and the depth maps can never disagree.

    Writes two files. The ``.ply`` is for a viewer. The ``.npz`` additionally carries a
    per-point ``frame_id``, which is what makes the timeline visualisation possible: "the
    reconstruction at time t" is the slice ``frame_id <= t``, so no re-inference is ever
    needed to scrub through it.

    The budget is therefore spent **per frame** rather than by sampling the merged cloud.
    A global random subsample would be cheaper and would destroy exactly the property the
    timeline needs -- frames would end up unevenly represented, and the reconstruction
    would appear to grow in fits and starts that the data does not contain.
    """
    per_frame = max(1, max_points // max(1, len(depth)))
    rng = np.random.default_rng(0)
    pts, cols, fids = [], [], []
    for i in range(len(depth)):
        d = depth[i]
        w = unproject_to_world(d, intrinsic[i], extrinsic[i]).reshape(-1, 3)
        c = images[i].reshape(-1, 3)             # write_ply wants [0,1] and scales itself
        keep = (d.reshape(-1) > 0) & np.isfinite(w).all(axis=1)
        if depth_pct < 100 and (d > 0).any():
            keep &= d.reshape(-1) <= np.percentile(d[d > 0], depth_pct)
        wk, ck = w[keep], c[keep]
        if len(wk) > per_frame:
            sel = rng.choice(len(wk), per_frame, replace=False)
            wk, ck = wk[sel], ck[sel]
        pts.append(wk)
        cols.append(ck)
        fids.append(np.full(len(wk), i, np.int16))

    pts = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
    cols = np.concatenate(cols) if cols else np.zeros((0, 3), np.float32)
    fids = np.concatenate(fids) if fids else np.zeros((0,), np.int16)

    np.savez_compressed(out_stem + ".npz", points=pts.astype(np.float32),
                        colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
                        frame_id=fids, n_frames=np.int32(len(depth)))
    return write_ply(out_stem + ".ply", pts, cols)


def dump_arrays(path, pred, extra=None):
    payload = {k: v for k, v in pred.items() if isinstance(v, np.ndarray)}
    if "depth" in payload:                       # fp16 halves the dump; PSNR is recomputed
        payload["depth"] = payload["depth"].astype(np.float16)   # from pose+intrinsic anyway
    payload.update(extra or {})
    np.savez_compressed(path, **payload)


def run_pair(model, folder_dir, out_dir, args):
    names = left_frames(folder_dir)
    rows, strips = [], []
    warped_dir = os.path.join(out_dir, "warped")
    ply_dir = os.path.join(out_dir, "ply")
    os.makedirs(warped_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    for n in tqdm(names, desc=f"pair {os.path.basename(folder_dir)}", leave=False):
        lp = os.path.join(folder_dir, n)
        rp = lp.replace("_L.png", "_R.png")
        if not os.path.exists(rp):
            continue
        pred = infer_sequence(model, [lp, rp], device=args.device)
        if "depth" not in pred:
            raise SystemExit("checkpoint has no depth head; warped PSNR needs depth")
        imgs = preprocessed([lp, rp])

        # Synthesise the RIGHT view (index 1) by sampling the LEFT image (index 0),
        # using the right view's own depth -- an inverse warp, hole-free by construction.
        res = score_pair(
            img_src=imgs[0], img_dst=imgs[1], depth_dst=pred["depth"][1],
            K_src=pred["intrinsic"][0], K_dst=pred["intrinsic"][1],
            E_src=pred["extrinsic"][0], E_dst=pred["extrinsic"][1],
            device=args.device,
        )
        stem = n[:-6]
        Image.fromarray((np.clip(res["warped"], 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(warped_dir, f"{stem}_warpedR.png"))
        dump_arrays(os.path.join(out_dir, f"{stem}.npz"), pred,
                    extra=dict(valid=res["valid"]))
        if args.ply:
            # A pair reconstructs too -- two views, one cloud. Without this the pair and
            # sequence runs would not be comparable as *reconstructions*, only as numbers.
            save_cloud(os.path.join(ply_dir, stem), pred["depth"], pred["intrinsic"],
                       pred["extrinsic"], imgs, args.max_points, args.depth_pct)
        rows.append(dict(pair=stem, psnr=res["psnr"],
                         psnr_no_specular=res["psnr_no_specular"],
                         valid_frac=res["valid_frac"],
                         specular_frac=res["specular_frac"]))
        # [left | real right | synthesised right] -- the PSNR made visible.
        strips.append(np.concatenate(
            [imgs[0], imgs[1], np.clip(res["warped"], 0, 1)], axis=1))

    if strips:
        write_video(os.path.join(out_dir, "pairs_LRwarp.mp4"), strips, fps=args.fps)

    finite = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    return dict(per_pair=rows,
                psnr_mean=float(np.mean(finite)) if finite else float("nan"),
                psnr_no_specular_mean=float(np.mean(
                    [r["psnr_no_specular"] for r in rows
                     if np.isfinite(r["psnr_no_specular"])])) if rows else float("nan"),
                n_pairs=len(rows))


def run_sequence(model, folder_dir, out_dir, args, stride):
    names = left_frames(folder_dir)[::stride]
    paths = [os.path.join(folder_dir, n) for n in names]
    pred = infer_sequence(model, paths, device=args.device)
    imgs = preprocessed(paths)
    dump_arrays(os.path.join(out_dir, "seq.npz"), pred,
                extra=dict(frames=np.array(names), images=(imgs * 255).astype(np.uint8)))
    # The clip lives next to the cloud: a point cloud with no footage beside it cannot be
    # judged by anyone who was not there when it was made.
    write_video(os.path.join(out_dir, "input.mp4"), imgs, fps=args.fps)

    info = dict(n_frames=len(paths), stride=stride, frames=names)
    if args.ply and "depth" in pred:
        stem = os.path.join(out_dir, "cloud")
        info["ply_points"] = save_cloud(stem, pred["depth"], pred["intrinsic"],
                                        pred["extrinsic"], imgs,
                                        args.max_points, args.depth_pct)
        info["ply"] = stem + ".ply"
    return info


def eval_ckpt(ckpt, args):
    model = load_vggt_for_eval(ckpt, device=args.device)
    base = args.out_dir or default_output_dir(ckpt, TOOL)
    results = {}

    for folder in args.folders:
        folder_dir = os.path.join(args.lesion_root, folder)
        res = {}
        for mode in args.modes:
            out_dir = os.path.join(base, mode, folder)
            os.makedirs(out_dir, exist_ok=True)
            try:
                if mode == "pair":
                    res[mode] = run_pair(model, folder_dir, out_dir, args)
                else:
                    res[mode] = run_sequence(model, folder_dir, out_dir, args,
                                             stride=2 if mode == "seq2" else 1)
            except Exception:
                traceback.print_exc()
                res[mode] = dict(error=True)
        results[folder] = res

    del model
    torch.cuda.empty_cache()

    payload = dict(meta=dict(ckpt=ckpt, lesion_root=args.lesion_root,
                             folders=args.folders, modes=args.modes,
                             note="no calibration in this set; PSNR is depth-vs-pose "
                                  "self-consistency, not metric accuracy"),
                   results=results)
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"-> {os.path.join(base, 'summary.json')}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        "checkpoints/VGGT-1B.pt",
        "logs/scared_cam_b16/ckpts/best_ate.pt",
    ])
    ap.add_argument("--lesion_root", default=data_path("eval", "lesion"))
    ap.add_argument("--folders", nargs="*", default=None)
    ap.add_argument("--modes", nargs="+", default=["pair", "seq", "seq2"],
                    choices=["pair", "seq", "seq2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ply", action="store_true", default=True)
    ap.add_argument("--no_ply", dest="ply", action="store_false")
    ap.add_argument("--max_points", type=int, default=2_000_000)
    ap.add_argument("--fps", type=float, default=4.0, help="playback fps for the dumped clips")
    ap.add_argument("--depth_pct", type=float, default=99.0,
                    help="drop points beyond this depth percentile (far-field noise)")
    ap.add_argument("--out_dir", default=None, help="override; otherwise outputs/<tool>/<exp>/")
    args = ap.parse_args()

    args.folders = args.folders or list_folders(args.lesion_root)
    print(f"{len(args.ckpts)} ckpts x {len(args.folders)} folders x {args.modes}")

    table = {}
    for ckpt in args.ckpts:
        print(f"\n=== {ckpt}")
        table[ckpt] = eval_ckpt(ckpt, args)

    if "pair" in args.modes:
        print(f"\n{'ckpt':<44}{'folder':<12}{'PSNR':>9}{'PSNR(-spec)':>13}{'n':>5}")
        for ckpt, payload in table.items():
            for folder, res in payload["results"].items():
                d = res.get("pair", {})
                if "psnr_mean" not in d:
                    continue
                print(f"{os.path.basename(ckpt):<44}{folder:<12}"
                      f"{d['psnr_mean']:9.2f}{d['psnr_no_specular_mean']:13.2f}"
                      f"{d['n_pairs']:5d}")


if __name__ == "__main__":
    main()
