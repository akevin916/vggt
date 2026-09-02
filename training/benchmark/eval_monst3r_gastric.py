"""Run MonST3R on the private gastric sequences and dump point clouds.

Usage::

    cd training
    python benchmark/eval_monst3r_gastric.py
    python benchmark/eval_monst3r_gastric.py --max_frames 30 --niter 200
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

MONST3R_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reference", "monst3r",
)
sys.path.insert(0, MONST3R_ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images

from data.paths import data_path
from eval_utils.media_io import write_video
from eval_utils.ply_io import write_ply
from eval_utils.seq_io import dump_seq_npz
from benchmark.eval_monst3r_lesion import MONST3R_VIDEO_OPTS

TOOL = "eval_gastric"
MODEL_SUBDIR = "MonST3R"

DEFAULT_CKPT = os.path.join(
    MONST3R_ROOT, "checkpoints",
    "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth",
)

DEFAULT_SEGS = [
    "seq_000189_000432",
    "seq_000984_001143",
    "seq_000606_000723",
]


# ---------------------------------------------------------------------------
# helpers (shared with eval_monst3r_lesion)
# ---------------------------------------------------------------------------

def build_K(focal, pp):
    import torch
    f = float(focal); cx, cy = float(pp[0]), float(pp[1])
    return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float32)


def c2w_to_extrinsic(c2w_4x4):
    import torch
    R = c2w_4x4[:3, :3]; t = c2w_4x4[:3, 3]
    R_inv = R.T; t_inv = -R_inv @ t
    return torch.cat([R_inv, t_inv[:, None]], dim=1)


def save_cloud(out_stem, depth_list, intrinsic_list, extrinsic_list,
               images_np, max_points, depth_pct):
    from eval_utils.warp_psnr import unproject_to_world
    per_frame = max(1, max_points // max(1, len(depth_list)))
    rng = np.random.default_rng(0)
    pts, cols, fids = [], [], []
    for i, (depth, K, E) in enumerate(zip(depth_list, intrinsic_list, extrinsic_list)):
        K_np = K.cpu().numpy().astype(np.float64)
        E_np = E.cpu().numpy().astype(np.float64)
        d = depth.astype(np.float64)
        w = unproject_to_world(d, K_np, E_np).reshape(-1, 3)
        c = images_np[i].reshape(-1, 3)
        keep = (d.reshape(-1) > 0) & np.isfinite(w).all(axis=1)
        if depth_pct < 100 and (d > 0).any():
            keep &= d.reshape(-1) <= np.percentile(d[d > 0], depth_pct)
        wk, ck = w[keep], c[keep]
        if len(wk) > per_frame:
            sel = rng.choice(len(wk), per_frame, replace=False)
            wk, ck = wk[sel], ck[sel]
        pts.append(wk); cols.append(ck)
        fids.append(np.full(len(wk), i, np.int16))
    pts  = np.concatenate(pts)  if pts  else np.zeros((0, 3), np.float32)
    cols = np.concatenate(cols) if cols else np.zeros((0, 3), np.float32)
    fids = np.concatenate(fids) if fids else np.zeros((0,),   np.int16)
    np.savez_compressed(
        out_stem + ".npz",
        points=pts.astype(np.float32),
        colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
        frame_id=fids,
        n_frames=np.int32(len(depth_list)),
    )
    return write_ply(out_stem + ".ply", pts, cols)


def scene_to_arrays(scene, imgs_raw):
    depths    = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    focals    = scene.get_focals().detach().cpu()
    pps       = scene.get_principal_points().detach().cpu()
    c2w_poses = scene.get_im_poses().detach().cpu()
    Ks = [build_K(focals[i], pps[i]) for i in range(len(depths))]
    Es = [c2w_to_extrinsic(c2w_poses[i]) for i in range(len(depths))]
    imgs_np = np.stack([img["img"].squeeze(0).permute(1, 2, 0).numpy()
                        for img in imgs_raw], axis=0)
    imgs_np = (imgs_np * 0.5 + 0.5).clip(0, 1).astype(np.float32)
    return depths, Ks, Es, imgs_np


# ---------------------------------------------------------------------------
# per-segment
# ---------------------------------------------------------------------------

def run_segment(model, seg_dir, out_dir, args):
    img_dir = os.path.join(seg_dir, "image")
    paths = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    )
    if not paths:
        return dict(error="no images found")
    if args.max_frames > 0:
        paths = paths[:args.max_frames]

    seg_name = os.path.basename(seg_dir)
    print(f"  {seg_name}: {len(paths)} frames, niter={args.niter}")

    imgs = load_images(paths, size=512, verbose=False)
    # Same story as the lesion runner: MonST3R's video settings have to be passed
    # explicitly or the optimizer runs plain DUSt3R. See MONST3R_VIDEO_OPTS there.
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=1, verbose=False)
    opts = ({} if args.no_video_opts else
            dict(MONST3R_VIDEO_OPTS, num_total_iter=args.niter))
    scene = global_aligner(
        output, device=args.device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False,
        **opts,
    )
    scene.compute_global_alignment(init="mst", niter=args.niter,
                                   schedule=args.schedule, lr=args.lr)

    depths, Ks, Es, imgs_np = scene_to_arrays(scene, imgs)

    os.makedirs(out_dir, exist_ok=True)
    write_video(os.path.join(out_dir, "input.mp4"), imgs_np, fps=args.fps)
    dump_seq_npz(os.path.join(out_dir, "seq.npz"), depths, Ks, Es, imgs_np,
                 frames=np.array([os.path.basename(p) for p in paths]))

    stem = os.path.join(out_dir, "cloud")
    n_pts = save_cloud(stem, depths, Ks, Es, imgs_np, args.max_points, args.depth_pct)
    print(f"    -> {n_pts:,} points  {stem}.ply")
    return dict(seg=seg_name, n_frames=len(paths), ply_points=n_pts)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--gastric_root", default=data_path("eval", "gastric"))
    ap.add_argument("--segs", nargs="*", default=None)
    ap.add_argument("--max_frames", type=int, default=30,
                    help="truncate each segment to the first N frames (0=all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--niter",    type=int,   default=300)
    ap.add_argument("--schedule", default="linear", choices=["cosine", "linear"])
    ap.add_argument("--scene_graph", default="swinstride-5-noncyclic")
    ap.add_argument("--no_video_opts", action="store_true",
                    help="bare DUSt3R-style alignment; only to reproduce the old numbers")
    ap.add_argument("--lr",       type=float, default=0.01)
    ap.add_argument("--max_points", type=int, default=8_000_000)
    ap.add_argument("--fps",      type=float, default=10.0)
    ap.add_argument("--depth_pct", type=float, default=99.0)
    ap.add_argument("--out_dir",  default=None)
    args = ap.parse_args()

    segs = args.segs or DEFAULT_SEGS
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    base = args.out_dir or os.path.join(repo_root, "outputs", TOOL, MODEL_SUBDIR)

    print(f"MonST3R gastric  |  {len(segs)} segments  |  max_frames={args.max_frames}")

    model = AsymmetricCroCo3DStereo.from_pretrained(args.ckpt).to(args.device).eval()
    print(f"loaded MonST3R from {args.ckpt}")

    results = {}
    for seg in segs:
        seg_dir = os.path.join(args.gastric_root, "prep", seg)
        if not os.path.isdir(seg_dir):
            print(f"  [skip] {seg_dir} not found")
            results[seg] = dict(error="not found")
            continue
        out_dir = os.path.join(base, seg)
        try:
            results[seg] = run_segment(model, seg_dir, out_dir, args)
        except Exception:
            traceback.print_exc()
            results[seg] = dict(error=True)

    os.makedirs(base, exist_ok=True)
    summary_path = os.path.join(base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(ckpt=args.ckpt, segs=segs, results=results),
                  f, indent=2, ensure_ascii=False)
    print(f"\n-> {summary_path}")


if __name__ == "__main__":
    main()
