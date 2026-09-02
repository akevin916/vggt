"""Run MonST3R on the private lesion set and dump point clouds for visual comparison.

MonST3R is a two-stage pipeline: pairwise feed-forward followed by a global
alignment optimisation (``GlobalAlignerMode.PointCloudOptimizer``). Each
sequence takes on the order of 1-3 minutes depending on frame count and
iteration budget -- this script is not fast, but it is correct.

PSNR is *not* computed here for two reasons:
  1. MonST3R is pretrained only (not fine-tuned on endoscopic data), so the
     number would not be a valid baseline -- see the baseline policy in
     eval_lesion.py.
  2. The pair-PSNR formulation (warp L→R) is defined w.r.t. VGGT's depth
     scale; MonST3R uses a different scale from its own global optimisation,
     making the numbers apples-to-oranges without additional alignment.

The output cloud format matches eval_lesion.py exactly (PLY + NPZ with
per-point frame_id), so the same timeline.py script can visualise all three
models side by side.

Usage::

    python -m pipeline.benchmark.eval_monst3r_lesion
    python -m pipeline.benchmark.eval_monst3r_lesion --folders 病灶3 --niter 200
"""

import argparse
import os
import sys
import traceback

import numpy as np
import torch

# MonST3R lives in reference/monst3r; add it to the path so that
# ``dust3r.*`` imports resolve without installing the package.
MONST3R_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reference", "monst3r",
)
sys.path.insert(0, MONST3R_ROOT)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images

from pipeline.data.paths import data_path
from pipeline.eval.media_io import write_video
from pipeline.eval.paths import default_output_dir
from pipeline.eval.ply_io import write_ply
from pipeline.eval.seq_io import dump_seq_npz

TOOL = "eval_lesion"          # same bucket as VGGT eval so outputs sit together
MODEL_SUBDIR = "MonST3R"

DEFAULT_CKPT = os.path.join(
    MONST3R_ROOT, "checkpoints",
    "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def left_frames(folder_dir):
    return sorted(f for f in os.listdir(folder_dir) if f.endswith("_L.png"))


def build_K(focal, pp, device="cpu"):
    """Build a (3,3) K matrix from MonST3R's focal scalar and [cx,cy] pp."""
    f = float(focal)
    cx, cy = float(pp[0]), float(pp[1])
    return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float32,
                        device=device)


def c2w_to_extrinsic(c2w_4x4):
    """Convert camera-to-world (4,4) to world-to-camera [R|t] (3,4)."""
    R = c2w_4x4[:3, :3]
    t = c2w_4x4[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return torch.cat([R_inv, t_inv[:, None]], dim=1)          # (3,4)


def save_cloud(out_stem, depth_list, intrinsic_list, extrinsic_list,
               images_np, max_points, depth_pct):
    """Unproject MonST3R depthmaps to world-frame point cloud.

    ``depth_list``    : list of (H,W) numpy float32 arrays (z-depth in cam frame)
    ``intrinsic_list``: list of (3,3) torch tensors
    ``extrinsic_list``: list of (3,4) torch tensors  [world-to-cam]
    ``images_np``     : (N,H,W,3) float32 in [0,1]
    """
    from pipeline.eval.warp_psnr import unproject_to_world   # same as eval_lesion

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
        pts.append(wk)
        cols.append(ck)
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


# ---------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------

def load_model(ckpt, device):
    model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(device).eval()
    print(f"loaded MonST3R from {ckpt}")
    return model


# MonST3R's video settings, copied from its own demo.py (the defaults passed at
# reference/monst3r/demo.py:424-447). These are NOT the optimizer's defaults: without them
# flow_loss_weight, temporal_smoothing_weight and use_self_mask are all off, the total loss
# in cloud_opt/optimizer.py collapses to DUSt3R's pairwise reprojection term, and what runs
# is DUSt3R with MonST3R weights -- every part of the method that handles video is disabled.
# Measured consequence of running it that way on the lesion clips: MonST3R's camera
# trajectory came out 4-18x longer relative to scene depth than VGGT's, i.e. the trajectory
# drifted, which is exactly what the flow and smoothing terms exist to prevent.
MONST3R_VIDEO_OPTS = dict(
    shared_focal=True,
    temporal_smoothing_weight=0.01,
    translation_weight=1.0,
    flow_loss_weight=0.01,
    flow_loss_start_epoch=0.1,
    flow_loss_thre=25,
    use_self_mask=True,
    # Two deviations from demo.py, both forced by what is on disk in reference/monst3r:
    #   sintel_ckpt=True  -> RAFT-sintel, because the SEA-RAFT weights the flow loss reaches
    #                        for by default (Tartan-C-T-TSKH-spring540x960-M.pth) are absent.
    #   sam2_mask_refine=False -> the SAM2 refinement of the motion mask needs
    #                        third_party/sam2/checkpoints/sam2.1_hiera_large.pt, also absent.
    sintel_ckpt=True,
    sam2_mask_refine=False,
    # batchify=False is a third deviation, forced by this card. Batchified alignment warps
    # the ego-flow for every pair in one go; at 60 square 512x512 frames that OOMs a 32 GB
    # 5090 inside warp_by_disp (measured). The per-pair loop computes the same loss with a
    # far smaller peak, at the cost of speed. empty_cache follows demo.py's intent (it turns
    # this on past 72 frames) but is on unconditionally here for the same headroom reason.
    batchify=False,
    empty_cache=True,
)


def run_monst3r_sequence(model, img_paths, device, niter, schedule, lr,
                         scene_graph="swinstride-5-noncyclic", video_opts=True):
    """Pairwise inference + global alignment → returns the optimised scene.

    ``scene_graph`` matters once the folder stops being a slideshow: "complete" is
    O(N^2) pairs, fine for the 12-19 shipped frames but 4032 pairs at 64, which neither
    fits nor finishes. "swinstride-5-noncyclic" is the sliding window demo.py builds for
    video, and is the default here for the same reason.

    ``video_opts=False`` restores the bare DUSt3R-style alignment, kept only so the old
    numbers can be reproduced.
    """
    imgs = load_images(img_paths, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    opts = dict(MONST3R_VIDEO_OPTS, num_total_iter=niter) if video_opts else {}
    scene = global_aligner(
        output, device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False,
        **opts,
    )
    scene.compute_global_alignment(init="mst", niter=niter,
                                   schedule=schedule, lr=lr)
    return scene, imgs


def run_monst3r_pair(model, lpath, rpath, device):
    """2-frame inference (L, R stereo pair). PairViewer mode -- no optimization."""
    imgs = load_images([lpath, rpath], size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    scene = global_aligner(
        output, device=device,
        mode=GlobalAlignerMode.PairViewer,
        verbose=False,
    )
    return scene, imgs


def scene_to_arrays(scene, imgs_raw, device):
    """Extract depth, intrinsics, extrinsics and RGB from an optimised scene."""
    depths    = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    focals    = scene.get_focals().detach().cpu()          # (N,)
    pps       = scene.get_principal_points().detach().cpu()  # (N, 2)
    c2w_poses = scene.get_im_poses().detach().cpu()        # (N, 4, 4)

    Ks = [build_K(focals[i], pps[i]) for i in range(len(depths))]
    Es = [c2w_to_extrinsic(c2w_poses[i]) for i in range(len(depths))]

    # MonST3R preprocesses to 512; grab the RGB at that resolution
    # load_images returns img["img"] with shape (1, 3, H, W) -- squeeze batch dim
    imgs_np = np.stack([img["img"].squeeze(0).permute(1, 2, 0).numpy()
                        for img in imgs_raw], axis=0)
    imgs_np = (imgs_np * 0.5 + 0.5).clip(0, 1).astype(np.float32)   # un-normalise

    return depths, Ks, Es, imgs_np


# ---------------------------------------------------------------------------
# per-folder runners
# ---------------------------------------------------------------------------

def run_pair_folder(model, folder_dir, out_dir, args):
    """Run MonST3R in 2-frame mode for each (L,R) pair in the folder."""
    from tqdm import tqdm
    names = left_frames(folder_dir)
    ply_dir = os.path.join(out_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    n_ok = 0
    for n in tqdm(names, desc=f"pair {os.path.basename(folder_dir)}", leave=False):
        lp = os.path.join(folder_dir, n)
        rp = lp.replace("_L.png", "_R.png")
        if not os.path.exists(rp):
            continue
        stem = n[:-6]  # strip "_L.png"
        try:
            scene, imgs_raw = run_monst3r_pair(model, lp, rp, args.device)
            depths, Ks, Es, imgs_np = scene_to_arrays(scene, imgs_raw, args.device)
            save_cloud(os.path.join(ply_dir, stem), depths, Ks, Es,
                       imgs_np, args.max_points, args.depth_pct)
            n_ok += 1
        except Exception:
            traceback.print_exc()
    print(f"    {n_ok}/{len(names)} pairs -> {ply_dir}")
    return dict(n_pairs=len(names), n_ok=n_ok, ply_dir=ply_dir)


def run_folder(model, folder_dir, out_dir, args):
    names  = left_frames(folder_dir)
    paths  = [os.path.join(folder_dir, n) for n in names]

    print(f"  {os.path.basename(folder_dir)}: {len(paths)} frames, "
          f"niter={args.niter}, scene_graph={args.scene_graph}")

    scene, imgs_raw = run_monst3r_sequence(
        model, paths, args.device, args.niter, args.schedule, args.lr,
        scene_graph=args.scene_graph, video_opts=not args.no_video_opts)
    depths, Ks, Es, imgs_np = scene_to_arrays(scene, imgs_raw, args.device)

    os.makedirs(out_dir, exist_ok=True)
    write_video(os.path.join(out_dir, "input.mp4"), imgs_np, fps=args.fps)
    # seq.npz as well as the cloud: without it this arm cannot be drawn by the polished
    # renderer that the VGGT arms use, and a 2x2 comparison would be mixing two renderers.
    dump_seq_npz(os.path.join(out_dir, "seq.npz"), depths, Ks, Es, imgs_np,
                 frames=np.array(names))

    info = dict(n_frames=len(paths), niter=args.niter)
    stem = os.path.join(out_dir, "cloud")
    n_pts = save_cloud(stem, depths, Ks, Es, imgs_np,
                       args.max_points, args.depth_pct)
    info["ply_points"] = n_pts
    info["ply"]        = stem + ".ply"
    print(f"    -> {n_pts:,} points  {stem}.ply")
    return info


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--lesion_root", default=data_path("eval", "lesion"))
    ap.add_argument("--folders", nargs="*", default=None)
    ap.add_argument("--modes", nargs="+", default=["seq"],
                    choices=["seq", "pair"],
                    help="seq: sequence mode; pair: per-(L,R) 2-frame mode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--niter",    type=int,   default=300,
                    help="global alignment optimisation steps (seq mode only)")
    ap.add_argument("--schedule", default="linear",
                    choices=["cosine", "linear"])
    ap.add_argument("--lr",       type=float, default=0.01)
    ap.add_argument("--scene_graph", default="swinstride-5-noncyclic",
                    help="pair graph for seq mode; demo.py's video default. 'complete' is "
                         "O(N^2) and only viable for the shipped ~15-frame folders")
    ap.add_argument("--no_video_opts", action="store_true",
                    help="run the bare DUSt3R-style alignment instead of MonST3R's video "
                         "settings -- only to reproduce the old numbers")
    ap.add_argument("--max_points", type=int, default=8_000_000)
    ap.add_argument("--fps",      type=float, default=4.0)
    ap.add_argument("--depth_pct", type=float, default=99.0)
    ap.add_argument("--out_dir",  default=None)
    args = ap.parse_args()

    folders = args.folders or sorted(
        d for d in os.listdir(args.lesion_root)
        if os.path.isdir(os.path.join(args.lesion_root, d))
    )
    print(f"MonST3R  |  {len(folders)} folders  |  modes={args.modes}")

    model  = load_model(args.ckpt, args.device)
    repo_root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    results = {}

    for folder in folders:
        folder_dir = os.path.join(args.lesion_root, folder)
        res = {}
        for mode in args.modes:
            if args.out_dir:
                base = os.path.join(args.out_dir, mode)
            else:
                base = os.path.join(repo_root, "outputs", TOOL, MODEL_SUBDIR, mode)
            out_dir = os.path.join(base, folder)
            try:
                if mode == "seq":
                    res[mode] = run_folder(model, folder_dir, out_dir, args)
                else:
                    res[mode] = run_pair_folder(model, folder_dir, out_dir, args)
            except Exception:
                traceback.print_exc()
                res[mode] = dict(error=True)
        results[folder] = res

    import json
    summary_base = args.out_dir or os.path.join(
        repo_root, "outputs", TOOL, MODEL_SUBDIR)
    os.makedirs(summary_base, exist_ok=True)
    summary_path = os.path.join(summary_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(ckpt=args.ckpt, folders=folders, modes=args.modes,
                       results=results), f, indent=2, ensure_ascii=False)
    print(f"\n-> {summary_path}")


if __name__ == "__main__":
    main()
