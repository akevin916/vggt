#!/usr/bin/env python3
"""Does per-frame pose error grow with sequence position / distance travelled?

Tests the "long-range accumulated error" hypothesis directly: after a SINGLE Sim(3)
alignment over the whole (full-length, untruncated) trajectory, plot the per-frame
absolute translation error two ways:
  (a) vs frame index      -- classic drift signature: error should trend upward
  (b) vs cumulative GT distance travelled -- separates "later frame" from "more distance",
      since a fast-moving camera covers more distance per frame than a slow one.

If error is flat/noisy with no trend in either plot, the earlier jitter findings (weak/
borderline gate confidence, frame-to-frame instability present even in "good" gate
sequences) are the better explanation than classical long-range drift accumulation --
VGGT is feedforward over the whole clip (global attention across all frames at once,
not a sequential/recursive integrator), so it has no structural reason to accumulate
error the way frame-chained visual odometry does.

Uses FULL, untruncated Sintel sequences (unlike the 12/20-frame windows used in
vis_trajectory.py / vis_gate_temporal.py) so genuine long-range drift has room to show up.

--gate_modes lets you overlay the model's own ("predicted") gate against an "oracle" gate
built directly from the GT flow-residual mask (m_geo, §5.3a) -- i.e. swap in the m_geo mask
as the attention bias instead of the model's own (possibly low-confidence) gate, and see
whether the per-frame error spikes found with "predicted" shrink or move.
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_DIR = os.path.dirname(_TRAINING_DIR)
sys.path[:0] = [_TRAINING_DIR, _REPO_DIR]

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

from data.motion_mask import (
    compute_ego_flow,
    derive_motion_mask,
    load_sintel_gt_flows,
    sintel_masks_and_fraction,
)
from eval_utils.gate_common import oracle_logits_from_masks
from eval_utils.paths import ERROR_GROWTH, default_output_dir
from eval_utils.metrics_pose import _make_traj, extrinsics_w2c_to_tum
from data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    load_sintel_gt_depths,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    matching_cam_path,
    sintel_cam_read,
    sintel_seq_paths,
)
from eval_utils.vggt_infer import infer_sequence_chunked, load_vggt_for_eval
from vggt.utils.load_fn import load_and_preprocess_images

MODE_STYLE = {"predicted": ("r-o", "k--"), "oracle": ("g-s", "m--"), "off": ("c-^", "y--")}


@torch.no_grad()
def frame0_dissimilarity(model, images: torch.Tensor, device: str) -> np.ndarray:
    """Cosine distance of each frame's mean-pooled patch token vs frame 0's.

    Uses the aggregator's own patch_embed (same features the network itself sees
    before any cross-frame attention), as a proxy for "how differently does the
    network perceive this frame vs the frame-0 anchor" -- tests the hypothesis
    that pose error tracks appearance dissimilarity to frame 0 (VGGT's coordinate
    anchor, see training/train_utils/normalization.py first_cam_extrinsic_inv and
    aggregator.py's distinct frame-0 camera/register token) rather than distance
    travelled or frame index.
    """
    agg = model.aggregator if hasattr(model, "aggregator") else model
    imgs = images[None].to(device)
    B, S, C_in, H, W = imgs.shape
    x = (imgs - agg._resnet_mean) / agg._resnet_std
    x = x.view(B * S, C_in, H, W)
    patch_tokens = agg.patch_embed(x)
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]
    feat = patch_tokens.mean(dim=1)  # (B*S, C)
    feat = torch.nn.functional.normalize(feat, dim=-1)
    sim_to_frame0 = (feat @ feat[0]).cpu().numpy()
    return 1.0 - sim_to_frame0


def frame0_rgb_dissimilarity(images: torch.Tensor) -> np.ndarray:
    """Cosine distance of each frame's raw flattened RGB pixels vs frame 0's.

    Model-free control for frame0_dissimilarity: tests the same "how different does
    this frame look from the frame-0 anchor" question at the raw-pixel level, without
    going through any learned features, so it can't be confounded by what the network
    itself chooses to attend to.
    """
    flat = images.reshape(images.shape[0], -1)
    flat = torch.nn.functional.normalize(flat, dim=-1)
    sim_to_frame0 = (flat @ flat[0]).cpu().numpy()
    return 1.0 - sim_to_frame0


def frame_dynamic_fraction(sintel_root: str, seq: str, rgb_paths: list, cam_dir: str, motion_thr: float) -> np.ndarray:
    """Per-frame GT dynamic-pixel fraction (m_geo, §5.3a) -- direct proxy for the §4.2
    "all-dynamic-frame underdetermined" mechanism: how much of THIS frame is unusable for the
    camera token, independent of any comparison to frame 0. Last frame has no forward flow, so
    it's left as NaN (excluded from correlation/fit, but plotted as a gap).
    """
    gt_flows = load_sintel_gt_flows(sintel_root, seq, rgb_paths)
    gt_depths = load_sintel_gt_depths(sintel_root, seq, rgb_paths)
    intrinsics, extrinsics = [], []
    for p in rgb_paths:
        K, ext = sintel_cam_read(matching_cam_path(cam_dir, p))
        intrinsics.append(K)
        extrinsics.append(ext)

    fractions = np.full(len(rgb_paths), np.nan, dtype=np.float32)
    for i in range(len(rgb_paths)):
        if gt_flows[i] is None or i + 1 >= len(extrinsics):
            continue
        ego = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
        mask = derive_motion_mask(gt_flows[i], ego, threshold=motion_thr)
        fractions[i] = float(mask.mean())
    return fractions


def parse_args():
    ap = argparse.ArgumentParser(description="Per-frame pose error vs frame index / distance travelled")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", nargs="*", default=None, help="Default: all SINTEL_EVAL_SEQUENCES")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument(
        "--gate_modes", nargs="*", default=["predicted"], choices=["predicted", "oracle", "off"],
        help="predicted = model's own gate; oracle = GT flow-residual (m_geo) mask as gate bias; "
        "off = no gate at all (bias forced to ~0 everywhere, true no-gate control)",
    )
    ap.add_argument("--motion_thr", type=float, default=2.0, help="px threshold for GT flow-residual oracle mask")
    ap.add_argument("--k", type=float, default=30.0, help="oracle logit magnitude (bias ~ -softplus(k))")
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument(
        "--frame0_dissim", action="store_true",
        help="add a panel: error vs patch-feature (learned) cosine dissimilarity to frame 0 "
        "(tests whether error tracks appearance dissimilarity to the frame-0 coordinate anchor "
        "rather than frame index / distance travelled)",
    )
    ap.add_argument(
        "--frame0_rgb_dissim", action="store_true",
        help="add a panel: error vs raw-pixel cosine dissimilarity to frame 0 (model-free control "
        "for --frame0_dissim)",
    )
    ap.add_argument(
        "--frame_dynamic_frac", action="store_true",
        help="add a panel: error vs this frame's own GT dynamic-pixel fraction (no comparison to "
        "frame 0 -- direct proxy for the §4.2 information-starvation mechanism)",
    )
    ap.add_argument("--out_dir", default=None, help="Default: outputs/{}/<exp>".format(ERROR_GROWTH))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir = args.out_dir or default_output_dir(args.ckpt, ERROR_GROWTH)
    return args


def per_frame_ape(pred_extrinsics: np.ndarray, gt_tum: np.ndarray, gt_ts: np.ndarray):
    pred_tum, pred_ts = extrinsics_w2c_to_tum(pred_extrinsics)
    pred_traj = _make_traj(pred_tum, pred_ts)
    gt_traj = _make_traj(gt_tum, gt_ts)

    n = min(pred_traj.num_poses, gt_traj.num_poses)
    pred_traj.reduce_to_ids(list(range(n)))
    gt_traj.reduce_to_ids(list(range(n)))
    pred_traj.timestamps = gt_traj.timestamps

    result = main_ape.ape(
        gt_traj, pred_traj, pose_relation=PoseRelation.translation_part, align=True, correct_scale=True
    )
    err = result.np_arrays["error_array"]  # per-frame absolute translation error, post Sim(3) alignment
    gt_xyz = gt_traj.positions_xyz
    cum_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1))])
    return err, cum_dist, float(result.stats["rmse"])


DISSIM_LABEL = {
    "patch_feat": "learned patch-feature cosine dissimilarity vs frame 0",
    "rgb": "raw-pixel cosine dissimilarity vs frame 0",
    "dyn_frac": "this frame's own GT dynamic-pixel fraction",
}


def plot_seq(seq: str, per_mode: dict, out_path: str, dissims: dict[str, np.ndarray] | None = None):
    dissims = dissims or {}
    ncols = 2 + len(dissims)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5))
    axes = np.atleast_1d(axes)
    ax1, ax2 = axes[0], axes[1]
    extra_axes = list(zip(dissims.keys(), axes[2:]))

    for mode, (err, cum_dist, ate) in per_mode.items():
        style, fit_style = MODE_STYLE.get(mode, ("b-o", "k--"))
        frames = np.arange(len(err))

        z = np.polyfit(frames, err, 1)
        ax1.plot(frames, err, style, markersize=3, label=f"{mode} (rmse={ate:.4f})")
        ax1.plot(frames, np.poly1d(z)(frames), fit_style, linewidth=1, label=f"{mode} fit slope={z[0]:.5f}/frame")

        z2 = np.polyfit(cum_dist, err, 1)
        ax2.plot(cum_dist, err, style, markersize=3, label=f"{mode} (rmse={ate:.4f})")
        ax2.plot(cum_dist, np.poly1d(z2)(cum_dist), fit_style, linewidth=1, label=f"{mode} fit slope={z2[0]:.5f}/dist")

        r_frame = np.corrcoef(frames, err)[0, 1]
        r_dist = np.corrcoef(cum_dist, err)[0, 1]
        msg = f"  {seq} [{mode}]: rmse={ate:.4f}, r(err,frame_idx)={r_frame:.3f}, r(err,cum_dist)={r_dist:.3f}"

        for name, ax in extra_axes:
            d_full = dissims[name]
            n = min(len(err), len(d_full))
            d, e = d_full[:n], err[:n]
            # scatter only -- x is not time-ordered here, so a connecting line would zigzag
            # back and forth across the axis in temporal order and look like noise
            ax.plot(d, e, style, markersize=3, linestyle="none", label=f"{mode} (rmse={ate:.4f})")

            valid = ~np.isnan(d)
            d_v, e_v = d[valid], e[valid]
            z3 = np.polyfit(d_v, e_v, 1)
            order = np.argsort(d_v)
            ax.plot(
                d_v[order], np.poly1d(z3)(d_v[order]), fit_style, linewidth=1,
                label=f"{mode} fit slope={z3[0]:.5f}",
            )
            r_dissim = np.corrcoef(d_v, e_v)[0, 1]
            msg += f", r(err,{name})={r_dissim:.3f}"

        print(msg)

    ax1.set_xlabel("frame index")
    ax1.set_ylabel("abs. translation error (post-align)")
    ax1.set_title(f"{seq}: error vs frame index")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("cumulative GT distance travelled")
    ax2.set_ylabel("abs. translation error (post-align)")
    ax2.set_title(f"{seq}: error vs distance travelled")
    ax2.legend(fontsize=8)

    for name, ax in extra_axes:
        ax.set_xlabel(DISSIM_LABEL.get(name, name))
        ax.set_ylabel("abs. translation error (post-align)")
        ax.set_title(f"{seq}: error vs {name}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    args = parse_args()
    from data.sintel_io import resolve_sintel_root

    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or SINTEL_EVAL_SEQUENCES

    needs_gate_predictor = "predicted" in args.gate_modes or "oracle" in args.gate_modes
    model = load_vggt_for_eval(args.ckpt, device=args.device, require_gate=needs_gate_predictor)

    for seq in seqs:
        try:
            rgb_paths = load_sintel_rgb_paths(sintel_root, seq)
            _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
            gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)

            dissims = {}
            if args.frame0_dissim or args.frame0_rgb_dissim:
                images_d = load_and_preprocess_images(rgb_paths, mode="crop")
                if args.frame0_dissim:
                    dissims["patch_feat"] = frame0_dissimilarity(model, images_d, args.device)
                if args.frame0_rgb_dissim:
                    dissims["rgb"] = frame0_rgb_dissimilarity(images_d)
            if args.frame_dynamic_frac:
                dissims["dyn_frac"] = frame_dynamic_fraction(sintel_root, seq, rgb_paths, cam_dir, args.motion_thr)

            oracle_logits = None
            off_logits = None
            if "oracle" in args.gate_modes or "off" in args.gate_modes:
                images = load_and_preprocess_images(rgb_paths, mode="crop")
                s, _, h, w = images.shape
                ph, pw = h // args.patch_size, w // args.patch_size
                if "oracle" in args.gate_modes:
                    masks, _ = sintel_masks_and_fraction(sintel_root, seq, rgb_paths, args.motion_thr)
                    oracle_logits = oracle_logits_from_masks(masks, s, ph, pw, args.k).to(args.device)
                if "off" in args.gate_modes:
                    # True no-gate control: bias ~= -softplus(-k) ~= 0 everywhere (not the same as
                    # g==0, since softplus(0) = ln(2) != 0 -- see gate_bias_ablation.py's NOTE).
                    off_logits = torch.full((1, s, ph * pw), -args.k, device=args.device)

            per_mode = {}
            for mode in args.gate_modes:
                override = {"oracle": oracle_logits, "off": off_logits}.get(mode)
                pred = infer_sequence_chunked(
                    model, rgb_paths, device=args.device, chunk_size=args.chunk_size, gate_logits_override=override
                )
                per_mode[mode] = per_frame_ape(pred["extrinsic"], gt_tum, gt_ts)
        except Exception as e:
            print(f"[skip] {seq}: {e}")
            continue
        out_path = os.path.join(args.out_dir, f"{seq}.png")
        plot_seq(seq, per_mode, out_path, dissims=dissims)


if __name__ == "__main__":
    main()
