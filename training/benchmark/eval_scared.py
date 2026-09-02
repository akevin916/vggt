#!/usr/bin/env python3
"""SCARED (endoscopy) pose + depth benchmark.

Promoted from diag/ on 2026-08-18: these numbers go in the mid-term report, and the
trainer now calls this module as its channel-B metric for SCARED runs (see
trainer.run_pose_eval), so the entry-point contract has to stay stable.

Protocol notes:
  * Sequences come from the val split: 6 keyframes x ~284 CONTIGUOUS frames. The test split
    is the official sparse sampling (stride 8-38) and must not be treated as video.
  * ``--n_frames`` frames are taken evenly across each keyframe in ONE forward pass. Chunked
    inference would split the sequence into independent passes and inflate ATE, so the frame
    count is capped instead (same frames for every checkpoint -> comparable).
  * GT extrinsics are already world-to-cam (docs/topics/scared_dataset.md #2), converted to TUM the
    same way as Sintel: c2w translation, mean-centred, wxyz quaternion.
  * ATE is computed with scale alignment (evo, correct_scale=True), so SCARED's millimetre
    units need no conversion.
  * Depth is in mm here, so --max_depth is in mm too (Sintel's 70 is metres).
  * ``--depth_protocol`` picks how depth is scored. ``monst3r`` (default, unchanged) is the
    repo's own pooled scale+shift regime. ``afsfm`` reproduces AF-SfMLearner's SCARED
    protocol -- the one EndoSfM3D / EndoDAC / DARES / Endo-FASt3r tables are all computed
    under: per-frame median scaling, depth range (1e-2, 150] mm, prediction clipped into
    that range after scaling, and per-frame metrics averaged UNWEIGHTED over frames.
    Source: EndoSfM3D dares/evaluate_depth_scared.py (github.com/MOYF-beta/EndoSfM3D),
    which is reference/AF-SfMLearner/evaluate_depth.py line for line bar MIN_DEPTH.
  * The POSE metric here already matches EndoSfM3D: their
    dares/evaluate_pose_and_intrinsics.py accumulates the predicted relative poses into one
    trajectory, aligns with evo ``align(correct_scale=True)``, and reports
    ``APE(PoseRelation.translation_part)`` plus ``RPE(rotation_angle_deg)`` -- i.e. the same
    full-sequence Sim3-aligned ATE computed below, on test_files_sequence{1,2}.
    Do NOT confuse this with AF-SfMLearner's own 2022 evaluate_pose.py, which averages
    errors over sliding 5-frame windows and yields numbers an order of magnitude smaller.
    EndoSfM3D dropped that; the tables we compare into are full-trajectory.

Run (from training/):
  python benchmark/eval_scared.py                       # all 4 ckpts, 6 val keyframes
  python benchmark/eval_scared.py --ckpts checkpoints/inst_g.pt --seqs dataset2/keyframe3
"""
from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import argparse
import json
import traceback
import warnings
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from data.paths import data_path
from data.sintel_io import compute_preprocess_meta, resize_pred_to_gt
from eval_utils.metrics_depth import average_depth_results, eval_sequence_depth
from eval_utils.metrics_pose import (eval_pose_metrics, snippet_metrics_from_chunks,
                                     snippet_pose_metrics)
from eval_utils.paths import exp_name_from_ckpt, output_dir_for_exp
from eval_utils.vggt_infer import infer_sequence, infer_sequence_stitched, load_vggt_for_eval

TOOL = "eval_scared"
DEPTH_SCALE = 100.0  # uint16 counts per mm


# EndoSfM3D's dares/evaluate_depth_scared.py is AF-SfMLearner's evaluate_depth.py line for
# line except MIN_DEPTH (1e-3 -> 1e-2). We follow EndoSfM3D, since that is the table we are
# comparing into; at SCARED's millimetre scale the two floors are equally inert anyway.
AFSFM_MIN_DEPTH = 1e-2
AFSFM_MAX_DEPTH = 150.0


def resolve_max_depth(args) -> float:
    """--max_depth defaults per protocol so afsfm cannot silently score at the wrong cap."""
    if getattr(args, "max_depth", None) is not None:
        return float(args.max_depth)
    return AFSFM_MAX_DEPTH if getattr(args, "depth_protocol", "monst3r") == "afsfm" else 200.0


def list_sequences(root: str, split: str):
    out = []
    split_dir = os.path.join(root, split)
    for ds in sorted(os.listdir(split_dir)):
        for kf in sorted(os.listdir(os.path.join(split_dir, ds))):
            if os.path.isdir(os.path.join(split_dir, ds, kf, "image_left")):
                out.append(f"{ds}/{kf}")
    return out


def load_sequence(seq_dir: str, n_frames: int):
    fids = np.loadtxt(os.path.join(seq_dir, "cam_data", "frames.txt"), dtype=np.int64, ndmin=1)
    E = np.loadtxt(os.path.join(seq_dir, "cam_data", "extrinsics.txt")).reshape(-1, 3, 4)
    # pose_seq is converted with --no_depth, so valid_frac.txt is header-only there.
    vf_path = os.path.join(seq_dir, "cam_data", "valid_frac.txt")
    with warnings.catch_warnings():           # empty file -> "input contained no data"
        warnings.simplefilter("ignore", UserWarning)
        vf = np.loadtxt(vf_path, ndmin=1)
    if vf.size != len(fids):
        vf = np.full(len(fids), np.nan)

    # n_frames <= 0 keeps every frame: the AF/EndoSfM3D pose protocol scores all 411 / 834,
    # and any subsampling both changes which windows the snippet mean runs over and thins the
    # drift a full-sequence ATE is supposed to accumulate.
    idx = (np.arange(len(fids)) if n_frames <= 0 else
           np.unique(np.linspace(0, len(fids) - 1, min(n_frames, len(fids))).astype(int)))
    paths = [os.path.join(seq_dir, "image_left", f"{int(fids[i]):06d}.png") for i in idx]

    # GT poses -> TUM (xyz + wxyz), mean-centred, exactly like load_sintel_gt_poses
    tum, ts = [], []
    for i in idx:
        w2c = np.vstack([E[i], [0, 0, 0, 1]])
        c2w = np.linalg.inv(w2c)
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # xyzw
        tum.append(np.concatenate([c2w[:3, 3], [q[3], q[0], q[1], q[2]]]))
        ts.append(float(fids[i]))
    tum = np.stack(tum)
    tum[:, :3] -= tum[:, :3].mean(axis=0, keepdims=True)
    # ATE alone is not readable across datasets: SCARED's camera travels tens of mm while
    # Sintel's travels scene-scale units. Report the trajectory's own extent so ATE can be
    # expressed as a fraction of it.
    C = tum[:, :3]
    scale = dict(bbox_diag=float(np.linalg.norm(C.max(0) - C.min(0))),
                 path_len=float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()))
    return paths, tum, np.array(ts, dtype=np.float64)[:, None], idx, vf[idx], scale, E[idx]


def load_gt_depths(seq_dir: str, fids_sel):
    out = []
    for fid in fids_sel:
        d = cv2.imread(os.path.join(seq_dir, "depth_left", f"{int(fid):06d}.png"), cv2.IMREAD_UNCHANGED)
        out.append(d.astype(np.float32) / DEPTH_SCALE)
    return out


def eval_ckpt(ckpt, root, split, seqs, args, model=None):
    """Score one checkpoint. ``model``: optional preloaded (in-memory) VGGT -- when given,
    nothing is loaded from disk and the model is left alive for the caller (the trainer
    scores its own live model this way; see benchmark.eval_sintel.evaluate for the twin)."""
    owns_model = model is None
    if owns_model:
        model = load_vggt_for_eval(ckpt, device=args.device)
    fids_all = {}
    per_seq_pose, per_seq_depth = {}, {}

    for seq in tqdm(seqs, desc=os.path.basename(ckpt)):
        seq_dir = os.path.join(root, split, seq)
        try:
            paths, gt_tum, gt_ts, idx, vfs, scale, gt_E = load_sequence(seq_dir, args.n_frames)
            override = None
            if args.gate_mode == "off" and getattr(model.aggregator, "gate_predictor", None):
                # zeros -> bias = clamp(log2 - softplus(0), max=0) = 0 exactly: gate disabled
                import torch as _t
                # SCARED is 1280x1024, so crop-mode gives 518x420 (not a square 518x518):
                # the patch grid is 37x30 = 1110, and a hard-coded (518//14)**2 is wrong.
                _m = compute_preprocess_meta(paths[0], target_size=args.img_size)
                _h = min(_m.new_h, args.img_size)
                n_patch = (args.img_size // 14) * (_h // 14)
                override = _t.zeros(1, len(paths), n_patch, device=args.device)
            # Snippet ATE is only meaningful on stride-1 frames: its 5-frame windows are
            # supposed to span ~5 video frames. On the depth test split (stride 3-296) or any
            # subsampled run they span hundreds, and the metric silently returns a large,
            # meaningless number instead of failing. Gate it on the frames actually selected.
            sel_fids = np.array([int(os.path.basename(p)[:-4]) for p in paths])
            contiguous = bool(len(sel_fids) > 1 and np.all(np.diff(sel_fids) == 1))

            chunk = int(getattr(args, "chunk_size", 0) or 0)
            if getattr(args, "single_view", False):
                # One frame per forward pass, so depth is predicted from a SINGLE image --
                # the setting AF-SfMLearner / EndoDAC / EndoSfM3D actually work in. Our
                # multi-view numbers are not comparable to theirs without this: VGGT
                # otherwise sees every frame of the keyframe at once and can triangulate.
                # Pose is undefined here (a lone view has nothing to be relative to), so the
                # pose columns are skipped rather than filled with the identity.
                depths = []
                for one in paths:
                    pr = infer_sequence(model, [one], device=args.device)
                    if "depth" not in pr:
                        break
                    depths.append(pr["depth"][0])
                pred = {"depth": np.stack(depths)} if len(depths) == len(paths) else {}
                per_seq_pose[seq] = {"single_view": True,
                                     "pose_skipped": "single-view input has no relative pose"}
            elif chunk > 0 and len(paths) > chunk:
                # Too long for one pass. We do NOT stitch: diag/stitch_error.py measured the
                # per-seam Sim3 moving full-sequence ATE by up to +32% on an 80-frame probe,
                # because VGGT predicts only ~0.4 mm of travel here and ATE's scale alignment
                # then magnifies every prediction-space wobble ~47x. Snippet ATE needs no
                # global frame at all -- each 5-frame window re-anchors and re-scales itself --
                # so independent overlapping chunks give the EXACT whole-sequence number
                # (verified bit-identical), and the full-sequence column is simply not
                # reported for these sequences rather than reported wrong.
                ov = min(max(int(getattr(args, "overlap", 16)), 4), chunk - 1)
                step = chunk - ov
                starts = list(range(0, max(len(paths) - ov, 1), step))
                if starts[-1] + chunk < len(paths):
                    starts.append(len(paths) - chunk)
                parts, depth_by_frame, best_margin = [], [None] * len(paths), [-1] * len(paths)
                pred = {}
                for st in starts:
                    sub = slice(st, min(st + chunk, len(paths)))
                    ov_sub = override[:, sub] if override is not None else None
                    pr = infer_sequence(model, paths[sub], device=args.device,
                                        gate_logits_override=ov_sub)
                    parts.append((st, pr["extrinsic"]))
                    if "depth" in pr:
                        # Depth is per-frame, so chunking is harmless -- but take each frame
                        # from the chunk where it sits furthest from a pass boundary, where
                        # VGGT has the most context on both sides.
                        for j in range(sub.stop - sub.start):
                            g = sub.start + j
                            margin = min(j, (sub.stop - sub.start) - 1 - j)
                            if margin > best_margin[g]:
                                best_margin[g], depth_by_frame[g] = margin, pr["depth"][j]
                if all(d is not None for d in depth_by_frame):
                    pred["depth"] = np.stack(depth_by_frame)
                per_seq_pose[seq] = (snippet_metrics_from_chunks(parts, gt_E) if contiguous
                                     else {"snippet_skipped": "frames not stride-1"})
                per_seq_pose[seq]["n_chunks"] = len(starts)
                per_seq_pose[seq]["full_seq_ate_reported"] = False
            else:
                pred = infer_sequence(model, paths, device=args.device,
                                      gate_logits_override=override)
                per_seq_pose[seq] = eval_pose_metrics(pred["extrinsic"], gt_tum, gt_ts)
                if contiguous:
                    per_seq_pose[seq].update(snippet_pose_metrics(pred["extrinsic"], gt_E))
                else:
                    per_seq_pose[seq]["snippet_skipped"] = "frames not stride-1"
                per_seq_pose[seq]["full_seq_ate_reported"] = True
            per_seq_pose[seq]["n_frames"] = len(paths)
            per_seq_pose[seq]["contiguous"] = contiguous
            per_seq_pose[seq]["valid_frac_mean"] = float(np.mean(vfs))
            per_seq_pose[seq].update(scale)
            if "ate" in per_seq_pose[seq]:
                per_seq_pose[seq]["ate_rel"] = (per_seq_pose[seq]["ate"]
                                                / max(scale["bbox_diag"], 1e-9))
            fids_all[seq] = [int(os.path.basename(p)[:-4]) for p in paths]

            if not args.no_depth and "depth" in pred:
                gt_d = load_gt_depths(seq_dir, fids_all[seq])
                metas = [compute_preprocess_meta(p) for p in paths]
                pred_d = [resize_pred_to_gt(pred["depth"][i], metas[i]) for i in range(len(paths))]
                afsfm = getattr(args, "depth_protocol", "monst3r") == "afsfm"
                max_d = resolve_max_depth(args)
                per_seq_depth[seq] = eval_sequence_depth(
                    pred_d, gt_d, max_depth=max_d,
                    align_with_lad2=not afsfm, per_frame=afsfm,
                    min_depth=AFSFM_MIN_DEPTH if afsfm else 0.0,
                    post_clip_min=AFSFM_MIN_DEPTH if afsfm else None,
                    post_clip_max=max_d, device=args.device)
        except Exception:
            traceback.print_exc()
    if owns_model:
        del model
    import torch
    torch.cuda.empty_cache()

    _keys = ("ate", "rpe_trans", "rpe_rot", "ate_rel", "snippet_ate", "snippet_rot")
    mean_pose = ({k: float(np.mean([v[k] for v in per_seq_pose.values() if k in v]))
                  for k in _keys if any(k in v for v in per_seq_pose.values())}
                 if per_seq_pose else {})
    return dict(pose=dict(per_seq=per_seq_pose, mean=mean_pose),
                depth=dict(per_seq=per_seq_depth,
                           mean=average_depth_results(
                               per_seq_depth,
                               weight_key=("num_frames"
                                           if getattr(args, "depth_protocol", "monst3r") == "afsfm"
                                           else "valid_pixels"))
                           if per_seq_depth else {}),
                frames=fids_all)


def evaluate(args, model=None):
    """Single-checkpoint entry point, mirroring benchmark.eval_sintel.evaluate.

    Returns ``{"pose": {...}, "depth": {...}, "frames": {...}}`` and drops a results.json
    in ``args.out_dir``. The trainer calls this once per epoch with its live model; the CLI
    below calls it once per ``--ckpts`` entry.
    """
    seqs = getattr(args, "seqs", None) or list_sequences(args.scared_root, args.split)
    seqs = [str(x) for x in seqs]   # config-supplied lists may be OmegaConf nodes (not JSON-able)
    suffix = "" if getattr(args, "gate_mode", "predicted") == "predicted" else f"_gate{args.gate_mode}"
    args.out_dir = getattr(args, "out_dir", None) or output_dir_for_exp(
        f"{args.split}_{args.n_frames}f{suffix}", TOOL)
    os.makedirs(args.out_dir, exist_ok=True)

    results = eval_ckpt(args.ckpt, args.scared_root, args.split, seqs, args, model=model)

    payload = dict(meta=dict(ckpt=args.ckpt, scared_root=args.scared_root, split=args.split,
                             seqs=seqs, n_frames=args.n_frames, max_depth=resolve_max_depth(args),
                             depth_protocol=getattr(args, "depth_protocol", "monst3r"),
                             timestamp=datetime.now().isoformat(timespec="seconds")),
                   results=results)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=["checkpoints/VGGT-1B.pt",
                                                  "checkpoints/inst_gate_init.pt",
                                                  "checkpoints/inst_g.pt",
                                                  "checkpoints/inst_gts.pt"])
    ap.add_argument("--scared_root", default=data_path("train", "scared"))
    ap.add_argument("--split", default="val",
                    choices=["val", "test", "train", "pose_seq"],
                    help="pose_seq = the two contiguous AF/EndoSfM3D pose trajectories "
                         "(411 and 834 frames, images+poses only, no depth)")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--n_frames", type=int, default=50,
                    help="frames per sequence, evenly sampled, in ONE pass. "
                         "0 or less = every frame (needs --chunk_size for long sequences).")
    ap.add_argument("--chunk_size", type=int, default=0,
                    help="0 = single pass. Above 0, sequences longer than this are inferred "
                         "as overlapping chunks joined by a per-seam Sim3 (see "
                         "eval_utils.vggt_infer.infer_sequence_stitched). Stitching adds its "
                         "own error -- measure it before publishing stitched numbers.")
    ap.add_argument("--overlap", type=int, default=16,
                    help="frames shared between consecutive chunks; >=3 needed to fit a Sim3")
    ap.add_argument("--max_depth", type=float, default=None,
                    help="mm (SCARED p99.9 <= 165). Default: 200 for monst3r, 150 for afsfm")
    ap.add_argument("--depth_protocol", default="monst3r", choices=["monst3r", "afsfm"],
                    help="afsfm = AF-SfMLearner SCARED protocol (per-frame median, <=150mm)")
    ap.add_argument("--single_view", action="store_true",
                    help="predict each frame from ONE image (S=1), the monocular setting the "
                         "published SCARED depth tables use. Depth only; pose is skipped.")
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate_mode", default="predicted", choices=["predicted", "off"],
                    help="'off' feeds zero gate logits, which makes the attention bias exactly 0")
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    seqs = args.seqs or list_sequences(args.scared_root, args.split)
    suffix = "" if args.gate_mode == "predicted" else f"_gate{args.gate_mode}"
    if args.single_view:
        suffix += "_single"
    if args.depth_protocol == "afsfm":
        suffix += "_afsfm"
    base_out = args.out_dir or output_dir_for_exp(f"{args.split}_{args.n_frames}f{suffix}", TOOL)
    print(f"{len(args.ckpts)} ckpts x {len(seqs)} sequences ({args.split}), "
          f"{args.n_frames} frames each -> {base_out}")

    results = {}
    for ckpt in args.ckpts:
        # Identity is the RUN, not the file name. Every run's ckpts are called best_ate.pt /
        # last.pt / epoch_N.pt, so keying on the basename alone made five different arms all
        # answer to "best_ate" -- one sub-dir, one json key, each ckpt silently destroying the
        # previous one (2026-08-25: a 6-ckpt sweep kept only the last arm, and overwrote two
        # older results.json in the process). <exp>_<stem> is unique in both directions; the
        # stem is dropped when it already equals <exp>, so checkpoints/VGGT-1B.pt stays
        # "VGGT-1B" rather than becoming "VGGT-1B_VGGT-1B".
        exp = exp_name_from_ckpt(ckpt)
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        name = exp if stem == exp else f"{exp}_{stem}"
        args.ckpt, args.seqs = ckpt, seqs
        # Always one sub-dir per ckpt. Keying it on len(ckpts) > 1 meant two single-ckpt runs
        # with the same split/flags wrote to the same results.json, and the second silently
        # destroyed the first -- which is exactly what happened when the VGGT-1B pose_seq run
        # landed on top of the fine-tuned one.
        args.out_dir = os.path.join(base_out, name)
        results[name] = evaluate(args)
    out_dir = base_out

    nan = float("nan")
    chunked = any(not v.get("full_seq_ate_reported", True) for r in results.values()
                  for v in r["pose"]["per_seq"].values())
    afsfm = args.depth_protocol == "afsfm"

    print(f"\n{'ckpt':<18}{'ATE[a]':>10}{'ATE/軌跡':>10}{'snipATE[b]':>12}{'snipRot':>10}"
          f"{'RPE_t':>9}{'RPE_r':>9}   {'abs_rel[c]':>11}{'delta_1':>9}")
    for name, r in results.items():
        p, d = r["pose"]["mean"], r["depth"]["mean"]
        print(f"{name:<18}{p.get('ate', nan):10.4f}{100*p.get('ate_rel', nan):9.2f}%"
              f"{p.get('snippet_ate', nan):12.4f}{p.get('snippet_rot', nan):10.4f}"
              f"{p.get('rpe_trans', nan):9.4f}{p.get('rpe_rot', nan):9.4f}"
              f"   {d.get('abs_rel', nan):11.4f}{d.get('delta_1', nan):9.4f}")

    # The two ATE columns are different quantities, not two estimates of one. Spell that out
    # here so a number never gets lifted out of this table into a paper row by mistake.
    print("\n[a] ATE      full-sequence, evo APE(translation) after Sim3 align(correct_scale)."
          "\n              Same as EndoSfM3D's released code (dares/evaluate_pose_and_intrinsics.py)."
          + ("\n              n/a for sequences longer than --chunk_size: they need several "
             "passes, and\n              joining those into one frame costs more than the "
             "metric is worth (up to +32%,\n              see diag/stitch_error.py). Not "
             "estimated rather than estimated wrong." if chunked else ""))
    print("[b] snipATE  mean over sliding 5-frame windows, each re-anchored and re-scaled\n"
          "              (AF-SfMLearner evaluate_pose.py). Drift never accumulates past 5\n"
          "              frames, so a drifting predictor scores far better here than on [a];\n"
          "              a purely jittery one scores similarly. This is the quantity the\n"
          "              published endoscopy pose tables (AF Table 10, EndoSfM3D Table 6)\n"
          "              contain. When the sequence needs several passes it is computed\n"
          "              per-chunk: every window sits entirely inside one pass, so the result\n"
          "              is exact either way (verified bit-identical against the single-pass\n"
          "              computation). NOTE EndoSfM3D's paper cites those snippet baselines\n"
          "              but ships full-sequence code; which one its own row used is not stated.")
    print(f"[c] abs_rel  depth_protocol={args.depth_protocol}"
          + (f", per-frame median scaling, ({AFSFM_MIN_DEPTH}, {resolve_max_depth(args)}] mm,"
             "\n              per-frame metrics averaged unweighted -- the AF/EndoSfM3D table "
             "protocol." if afsfm else
             f", pooled scale+shift over the sequence (MonST3R),\n              cap "
             f"{resolve_max_depth(args)} mm. NOT comparable to published SCARED tables; "
             "use --depth_protocol afsfm."))

    print(f"\nper-sequence")
    print(f"{'seq':<20}" + "".join(f"{n[:16]:>20}" for n in results))
    for label, key in (("ATE [a]", "ate"), ("snipATE [b]", "snippet_ate")):
        print(f"  {label}")
        for seq in seqs:
            row = "".join(
                f"{results[n]['pose']['per_seq'].get(seq, {}).get(key, float('nan')):20.4f}"
                for n in results)
            print(f"    {seq:<18}{row}")

    payload = dict(meta=dict(scared_root=args.scared_root, split=args.split, seqs=seqs,
                             n_frames=args.n_frames, max_depth=resolve_max_depth(args),
                             depth_protocol=args.depth_protocol,
                             chunk_size=args.chunk_size, overlap=args.overlap,
                             chunked_pose=chunked, single_view=bool(args.single_view),
                             protocol_notes=dict(
                                 ate="full-sequence evo APE(translation), Sim3 align with "
                                     "scale; matches EndoSfM3D released code",
                                 snippet_ate="mean over sliding 5-frame windows, per-window "
                                             "anchor + least-squares scale, error divided by "
                                             "N not sqrt(N); matches AF-SfMLearner "
                                             "evaluate_pose.py and the published tables",
                                 depth="afsfm = per-frame median scaling, (1e-2, 150] mm, "
                                       "per-frame metrics averaged unweighted"),
                             timestamp=datetime.now().isoformat(timespec="seconds")),
                   results=results)
    p = os.path.join(out_dir, "results.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
