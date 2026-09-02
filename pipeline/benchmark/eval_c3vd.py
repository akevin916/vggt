#!/usr/bin/env python3
"""C3VD (colonoscopy phantom) pose + depth benchmark.

C3VD is converted by data/preprocess/c3vd_convert.py into the SAME on-disk layout SCARED
uses (image_left/, depth_left/ as uint16 hundredths of a millimetre, cam_data/{frames,
extrinsics,intrinsics,valid_frac}.txt, nested two levels). That was a deliberate choice at
conversion time, and it pays off here: the scoring machinery in benchmark/eval_scared.py is
already dataset-agnostic given that layout, so this module REUSES it rather than copying it.
Only the defaults and the paths differ. Do not fork eval_scared.eval_ckpt for C3VD -- if a
change is needed there, it should stay shared, since both datasets are scored under the same
protocol (full-sequence Sim3-aligned ATE via evo, plus AF/EndoSfM3D-style snippet ATE).

What differs from SCARED, and why:

  * DEPTH CEILING 100 mm, not 200. The C3VD release itself clamps depth at 100 mm and marks
    everything beyond it invalid, so a higher cap cannot admit any real surface -- it would
    only widen the range the depth metrics normalise over.
  * SPLIT DEFAULT IS `test`. C3VD, unlike SCARED, HAS a held-out split: all texture-4
    sequences, which also contains the only descending-colon sequence (unseen texture AND
    unseen anatomy). Report numbers belong there. The trainer's channel B passes
    `split: val` instead, so its per-epoch signal stays on the same data channel A sees.
  * FRAMES ARE GAPLESS AND CONTIGUOUS in every C3VD sequence, so snippet ATE is always
    meaningful when --n_frames keeps stride 1. Sequences run 148-1142 frames while VGGT
    tops out near 80 at 518 resolution, so --n_frames subsamples by default -- which makes
    the selection non-contiguous and skips the snippet column. Pass --n_frames 0 with
    --chunk_size to score every frame instead (see eval_scared for why full-sequence ATE is
    then withheld rather than stitched).

THE BASELINE POLICY STILL APPLIES (project memory): the comparison rows are a FINE-TUNED
VGGT-1B and MonST3R. Pretrained VGGT-1B is a sanity reference, not a baseline.

Run (from the repo root):
  python -m pipeline.benchmark.eval_c3vd --ckpts logs/c3vd_cam_vanilla/ckpts/best.pt
  python -m pipeline.benchmark.eval_c3vd --ckpts a.pt b.pt --split val --n_frames 50
"""
from __future__ import annotations

import os


import argparse
import json
from datetime import datetime

from pipeline.benchmark.eval_scared import eval_ckpt, list_sequences
from pipeline.data.paths import data_path
from pipeline.eval.paths import default_output_dir, output_dir_for_exp

TOOL = "eval_c3vd"

C3VD_MAX_DEPTH_MM = 100.0   # the release's own clamp; see module header


def evaluate(args, model=None):
    """Single-checkpoint entry point, mirroring benchmark.eval_scared.evaluate.

    Returns ``{"pose": {...}, "depth": {...}, "frames": {...}}`` and drops a results.json in
    ``args.out_dir``. The trainer calls this once per epoch with its live model; the CLI
    below calls it once per ``--ckpts`` entry.
    """
    root = getattr(args, "c3vd_root", None) or data_path("train", "c3vd")
    seqs = getattr(args, "seqs", None) or list_sequences(root, args.split)
    seqs = [str(x) for x in seqs]   # OmegaConf nodes are not JSON-serialisable
    # eval_scared.eval_ckpt reads args.max_depth through resolve_max_depth, which defaults to
    # the SCARED cap when unset. Pin it here so a C3VD run can never inherit 200 mm silently.
    if getattr(args, "max_depth", None) is None:
        args.max_depth = C3VD_MAX_DEPTH_MM

    suffix = "" if getattr(args, "gate_mode", "predicted") == "predicted" else f"_gate{args.gate_mode}"
    args.out_dir = getattr(args, "out_dir", None) or output_dir_for_exp(
        f"{args.split}_{args.n_frames}f{suffix}", TOOL)
    os.makedirs(args.out_dir, exist_ok=True)

    results = eval_ckpt(args.ckpt, root, args.split, seqs, args, model=model)

    payload = dict(meta=dict(ckpt=args.ckpt, c3vd_root=root, split=args.split, seqs=seqs,
                             n_frames=args.n_frames, max_depth=float(args.max_depth),
                             depth_protocol=getattr(args, "depth_protocol", "monst3r"),
                             timestamp=datetime.now().isoformat(timespec="seconds")),
                   results=results)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--c3vd_root", default=data_path("train", "c3vd"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="e.g. cecum/t4_a; default: every sequence in the split")
    ap.add_argument("--n_frames", type=int, default=50,
                    help="frames taken evenly across each sequence in ONE forward pass. "
                         "0 = every frame (needs --chunk_size; see header)")
    ap.add_argument("--chunk_size", type=int, default=0,
                    help="0 = single pass. >0 splits into overlapping chunks; the "
                         "full-sequence ATE column is then withheld, not stitched")
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--max_depth", type=float, default=None,
                    help=f"millimetres; default {C3VD_MAX_DEPTH_MM} (the release's clamp)")
    ap.add_argument("--depth_protocol", default="monst3r", choices=["monst3r", "afsfm"])
    ap.add_argument("--single_view", action="store_true",
                    help="one frame per forward pass: comparable to monocular baselines, "
                         "pose undefined and therefore skipped")
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate_mode", default="predicted", choices=["predicted", "off"])
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    explicit_out = args.out_dir
    for ckpt in args.ckpts:
        # Per-checkpoint directory keyed on the EXP name, not the checkpoint file name:
        # several runs all produce a best_ate.pt, and keying on the file name makes them
        # overwrite each other's results.json.
        args.out_dir = explicit_out or default_output_dir(ckpt, TOOL)
        args.ckpt = ckpt
        res = evaluate(args, model=None)
        mp = res.get("pose", {}).get("mean", {})
        md = res.get("depth", {}).get("mean", {})
        print(f"\n== {ckpt} -> {args.out_dir}")
        print("   pose :", {k: round(v, 4) for k, v in mp.items()})
        print("   depth:", {k: round(v, 4) for k, v in md.items()})


if __name__ == "__main__":
    main()
