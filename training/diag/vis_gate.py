#!/usr/bin/env python3
"""Gate diagnostic: visualize m_gt / m_star_patch / g (PO) or m_star_raft_patch / g (Sintel).

Primary tool for inspecting v3 gate behavior. Optional --metrics writes a minimal summary.json.
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import argparse
import json
from datetime import datetime
from typing import Any, Dict

from eval.gate_vis import print_diag_summary, run_po_vis, run_sintel_vis
from eval.paths import TRAINING_DIR, VIS_GATE, default_eval_dir
from eval.sintel_io import resolve_sintel_root
from eval.vggt_infer import load_vggt_for_eval


def parse_args():
    ap = argparse.ArgumentParser(description="Gate diagnostic visualization (PO + Sintel)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--dataset",
        choices=["po", "sintel", "all"],
        default="po",
        help="po=in-domain PO (default); sintel=cross-domain flow-residual; all=both",
    )
    ap.add_argument("--out_dir", default=None, help=f"Default: logs/<exp>/{VIS_GATE}")
    ap.add_argument("--metrics", action="store_true", help="Write minimal summary.json")
    ap.add_argument("--n_clips", type=int, default=2, help="PO clips to visualize")
    ap.add_argument("--img_per_seq", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--gate_block_iter", type=int, default=7)
    ap.add_argument("--dynamic_source", default="instance", choices=["native", "raft", "instance"])
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--seqs", nargs="*", default=None, help="Sintel sequences (default: DIAG_SEQUENCES)")
    ap.add_argument("--max_frames", type=int, default=12)
    ap.add_argument("--motion_thr", type=float, default=2.0)
    ap.add_argument("--n_vis_per_seq", type=int, default=3, help="Sintel frames saved per sequence")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir = args.out_dir or default_eval_dir(args.ckpt, VIS_GATE, TRAINING_DIR)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    if args.dataset in ("sintel", "all"):
        args.sintel_root = resolve_sintel_root(args.sintel_root)
        print(f"Sintel root: {args.sintel_root}")

    model = load_vggt_for_eval(
        args.ckpt,
        img_size=args.img_size,
        gate_block_iter=args.gate_block_iter,
        device=args.device,
        require_gate=True,
    )

    report: Dict[str, Any] = {
        "meta": {
            "ckpt": args.ckpt,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
        }
    }

    if args.dataset in ("po", "all"):
        report["po"] = run_po_vis(model, args, args.out_dir)
        print_diag_summary("PointOdyssey", report["po"])

    if args.dataset in ("sintel", "all"):
        report["sintel"] = run_sintel_vis(model, args, args.out_dir, args.sintel_root)
        print_diag_summary("Sintel", report["sintel"])

    if args.metrics:
        json_path = os.path.join(args.out_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved {json_path}")

    print(f"\nDone. Images -> {args.out_dir}/")


if __name__ == "__main__":
    main()
