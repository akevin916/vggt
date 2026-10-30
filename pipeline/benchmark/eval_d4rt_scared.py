#!/usr/bin/env python3
"""SCARED pose + depth benchmark for OpenD4RT (unofficial D4RT reproduction, CVPR'26 baseline).

The protocol is eval_scared's, unchanged: this entry point runs eval_scared.evaluate with
OpenD4RT swapped in as the model, so frame selection, chunking (independent overlapping
chunks + snippet ATE, never stitched), depth resizing and the afsfm depth protocol are the
exact code paths our own numbers go through. Only the forward pass differs
(pipeline/eval/d4rt_infer.py: paper Sec. 2.3 readout of depth and pose from point queries).

OpenD4RT sees at most 48 frames per pass (its time embeddings have 48 entries), so the
multi-frame rows use --chunk_size 48 --overlap 16 -- the 48-frame counterpart of the
c64 rows in docs/results/sota.md. Compare against our checkpoints re-run at c48, not c64.

Run (from the repo root):
  python -m pipeline.benchmark.eval_d4rt_scared --split test/pose --n_frames 0 --chunk_size 48 --overlap 16
  python -m pipeline.benchmark.eval_d4rt_scared --split test/depth --n_frames 0 --chunk_size 48 \
      --overlap 16 --depth_protocol afsfm
  python -m pipeline.benchmark.eval_d4rt_scared --split test/depth --n_frames 0 --single_view \
      --depth_protocol afsfm
"""
from __future__ import annotations

import argparse
import os

from pipeline.benchmark import eval_scared as es
from pipeline.data.paths import data_path
from pipeline.eval.d4rt_infer import DEFAULT_CKPT, infer_sequence_d4rt, load_d4rt_for_eval
from pipeline.eval.paths import exp_name_from_ckpt, output_dir_for_exp

MAX_CLIP = 48


def _out_dir(args) -> str:
    tag = es.protocol_tag(args)
    ckpt = os.path.abspath(args.ckpt)
    if "ckpts" in ckpt.split(os.sep):                       # our own fine-tuned OpenD4RT (B)
        return os.path.join(output_dir_for_exp(exp_name_from_ckpt(ckpt), es.OUT_TOOL),
                            f"{tag}_{es.ckpt_role(ckpt)}")
    exp = os.path.basename(os.path.dirname(ckpt))           # released HF weights (A)
    return os.path.join(output_dir_for_exp(exp, es.OUT_TOOL), f"{tag}_zeroshot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--scared_root", default=data_path("train", "scared"))
    ap.add_argument("--split", default="test/pose", choices=es.SPLITS)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--n_frames", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=MAX_CLIP)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--max_depth", type=float, default=None)
    ap.add_argument("--depth_protocol", default="afsfm", choices=["monst3r", "afsfm"])
    ap.add_argument("--single_view", action="store_true")
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if not args.single_view and not 0 < args.chunk_size <= MAX_CLIP:
        raise SystemExit(f"--chunk_size must be in 1..{MAX_CLIP} (OpenD4RT clip limit)")

    # eval_scared fields that only matter for VGGT gate/key-mask variants: neutral values keep
    # protocol_tag identical to the matching VGGT run (e.g. pose_c48o16).
    args.gate_mode, args.key_mask, args.img_size, args.fid_step = "predicted", None, 518, 1
    args.save_pred = False
    args.out_dir = args.out_dir or _out_dir(args)

    es.load_vggt_for_eval = lambda ckpt, device="cuda": load_d4rt_for_eval(ckpt, device=device)
    es.infer_sequence = lambda model, paths, device="cuda", **kw: infer_sequence_d4rt(
        model, paths, device=device, want_depth=not args.no_depth)

    print(f"OpenD4RT {args.ckpt} | {args.split} c{args.chunk_size}o{args.overlap} -> {args.out_dir}")
    res = es.evaluate(args)
    p, d = res["pose"]["mean"], res["depth"]["mean"]
    for seq, v in res["pose"]["per_seq"].items():
        if "snippet_ate" in v:
            print(f"  {seq:<22} snippet_ate {v['snippet_ate']:.4f}  snippet_rot {v['snippet_rot']:.4f}")
    if d:
        print("  depth " + "  ".join(f"{k} {d[k]:.4f}" for k in ("abs_rel", "sq_rel", "rmse", "rmse_log")
                                     if k in d))
    print(f"results.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
