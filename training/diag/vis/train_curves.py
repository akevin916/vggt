#!/usr/bin/env python3
"""Parse a trainer log.txt and plot per-epoch train/val loss + val-metric curves.

One subplot per loss component and per val metric, train vs val overlaid.
Writes two figures: loss.png and metrics.png.

Usage:
    python diag/vis/train_curves.py --log logs/<exp>/log.txt
"""
from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import argparse
import math
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_utils.paths import TRAIN_CURVES, output_dir_for_exp

PHASE_RE = re.compile(r"(Train|Val) Epoch: \[(\d+)\]\[\s*(\d+)/\d+\]")
# "Loss/train_loss_gate: 0.0981 (0.0841)" -> key, running average
FIELD_RE = re.compile(r"(Loss/\S+|Metric/\S+): [\d.]+ \(([\d.]+)\)")

TRAIN_KW = dict(color="tab:blue", marker="o", ms=4, label="train")
VAL_KW = dict(color="tab:orange", marker="s", ms=4, label="val")

TITLES = {
    "objective": "loss all (total objective)",
    "gate": "loss_gate (BCE)",
    "camera": "loss_camera (T·w + R·w)",
    "T": "loss_T (translation)",
    "R": "loss_R (rotation)",
    "camera_smooth": "loss_camera_smooth",
    "abs_rel": "depth AbsRel",
    "delta_1": "depth delta_1",
    "rmse": "depth RMSE",
    "ate": "pose ATE",
    "rpe_trans": "pose RPE-trans",
    "rpe_rot": "pose RPE-rot (deg)",
}
LOSS_ORDER = ["objective", "gate", "camera", "T", "R", "camera_smooth"]
METRIC_ORDER = ["abs_rel", "delta_1", "rmse", "ate", "rpe_trans", "rpe_rot"]


def parse(log_path: str):
    """({phase: {epoch: last-line {short_key: running_avg}}}, {short_key: "Loss"|"Metric"}).

    The last line's running average over an epoch is that epoch's mean.
    """
    out = {"Train": {}, "Val": {}}
    groups = {}
    for line in open(log_path, "r", encoding="utf-8", errors="ignore"):
        m = PHASE_RE.search(line)
        if not m:
            continue
        phase, epoch = m.group(1), int(m.group(2))
        fields = {}
        for key, avg in FIELD_RE.findall(line):
            group, short = key.split("/", 1)
            for prefix in ("train_loss_", "val_loss_", "train_", "val_"):
                if short.startswith(prefix):
                    short = short[len(prefix):]
                    break
            fields[short] = float(avg)
            groups[short] = group
        if fields:
            out[phase][epoch] = fields
    return out, groups


def series(per_epoch, key):
    xs, ys = [], []
    for epoch in sorted(per_epoch):
        if key in per_epoch[epoch]:
            xs.append(epoch)
            ys.append(per_epoch[epoch][key])
    # A metric that is identically zero carries no signal (e.g. a disabled loss).
    return (xs, ys) if any(ys) else ([], [])


def make_figure(panels, exp_name, subtitle, out_path, ncols):
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, (key, (tx, ty), (vx, vy)) in zip(flat, panels):
        if tx:
            ax.plot(tx, ty, **TRAIN_KW)
        if vx:
            ax.plot(vx, vy, **VAL_KW)
        ax.set_title(TITLES.get(key, key), fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in flat[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"{exp_name} — {subtitle} (per-epoch mean, from log.txt)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ncols", type=int, default=3)
    args = ap.parse_args()

    data, groups = parse(args.log)
    train, val = data["Train"], data["Val"]
    if not train:
        raise SystemExit(f"no 'Train Epoch:' lines parsed from {args.log}")

    exp_name = os.path.basename(os.path.dirname(os.path.abspath(args.log)))
    out_dir = args.out_dir or output_dir_for_exp(exp_name, TRAIN_CURVES)
    os.makedirs(out_dir, exist_ok=True)

    seen = set(groups)
    for group, order, subtitle, fname in [
        ("Loss", LOSS_ORDER, "training/val loss", "loss.png"),
        ("Metric", METRIC_ORDER, "val metrics", "metrics.png"),
    ]:
        keys = {k for k in seen if groups[k] == group}
        keys = [k for k in order if k in keys] + sorted(keys - set(order))
        panels = [(k, series(train, k), series(val, k)) for k in keys]
        panels = [p for p in panels if p[1][0] or p[2][0]]
        if not panels:
            continue
        make_figure(panels, exp_name, subtitle, os.path.join(out_dir, fname), args.ncols)


if __name__ == "__main__":
    main()
