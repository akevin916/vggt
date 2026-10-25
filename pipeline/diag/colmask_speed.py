#!/usr/bin/env python3
"""How much faster is a forward pass when global attention only computes the KEPT keys?

The masks in the model (column mask, frame-distance tile mask) only push dropped keys' scores to
-1e4 inside flex_attention, so every key is still computed and the masked model is SLOWER than
the plain one. This tool times the upper bound instead: gather the kept keys (plus one mean
token) and run plain SDPA on them.

  column mask      one kept set for every query -> one gathered SDPA per layer
  frame-distance   the kept set differs by query frame -> one gathered SDPA per query frame

TIMING ONLY. The gathered path drops the diagonal rule and the mean token's count weight, so
its outputs are not the masked model's outputs.

Settings (any subset, --settings): plain | flex_colmask | gather_colmask | flex_relmask |
gather_relmask. Reports the median of --n timed forwards after --warm warmups, for the whole
forward and for the global-attention share, plus speed-ups against plain.

Run (from the repo root):
  python -m pipeline.diag.colmask_speed --ckpt logs/scared_cam_depth_tfs_fpos_seed43/ckpts/last.pt \\
      --colmask outputs/colmask/d1_b50_learned.npy --settings plain,gather_colmask --tag _b50
Measured 2026-10-25 (RTX 5090 @400 W, 64 frames, bf16, flash on): column mask 25% -> 1.65x
whole forward, 50% -> 1.36x; frame-distance mask 25% -> 1.08x (per-frame loop).
"""
from __future__ import annotations

import argparse
import json
import os
import types

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data.paths import data_path
from pipeline.eval.paths import default_output_dir
from pipeline.eval.vggt_infer import load_vggt_for_eval
from vggt.utils.load_fn import load_and_preprocess_images

SETTINGS = ("plain", "flex_colmask", "gather_colmask", "flex_relmask", "gather_relmask")


def gather_block(block, tokens, pos, groups, H, D):
    """Global block with keys/values gathered first. groups: [(query slice or None, kept idx)]."""
    B, N, _ = tokens.shape
    x = block.norm1(tokens)
    q, k, v = block.attn.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = block.attn.q_norm(q), block.attn.k_norm(k)
    if block.attn.rope is not None and pos is not None:
        q, k = block.attn.rope(q, pos), block.attn.rope(k, pos)
    km, vm = k.mean(2, keepdim=True), v.mean(2, keepdim=True)       # stand-in mean token
    if len(groups) == 1 and groups[0][0] is None:
        idx = groups[0][1]
        out = F.scaled_dot_product_attention(q, torch.cat([k[:, :, idx], km], 2), torch.cat([v[:, :, idx], vm], 2))
    else:
        out = torch.empty(q.shape, device=q.device, dtype=v.dtype)
        for qs, idx in groups:
            out[:, :, qs] = F.scaled_dot_product_attention(
                q[:, :, qs], torch.cat([k[:, :, idx], km], 2), torch.cat([v[:, :, idx], vm], 2))
    out = block.attn.proj(out.transpose(1, 2).reshape(B, N, -1))
    tokens = tokens + block.ls1(out)
    return tokens + block.ls2(block.mlp(block.norm2(tokens)))


def colmask_groups(keep, S, P, psi, device):
    """Per layer: frame 0, every special token, and the kept patch positions of frames 1..S-1."""
    frame0 = torch.arange(P, device=device)
    spec = (torch.arange(1, S, device=device)[:, None] * P + torch.arange(psi, device=device)).flatten()
    out = {}
    for l in range(keep.shape[0]):
        kept = torch.nonzero(keep[l]).flatten() + psi
        other = (torch.arange(1, S, device=device)[:, None] * P + kept[None]).flatten()
        out[l] = [(None, torch.cat([frame0, spec, other]).sort().values)]
    return out


def relmask_groups(keep, S, P, psi, grid, buckets, start, depth, device):
    """Per query frame s: frame 0, own frame, specials, and key frame t's cells kept for |s - t|."""
    gh, gw = grid
    th, tw = keep.shape[1], keep.shape[2]
    pk = torch.arange(gh * gw, device=device)
    tile = (pk // gw % th) * tw + (pk % gw % tw)
    frame0 = torch.arange(P, device=device)
    spec = (torch.arange(1, S, device=device)[:, None] * P + torch.arange(psi, device=device)).flatten()
    groups = []
    for s in range(S):
        parts = [frame0, spec] + ([torch.arange(s * P, (s + 1) * P, device=device)] if s > 0 else [])
        for t in range(1, S):
            if t != s:
                b = sum(abs(s - t) > u for u in buckets)
                parts.append(t * P + torch.nonzero(keep[b].flatten()[tile]).flatten() + psi)
        groups.append((slice(s * P, (s + 1) * P), torch.unique(torch.cat(parts))))
    return {l: groups for l in range(start, depth)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--colmask", default="outputs/colmask/d1_b25_learned.npy", help="bool [24, h*w]")
    ap.add_argument("--relmask", default="outputs/colmask/relmask_b25.npy", help="bool [n_buckets, th, tw]")
    ap.add_argument("--relmask_buckets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--relmask_start", type=int, default=8)
    ap.add_argument("--seq", default="test/pose/dataset3/keyframe4", help="under data/train/scared")
    ap.add_argument("--n_frames", type=int, default=64)
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix of the output json")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    names = args.settings.split(",")
    assert all(n in SETTINGS for n in names), names

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    img_dir = os.path.join(data_path("train", "scared"), args.seq, "image_left")
    paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))[: args.n_frames]]
    images = load_and_preprocess_images(paths, mode="crop").cuda()[None]
    model = load_vggt_for_eval(args.ckpt)
    agg = model.aggregator
    S, (Hpx, Wpx) = images.shape[1], images.shape[-2:]
    psi = agg.patch_start_idx
    grid = (Hpx // 14, Wpx // 14)
    P = psi + grid[0] * grid[1]
    blk0 = agg.global_blocks[0].attn
    H, D = blk0.num_heads, blk0.head_dim
    orig = agg._process_global_attention
    events = []

    col = torch.as_tensor(np.load(args.colmask)).cuda() if any("colmask" in n for n in names) else None
    rel = torch.as_tensor(np.load(args.relmask)).cuda() if any("relmask" in n for n in names) else None
    gathers = {}
    if "gather_colmask" in names:
        gathers["gather_colmask"] = colmask_groups(col, S, P, psi, "cuda")
    if "gather_relmask" in names:
        gathers["gather_relmask"] = relmask_groups(rel, S, P, psi, grid, args.relmask_buckets,
                                                  args.relmask_start, agg.depth, "cuda")

    def make_pga(gather):
        def pga(self, tokens, B_, S_, P_, C, global_idx, pos=None, **kw):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            if gather is not None and global_idx in gather:
                t = tokens.view(B_, S_ * P_, C)
                p = pos.view(B_, S_ * P_, -1) if pos is not None else None
                t = gather_block(self.global_blocks[global_idx], t, p, gather[global_idx], H, D)
                res = (t, global_idx + 1, [t.view(B_, S_, P_, C)])
            else:
                res = orig(tokens, B_, S_, P_, C, global_idx, pos=pos, **kw)
            e1.record(); events.append((e0, e1))
            return res
        return pga

    def bench(name):
        agg.enable_colmask = False
        agg.enable_relmask = False
        if name == "flex_colmask":
            agg.set_colmask_fixed(col.cpu().numpy(), diag=True, mean=True, grid=grid)
        elif name == "flex_relmask":
            agg.set_relmask_fixed(rel.cpu().numpy(), buckets=args.relmask_buckets,
                                  start=args.relmask_start, grid=grid)
        agg._process_global_attention = types.MethodType(make_pga(gathers.get(name)), agg)
        whole, glob = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(args.warm + args.n):
                events.clear()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); model(images=images); e1.record(); torch.cuda.synchronize()
                if i >= args.warm:
                    whole.append(e0.elapsed_time(e1))
                    glob.append(sum(a.elapsed_time(b) for a, b in events))
        return dict(whole_ms=float(np.median(whole)), global_ms=float(np.median(glob)))

    res = dict(ckpt=args.ckpt, seq=args.seq, n_frames=S, colmask=args.colmask if col is not None else None,
               relmask=args.relmask if rel is not None else None,
               gpu=torch.cuda.get_device_name(), kept_key_fraction={})
    N = S * P
    for k, g in gathers.items():
        res["kept_key_fraction"][k] = float(np.mean([np.mean([len(ix) for _, ix in grp]) for grp in g.values()]) / N)
    for name in names:
        res[name] = bench(name)
        print(name, res[name], flush=True)
    if "plain" in res:
        for name in names:
            if name != "plain":
                res[name]["speedup_whole"] = res["plain"]["whole_ms"] / res[name]["whole_ms"]
                res[name]["speedup_global"] = res["plain"]["global_ms"] / res[name]["global_ms"]
        res["global_share_plain"] = res["plain"]["global_ms"] / res["plain"]["whole_ms"]
    out = args.out_dir or default_output_dir(args.ckpt, "speed")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"speed_c{S}{args.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
