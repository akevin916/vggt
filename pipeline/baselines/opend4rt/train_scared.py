#!/usr/bin/env python3
"""Fine-tune OpenD4RT on SCARED train (baseline B), single GPU.

Matched to our VGGT baseline (scared_cam_vanilla): same split (train keyframes), 10 epochs x
1500 steps, and the same checkpoint rule -- after every epoch score test/pose and keep the
best snippet ATE as best_ate.pt (our runs do it at c64; OpenD4RT at c48, its clip limit).
Freezing matches VGGT's share: VGGT freezes patch_embed + all 24 frame blocks + global
blocks 0-7 (32 of 48 aggregator blocks); here patch_embed + encoder blocks 0..26 (27 of 40).

Everything else is OpenD4RT's own recipe (configs/train_effective.yaml): loss D4RTLoss with
its weights, AdamW wd 0.03, grad-clip 10, 500 warmup steps to 4e-6 then cosine to 4e-7,
fp16 autocast + GradScaler, 4096 queries per 48-frame clip, batch 1 clip per step.

Run (from the repo root):
  python -m pipeline.baselines.opend4rt.train_scared --exp opend4rt_scared_ft
  python -m pipeline.baselines.opend4rt.train_scared --exp opend4rt_scared_smoke --epochs 1 \
      --steps_per_epoch 60 --no_eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from pipeline.data.paths import data_path
from pipeline.eval.d4rt_infer import (DEFAULT_CKPT, OPEN_D4RT_DIR, _import_open_d4rt,
                                      load_d4rt_for_eval)
from pipeline.baselines.opend4rt.scared_dataset import ScaredD4RTDataset

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def lr_at(step, total, warmup=500, peak=4e-6, final=4e-7):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * p))


def freeze(model, n_frozen_blocks):
    for p in model.encoder.patch_embed.parameters():
        p.requires_grad_(False)
    for blk in model.encoder.blocks[:n_frozen_blocks]:
        for p in blk.parameters():
            p.requires_grad_(False)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    return n_tr, n_all


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    return x


def eval_test_pose(model, exp, epoch, device):
    """test/pose c48o16 snippet ATE through the benchmark path (eval_d4rt_scared)."""
    from pipeline.benchmark import eval_scared as es
    from pipeline.eval.d4rt_infer import infer_sequence_d4rt
    es.infer_sequence = lambda m, paths, device="cuda", **kw: infer_sequence_d4rt(
        m, paths, device=device, want_depth=False)
    out = os.path.join(REPO_DIR, "logs", exp, "pose_eval", f"epoch_{epoch}")
    args = SimpleNamespace(ckpt=f"logs/{exp}/live", scared_root=data_path("train", "scared"),
                           split="test/pose", seqs=None, n_frames=0, chunk_size=48, overlap=16,
                           max_depth=None, depth_protocol="monst3r", single_view=False,
                           no_depth=True, device=device, gate_mode="predicted", key_mask=None,
                           img_size=518, fid_step=1, save_pred=False, out_dir=out)
    model.eval()
    res = es.evaluate(args, model=model)
    model.train()
    per = {s: v["snippet_ate"] for s, v in res["pose"]["per_seq"].items() if "snippet_ate" in v}
    return res["pose"]["mean"].get("snippet_ate", float("nan")), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--init_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps_per_epoch", type=int, default=1500)
    ap.add_argument("--frozen_blocks", type=int, default=27)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--no_eval", action="store_true", help="smoke: skip per-epoch test/pose eval")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(42)
    log_dir = os.path.join(REPO_DIR, "logs", args.exp)
    ckpt_dir = os.path.join(log_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    # load_d4rt_for_eval builds the model from the model.yaml next to the ckpt
    import shutil
    shutil.copy(os.path.join(os.path.dirname(args.init_ckpt), "model.yaml"), ckpt_dir)
    logf = open(os.path.join(log_dir, "log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    _, load_yaml_config, _, _ = _import_open_d4rt()
    from src.losses import build_loss
    train_cfg = load_yaml_config(os.path.join(OPEN_D4RT_DIR, "configs", "train_effective.yaml"))
    loss_fn = build_loss("d4rt", train_cfg)

    model = load_d4rt_for_eval(args.init_ckpt, device=args.device).train()
    n_tr, n_all = freeze(model, args.frozen_blocks)
    log(f"args {vars(args)}")
    log(f"trainable {n_tr/1e6:.0f}M / {n_all/1e6:.0f}M params")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr_at(0, 1), weight_decay=0.03)
    scaler = torch.amp.GradScaler("cuda")
    total = args.epochs * args.steps_per_epoch

    ds = ScaredD4RTDataset(data_path("train", "scared"), "train", length=args.steps_per_epoch)
    log(f"{len(ds.seqs)} train keyframes")
    best, step = float("inf"), 0
    for epoch in range(args.epochs):
        torch.manual_seed(42 + epoch)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=False)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            batch = to_dev(batch, args.device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(batch)
                loss, metrics = loss_fn(out, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % args.log_every == 0 or step == 1:
                parts = " ".join(f"{k} {float(v):.4f}" for k, v in metrics.items()
                                 if k.startswith("loss_") and torch.is_tensor(v) and v.numel() == 1)
                log(f"ep {epoch} step {step}/{total} loss {loss.item():.4f} {parts} gn {gn.item():.2f} "
                    f"lr {lr_at(step - 1, total):.2e} peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB "
                    f"{(time.time()-t0)/max(1, step - epoch*args.steps_per_epoch):.2f}s/it")
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "step": step},
                   os.path.join(ckpt_dir, "last.pt"))
        if args.no_eval:
            continue
        ate, per = eval_test_pose(model, args.exp, epoch, args.device)
        log(f"epoch {epoch} test/pose c48 snippet_ate {ate:.4f} {json.dumps(per)}")
        if ate < best:
            best = ate
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "step": step,
                        "snippet_ate": ate}, os.path.join(ckpt_dir, "best_ate.pt"))
            log(f"new best_ate.pt (epoch {epoch})")
    log("done")


if __name__ == "__main__":
    main()
