#!/usr/bin/env python3
"""Precompute domain-invariant DYNAMIC MASKS for PointOdyssey via RAFT flow residual.

Faithful to docs/method.md §3.4 (m_geo) (and identical to the Sintel eval convention):
    m*_raft(t) = 1[ || f^gt - f^cam || > thr ]
      f^gt  = RAFT optical flow  frame t -> t+1              (observed, domain-invariant)
      f^cam = camera-induced ego-flow from GT depth + GT pose (compute_ego_flow)

A pixel whose observed flow disagrees with the flow the camera alone would induce is moving
in the world -> dynamic. This replaces PointOdyssey's native appearance mask (which marks every
foreground agent, moving or not) with a motion-defined, geometry-grounded label.

Output (drop-in for the native mask): <seq>/dynmask_raft/dyn_{fid:05d}.png  (uint8 {0,255}, native res)
The dataset loads it exactly like masks/ (rides the same augmentation via RNG replay).

Run (from the repo root):
  python -m pipeline.data.preprocess.po_raft_dynmask --split train            # all train seqs
  python -m pipeline.data.preprocess.po_raft_dynmask --split train --seqs ani --vis   # one seq + viz
"""
import os, os.path as osp, glob, argparse

import numpy as np, cv2, torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pipeline.data.motion_mask import compute_ego_flow
from pipeline.data.paths import data_path

DEPTH_MAX = 1000.0  # matches PointOdysseyDataset.depth_max
PROC_W, PROC_H = 480, 272  # RAFT processing res (both divisible by 8); ~half native, gate is patch-level


def load_depth(path):
    d16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    d = d16.astype(np.float32) / 65535.0 * DEPTH_MAX
    d[d >= DEPTH_MAX] = 0.0  # sky/invalid -> 0
    return d


@torch.no_grad()
def process_seq(seq_dir, model, tf, device, thr, gap=5, save_vis=False):
    rgb_paths = sorted(glob.glob(osp.join(seq_dir, "rgbs", "rgb_*.jpg")))
    if len(rgb_paths) < 2:
        return "too_few_frames"
    anno = np.load(osp.join(seq_dir, "anno.npz"), allow_pickle=True)
    K_all = anno["intrinsics"].astype(np.float32)  # (F,3,3) native res
    ext_all = anno["extrinsics"].astype(np.float32)  # (F,4,4) world->cam

    out_dir = osp.join(seq_dir, "dynmask_raft")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = osp.join(seq_dir, "dynmask_raft_vis")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    nat = cv2.imread(rgb_paths[0]).shape[:2]  # (H,W) native
    natH, natW = nat
    sx, sy = PROC_W / natW, PROC_H / natH

    def read_proc(p):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        # NOTE: keep uint8 - weights.transforms() expects uint8 (it divides by 255 then
        # normalises to [-1,1]). Passing float [0,255] silently yields garbage flow.
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,H,W) uint8

    F = min(len(rgb_paths), len(K_all))
    for t in range(F):
        fid = int(osp.basename(rgb_paths[t]).split("_")[1].split(".")[0])
        # residual over a gap t -> t2 (t2 = t+gap). Larger gap lets slow motion accumulate above the
        # static noise floor (SNR ∝ gap), so thr separates slow-moving foreground cleanly; near the
        # tail the gap shrinks, and the last frame (t2==t) is all-static.
        t2 = min(t + gap, F - 1)
        if t2 == t:
            cv2.imwrite(osp.join(out_dir, f"dyn_{fid:05d}.png"), np.zeros((natH, natW), np.uint8))
            continue

        i1, i2 = read_proc(rgb_paths[t]), read_proc(rgb_paths[t2])
        b1, b2 = tf(i1[None], i2[None])
        flow = model(b1.to(device), b2.to(device))[-1][0].cpu().numpy()  # (2,H,W) proc px

        depth = cv2.resize(load_depth(osp.join(seq_dir, "depths", f"depth_{fid:05d}.png")), (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)
        K = K_all[t].copy()
        K[0] *= sx
        K[1] *= sy
        ego = compute_ego_flow(depth, K, ext_all[t][:3], ext_all[t2][:3])  # (H,W,2) proc px

        res = flow.transpose(1, 2, 0) - ego  # (H,W,2) proc px
        res[..., 0] *= natW / PROC_W
        res[..., 1] *= natH / PROC_H  # -> native px units
        resid = np.linalg.norm(res, axis=-1)  # (H,W) native px
        resid[depth <= 0] = 0.0  # can't judge sky/invalid -> static
        mask_proc = (resid > thr).astype(np.uint8) * 255
        mask = cv2.resize(mask_proc, (natW, natH), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(osp.join(out_dir, f"dyn_{fid:05d}.png"), mask)

        if save_vis and t < 6:
            rgb = cv2.resize(cv2.imread(rgb_paths[t]), (natW, natH))
            native_mask_p = osp.join(seq_dir, "masks", f"mask_{fid:05d}.png")
            nm = cv2.imread(native_mask_p, cv2.IMREAD_UNCHANGED)
            nm = (nm.sum(-1) > 0).astype(np.uint8) * 255 if nm is not None and nm.ndim == 3 else np.zeros((natH, natW), np.uint8)
            nm = cv2.resize(nm, (natW, natH), interpolation=cv2.INTER_NEAREST)
            resid_col = cv2.applyColorMap(np.clip(resid / max(thr * 3, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            resid_col = cv2.resize(resid_col, (natW, natH))
            row = np.concatenate([rgb, cv2.cvtColor(nm, cv2.COLOR_GRAY2BGR), resid_col, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=1)
            cv2.imwrite(osp.join(vis_dir, f"cmp_{fid:05d}.png"), row)
    return f"ok({F})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--po_dir", default=data_path("train", "point_odyssey"))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of seq names; default all")
    ap.add_argument("--thr", type=float, default=2.0, help="flow-residual threshold in NATIVE px")
    ap.add_argument("--gap", type=int, default=5, help="frame gap delta for residual (t -> t+delta); larger lifts slow motion above noise")
    ap.add_argument("--vis", action="store_true", help="save comparison viz for first frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    tf = weights.transforms()

    seq_dirs = sorted(glob.glob(osp.join(args.po_dir, args.split, "*/")))
    if args.seqs:
        seq_dirs = [s for s in seq_dirs if osp.basename(s.rstrip("/")) in args.seqs]
    print(f"{len(seq_dirs)} sequences | thr={args.thr}px | gap={args.gap} | proc={PROC_W}x{PROC_H}")
    from tqdm import tqdm

    for s in tqdm(seq_dirs):
        name = osp.basename(s.rstrip("/"))
        done_flag = osp.join(s, "dynmask_raft", ".done")
        if osp.isfile(done_flag) and not args.overwrite:
            continue
        st = process_seq(s, model, tf, device, args.thr, gap=args.gap, save_vis=args.vis)
        if st.startswith("ok"):
            open(done_flag, "w").close()
        tqdm.write(f"  {name}: {st}")
    print("DONE.")


if __name__ == "__main__":
    main()
