#!/usr/bin/env python3
"""Precompute domain-invariant DYNAMIC MASKS for Spring via RAFT flow residual.

Same convention as po_raft_dynmask.py (dyn_vggt_method_v3 §5.3a):
    m*_raft(t) = 1[ || f^gt - f^cam || > thr ]
      f^gt  = RAFT optical flow  frame t -> t+delta            (observed, domain-invariant)
      f^cam = camera-induced ego-flow from GT depth + GT pose  (compute_ego_flow)

Spring is a DYNAMIC-scene dataset (animated characters, see spring.py header) but ships no
per-pixel dynamic segmentation GT - only dense stereo disparity + camera poses. This is exactly
the case §5.3(a) targets: derive the label geometrically from GT depth+pose vs. observed
(RAFT) flow, since there is no instance/track GT here to fall back on (unlike PointOdyssey's
m*_inst, §5.3b).

Output (drop-in for future dataset-loader wiring): <seq_dir>/dynmask_raft/dyn_{frame_idx:04d}.png
(uint8 {0,255}, native res, 1-based frame_idx to match frame_left_XXXX.png/disp1_left_XXXX.dsp5).
NOTE: SpringDataset does not currently load this file - motion_mask is intentionally absent from
its get_data() output. Wiring it in is a separate follow-up.

Run (from training/):
  python data/preprocess/spring_raft_dynmask.py                       # all train seqs
  python data/preprocess/spring_raft_dynmask.py --seqs 0001 --vis     # one seq + viz
"""
import os, os.path as osp, glob, argparse, sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))  # training/
import numpy as np, cv2, torch, h5py
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from eval.motion_mask import compute_ego_flow

BASELINE = 0.065  # Spring stereo baseline, metres (matches SpringDataset.BASELINE)
DEPTH_MAX = 200.0  # matches SpringDataset.depth_max
PROC_W, PROC_H = 960, 544  # RAFT processing res (divisible by 8, ~half native 1920x1080)


def _read_dsp5(path):
    with h5py.File(path, "r") as f:
        return f["disparity"][()].astype(np.float32)


def disp_to_depth(disp_2x, fx):
    """2x-res disparity -> depth at image resolution (mirrors SpringDataset._disp_to_depth)."""
    disp = disp_2x[::2, ::2]
    valid = np.isfinite(disp) & (disp > 0.01)
    depth = np.zeros_like(disp)
    depth[valid] = fx * BASELINE / disp[valid]
    depth[depth > DEPTH_MAX] = 0.0
    return depth


@torch.no_grad()
def process_seq(seq_dir, model, tf, device, thr, gap=5, save_vis=False):
    frame_paths = sorted(glob.glob(osp.join(seq_dir, "frame_left", "frame_left_*.png")))
    intri_path = osp.join(seq_dir, "cam_data", "intrinsics.txt")
    extri_path = osp.join(seq_dir, "cam_data", "extrinsics.txt")
    if len(frame_paths) < 2 or not (osp.isfile(intri_path) and osp.isfile(extri_path)):
        return "too_few_frames_or_no_cam"
    all_intri = np.loadtxt(intri_path, dtype=np.float64)  # (N, 4): fx fy cx cy
    all_extri = np.loadtxt(extri_path, dtype=np.float64)  # (N, 16): flattened 4x4 c2w

    out_dir = osp.join(seq_dir, "dynmask_raft")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = osp.join(seq_dir, "dynmask_raft_vis")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    natH, natW = cv2.imread(frame_paths[0]).shape[:2]
    sx, sy = PROC_W / natW, PROC_H / natH

    def w2c(fid):
        c2w = all_extri[fid].reshape(4, 4).copy()
        return np.linalg.inv(c2w)[:3]  # (3,4)

    def read_proc(p):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        # NOTE: keep uint8 - weights.transforms() expects uint8 (divides by 255, normalises to
        # [-1,1]); passing float [0,255] silently yields garbage flow.
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    F = min(len(frame_paths), len(all_intri), len(all_extri))
    for t in range(F):
        frame_idx = t + 1  # Spring frame indices are 1-based
        # residual over a gap t -> t2 (t2 = t+gap); see po_raft_dynmask.py for the
        # SNR-vs-gap rationale. Last frame (t2==t) is trivially all-static.
        t2 = min(t + gap, F - 1)
        if t2 == t:
            cv2.imwrite(osp.join(out_dir, f"dyn_{frame_idx:04d}.png"), np.zeros((natH, natW), np.uint8))
            continue

        i1, i2 = read_proc(frame_paths[t]), read_proc(frame_paths[t2])
        b1, b2 = tf(i1[None], i2[None])
        flow = model(b1.to(device), b2.to(device))[-1][0].cpu().numpy()  # (2,H,W) proc px

        disp_path = osp.join(seq_dir, "disp1_left", f"disp1_left_{frame_idx:04d}.dsp5")
        if not osp.isfile(disp_path):
            cv2.imwrite(osp.join(out_dir, f"dyn_{frame_idx:04d}.png"), np.zeros((natH, natW), np.uint8))
            continue
        fx = all_intri[t, 0]
        depth = disp_to_depth(_read_dsp5(disp_path), fx)
        depth = cv2.resize(depth, (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)

        K = np.array(
            [[all_intri[t, 0] * sx, 0.0, all_intri[t, 2] * sx], [0.0, all_intri[t, 1] * sy, all_intri[t, 3] * sy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        ego = compute_ego_flow(depth, K, w2c(t), w2c(t2))  # (H,W,2) proc px

        res = flow.transpose(1, 2, 0) - ego  # (H,W,2) proc px
        res[..., 0] *= natW / PROC_W
        res[..., 1] *= natH / PROC_H  # -> native px units
        resid = np.linalg.norm(res, axis=-1)  # (H,W) native px
        resid[depth <= 0] = 0.0  # can't judge invalid depth -> static
        mask_proc = (resid > thr).astype(np.uint8) * 255
        mask = cv2.resize(mask_proc, (natW, natH), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(osp.join(out_dir, f"dyn_{frame_idx:04d}.png"), mask)

        if save_vis and t < 6:
            rgb = cv2.imread(frame_paths[t])
            resid_col = cv2.applyColorMap(np.clip(resid / max(thr * 3, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            resid_col = cv2.resize(resid_col, (natW, natH))
            row = np.concatenate([rgb, resid_col, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=1)
            cv2.imwrite(osp.join(vis_dir, f"cmp_{frame_idx:04d}.png"), row)
    return f"ok({F})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spring_dir", default="/media/cvml-75/ssd2t1/data/spring")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of seq names; default all")
    ap.add_argument("--thr", type=float, default=2.0, help="flow-residual threshold in NATIVE px")
    ap.add_argument("--gap", type=int, default=5, help="frame gap delta for residual (t -> t+delta)")
    ap.add_argument("--vis", action="store_true", help="save comparison viz for first frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    tf = weights.transforms()

    split_dir = osp.join(args.spring_dir, args.split)
    seq_dirs = sorted(glob.glob(osp.join(split_dir, "*/")))
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
