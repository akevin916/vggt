#!/usr/bin/env python3
"""Precompute domain-invariant DYNAMIC MASKS for Waymo via RAFT flow residual.

Same convention as po_raft_dynmask.py (docs/method.md §3.4 (m_geo)):
    m*_raft(t) = 1[ || f^gt - f^cam || > thr ]
      f^gt  = RAFT optical flow  frame t -> t+delta            (observed, domain-invariant)
      f^cam = camera-induced ego-flow from GT depth + GT pose  (compute_ego_flow)

Waymo is a real-world DYNAMIC-scene dataset (vehicles, pedestrians - see waymo.py header) with
sparse LiDAR depth and no per-pixel dynamic segmentation GT. Residual is only evaluable at
pixels with a valid (non-zero) LiDAR return, exactly like PointOdyssey's depth<=0 -> static
fallback; the resulting mask will be comparatively sparse away from those points but should
still concentrate correctly around moving vehicles/pedestrians.

Each (segment, camera) pair is one sequence (matches WaymoDataset). Frame numbers within a
sequence are NOT guaranteed contiguous (dropped frames), so the delta-gap is applied over the
sorted list of AVAILABLE frame indices for that camera, not raw frame-number arithmetic
(mirrors WaymoDataset's own local_ids -> frame_ids translation).

Output (drop-in for future dataset-loader wiring, one dynmask_raft/ per segment shared across
its cameras): <seg_dir>/dynmask_raft/dyn_{fid:05d}_{cam_id}.png (uint8 {0,255}, native res).
NOTE: WaymoDataset does not currently load this file - motion_mask is intentionally absent from
its get_data() output. Wiring it in is a separate follow-up.

Run (from the repo root):
  python -m pipeline.data.preprocess.waymo_raft_dynmask                              # all segments/cams
  python -m pipeline.data.preprocess.waymo_raft_dynmask --cameras 1 --vis            # front camera only
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # must precede cv2 import (EXR depth)
import os.path as osp, glob, argparse
from collections import defaultdict

import numpy as np, cv2, torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pipeline.data.motion_mask import compute_ego_flow
from pipeline.data.paths import data_path
from pipeline.data.datasets.waymo import _read_waymo_depth

DEPTH_MAX = 80.0  # matches WaymoDataset.depth_max (LiDAR valid range)


def _round8(x):
    # RAFT downsamples by 8 internally and needs >=16 cells per side in the correlation
    # pyramid, i.e. >=128px (see torchvision raft.py build_pyramid). Waymo's side cameras
    # (4/5) are only 236px tall, so halving them (~112px) undershoots this floor.
    return max(128, (x // 2 // 8) * 8)


@torch.no_grad()
def process_seq(seg_dir, cam_id, frame_ids, model, tf, device, thr, gap=5, save_vis=False):
    if len(frame_ids) < 2:
        return "too_few_frames"

    out_dir = osp.join(seg_dir, "dynmask_raft")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = osp.join(seg_dir, "dynmask_raft_vis")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    img0 = cv2.imread(osp.join(seg_dir, f"{frame_ids[0]:05d}_{cam_id}.jpg"))
    if img0 is None:
        return "no_first_frame"
    natH, natW = img0.shape[:2]
    PROC_W, PROC_H = _round8(natW), _round8(natH)
    sx, sy = PROC_W / natW, PROC_H / natH

    def read_proc(fid):
        img = cv2.cvtColor(cv2.imread(osp.join(seg_dir, f"{fid:05d}_{cam_id}.jpg")), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        # NOTE: keep uint8 - weights.transforms() expects uint8 (divides by 255, normalises to
        # [-1,1]); passing float [0,255] silently yields garbage flow.
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def w2c(fid):
        cam_data = np.load(osp.join(seg_dir, f"{fid:05d}_{cam_id}.npz"))
        c2w = cam_data["cam2world"].astype(np.float64)
        return np.linalg.inv(c2w)[:3]  # (3,4); absolute translation cancels in the relative
        # transform inside compute_ego_flow, so no need for the
        # mean-centering WaymoDataset applies for training batches.

    F = len(frame_ids)
    for t in range(F):
        fid = int(frame_ids[t])
        # residual over a gap in the AVAILABLE-frame index space (t -> t2), not raw frame number,
        # since Waymo sequences can have dropped frames. Last frame (t2==t) is trivially static.
        t2 = min(t + gap, F - 1)
        if t2 == t:
            cv2.imwrite(osp.join(out_dir, f"dyn_{fid:05d}_{cam_id}.png"), np.zeros((natH, natW), np.uint8))
            continue
        fid2 = int(frame_ids[t2])

        i1, i2 = read_proc(fid), read_proc(fid2)
        b1, b2 = tf(i1[None], i2[None])
        flow = model(b1.to(device), b2.to(device))[-1][0].cpu().numpy()  # (2,H,W) proc px

        depth = _read_waymo_depth(osp.join(seg_dir, f"{fid:05d}_{cam_id}.exr"))
        if depth is None:
            cv2.imwrite(osp.join(out_dir, f"dyn_{fid:05d}_{cam_id}.png"), np.zeros((natH, natW), np.uint8))
            continue
        depth[depth > DEPTH_MAX] = 0.0
        depth = cv2.resize(depth, (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)

        K_native = np.load(osp.join(seg_dir, f"{fid:05d}_{cam_id}.npz"))["intrinsics"].astype(np.float64)
        K = K_native.copy()
        K[0] *= sx
        K[1] *= sy
        ego = compute_ego_flow(depth, K, w2c(fid), w2c(fid2))  # (H,W,2) proc px

        res = flow.transpose(1, 2, 0) - ego  # (H,W,2) proc px
        res[..., 0] *= natW / PROC_W
        res[..., 1] *= natH / PROC_H  # -> native px units
        resid = np.linalg.norm(res, axis=-1)  # (H,W) native px
        resid[depth <= 0] = 0.0  # sparse LiDAR: no depth -> can't judge -> static
        mask_proc = (resid > thr).astype(np.uint8) * 255
        mask = cv2.resize(mask_proc, (natW, natH), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(osp.join(out_dir, f"dyn_{fid:05d}_{cam_id}.png"), mask)

        if save_vis and t < 6:
            rgb = cv2.imread(osp.join(seg_dir, f"{fid:05d}_{cam_id}.jpg"))
            valid = (depth > 0).astype(np.uint8) * 255
            valid = cv2.resize(valid, (natW, natH), interpolation=cv2.INTER_NEAREST)
            resid_col = cv2.applyColorMap(np.clip(resid / max(thr * 3, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            resid_col = cv2.resize(resid_col, (natW, natH))
            row = np.concatenate([rgb, cv2.cvtColor(valid, cv2.COLOR_GRAY2BGR), resid_col, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=1)
            cv2.imwrite(osp.join(vis_dir, f"cmp_{fid:05d}_{cam_id}.png"), row)
    return f"ok({F})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waymo_dir", default=data_path("train", "waymo_processed"))
    ap.add_argument("--cameras", nargs="*", type=int, default=None, help="subset of camera ids 1-5; default all")
    ap.add_argument("--segments", nargs="*", default=None, help="subset of segment dir names; default all")
    ap.add_argument("--thr", type=float, default=2.0, help="flow-residual threshold in NATIVE px")
    ap.add_argument("--gap", type=int, default=5, help="frame gap delta (in available-frame index space)")
    ap.add_argument("--vis", action="store_true", help="save comparison viz for first frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    tf = weights.transforms()

    cameras = args.cameras if args.cameras else [1, 2, 3, 4, 5]
    seg_dirs = sorted(d for d in glob.glob(osp.join(args.waymo_dir, "*")) if osp.isdir(d))
    if args.segments:
        seg_dirs = [d for d in seg_dirs if osp.basename(d) in args.segments]

    jobs = []
    for seg_dir in seg_dirs:
        cam_frames = defaultdict(list)
        for fname in os.listdir(seg_dir):
            if not fname.endswith(".jpg"):
                continue
            stem = fname[:-4]
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            try:
                frame_idx, cam_id = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if cam_id in cameras and osp.isfile(osp.join(seg_dir, f"{frame_idx:05d}_{cam_id}.npz")):
                cam_frames[cam_id].append(frame_idx)
        for cam_id, frame_ids in cam_frames.items():
            jobs.append((seg_dir, cam_id, np.array(sorted(frame_ids), dtype=np.int64)))

    print(f"{len(jobs)} (segment, camera) sequences | thr={args.thr}px | gap={args.gap}")
    from tqdm import tqdm

    for seg_dir, cam_id, frame_ids in tqdm(jobs):
        name = f"{osp.basename(seg_dir)}__cam{cam_id}"
        done_flag = osp.join(seg_dir, "dynmask_raft", f".done_cam{cam_id}")
        if osp.isfile(done_flag) and not args.overwrite:
            continue
        st = process_seq(seg_dir, cam_id, frame_ids, model, tf, device, args.thr, gap=args.gap, save_vis=args.vis)
        if st.startswith("ok"):
            open(done_flag, "w").close()
        tqdm.write(f"  {name}: {st}")
    print("DONE.")


if __name__ == "__main__":
    main()
