#!/usr/bin/env python3
"""Precompute domain-invariant DYNAMIC MASKS for TartanAir via RAFT flow residual.

Same convention as po_raft_dynmask.py (docs/method.md §5.3a):
    m*_raft(t) = 1[ || f^gt - f^cam || > thr ]
      f^gt  = RAFT optical flow  frame t -> t+delta            (observed, domain-invariant)
      f^cam = camera-induced ego-flow from GT depth + GT pose  (compute_ego_flow)

TartanAir is used as a STATIC negative-example dataset (see tartanair.py header): every scene
is nominally camera-only motion, so this script is expected to mostly produce all-zero masks.
Running it rather than assuming all-zero is itself the point: if some trajectories trip the
mask (e.g. via the fast-camera / static-background residual blow-up documented in
method.md §5.3(a) - large camera baseline + depth error inflates the "static"
residual above threshold), that's actionable (raise --thr, or exclude the trajectory) rather
than silently trusting "static by construction".

Output (drop-in, parallel to PointOdyssey's dynmask_raft/): <traj_dir>/dynmask_raft/dyn_{fid:06d}.png
(uint8 {0,255}, native 640x640 res). NOTE: TartanAirDataset does not currently load this file -
motion_mask is intentionally absent from its get_data() output (static dataset). Wiring this
in (if the masks turn out non-trivial) is a separate follow-up.

Run (from training/):
  python data/preprocess/tartanair_raft_dynmask.py                              # all train trajs
  python data/preprocess/tartanair_raft_dynmask.py --envs abandonedfactory --vis
"""
import os, os.path as osp, glob, argparse, sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))  # training/
import numpy as np, cv2, torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from data.motion_mask import compute_ego_flow
from data.paths import data_path
from data.datasets.tartanair import TartanAirDataset

DEPTH_MAX = 1000.0  # matches TartanAirDataset.depth_max
NATIVE = 640  # fixed native resolution (both dims), TartanAir V1
PROC = 320  # RAFT processing res (divisible by 8, half native)
_FX = _FY = TartanAirDataset._FX
_CX = _CY = TartanAirDataset._CX


def load_depth(path):
    d = np.load(path).astype(np.float32)
    d[d >= DEPTH_MAX] = 0.0
    d[~np.isfinite(d)] = 0.0
    return d


@torch.no_grad()
def process_traj(traj_dir, model, tf, device, thr, gap=5, save_vis=False):
    frame_paths = sorted(glob.glob(osp.join(traj_dir, "image_left", "*_left.png")))
    pose_path = osp.join(traj_dir, "pose_left.txt")
    if len(frame_paths) < 2 or not osp.isfile(pose_path):
        return "too_few_frames_or_no_pose"
    all_poses = np.loadtxt(pose_path, dtype=np.float64)  # (N, 7): x y z qx qy qz qw

    out_dir = osp.join(traj_dir, "dynmask_raft")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = osp.join(traj_dir, "dynmask_raft_vis")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    s = PROC / NATIVE
    K_proc = np.array([[_FX * s, 0.0, _CX * s], [0.0, _FY * s, _CY * s], [0.0, 0.0, 1.0]], dtype=np.float32)

    def read_proc(p):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (PROC, PROC), interpolation=cv2.INTER_LINEAR)
        # NOTE: keep uint8 - weights.transforms() expects uint8 (divides by 255, normalises to
        # [-1,1]); passing float [0,255] silently yields garbage flow.
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    F = min(len(frame_paths), len(all_poses))
    for t in range(F):
        fid = int(osp.basename(frame_paths[t]).split("_")[0])
        # residual over a gap t -> t2 (t2 = t+gap); see po_raft_dynmask.py for the
        # SNR-vs-gap rationale. Last frame (t2==t) is trivially all-static.
        t2 = min(t + gap, F - 1)
        if t2 == t:
            cv2.imwrite(osp.join(out_dir, f"dyn_{fid:06d}.png"), np.zeros((NATIVE, NATIVE), np.uint8))
            continue

        i1, i2 = read_proc(frame_paths[t]), read_proc(frame_paths[t2])
        b1, b2 = tf(i1[None], i2[None])
        flow = model(b1.to(device), b2.to(device))[-1][0].cpu().numpy()  # (2,H,W) proc px

        fid2 = int(osp.basename(frame_paths[t2]).split("_")[0])
        depth = cv2.resize(load_depth(osp.join(traj_dir, "depth_left", f"{fid:06d}_left_depth.npy")), (PROC, PROC), interpolation=cv2.INTER_NEAREST)
        ext_t = TartanAirDataset._pose_to_extri(all_poses[fid])
        ext_t2 = TartanAirDataset._pose_to_extri(all_poses[fid2])
        ego = compute_ego_flow(depth, K_proc, ext_t, ext_t2)  # (H,W,2) proc px

        res = flow.transpose(1, 2, 0) - ego  # (H,W,2) proc px
        res *= NATIVE / PROC  # -> native px units (uniform scale)
        resid = np.linalg.norm(res, axis=-1)  # (H,W) native px
        resid[depth <= 0] = 0.0  # can't judge invalid depth -> static
        mask_proc = (resid > thr).astype(np.uint8) * 255
        mask = cv2.resize(mask_proc, (NATIVE, NATIVE), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(osp.join(out_dir, f"dyn_{fid:06d}.png"), mask)

        if save_vis and t < 6:
            rgb = cv2.resize(cv2.imread(frame_paths[t]), (NATIVE, NATIVE))
            resid_col = cv2.applyColorMap(np.clip(resid / max(thr * 3, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            resid_col = cv2.resize(resid_col, (NATIVE, NATIVE))
            row = np.concatenate([rgb, resid_col, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=1)
            cv2.imwrite(osp.join(vis_dir, f"cmp_{fid:06d}.png"), row)
    return f"ok({F})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tartanair_dir", default=data_path("train", "tartanair"))
    ap.add_argument("--envs", nargs="*", default=None, help="subset of env names; default all")
    ap.add_argument("--thr", type=float, default=2.0, help="flow-residual threshold in NATIVE px")
    ap.add_argument("--gap", type=int, default=5, help="frame gap delta for residual (t -> t+delta)")
    ap.add_argument("--vis", action="store_true", help="save comparison viz for first frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    tf = weights.transforms()

    train_dir = osp.join(args.tartanair_dir, "train")
    traj_dirs = sorted(glob.glob(osp.join(train_dir, "*", "*", "*/")))
    if args.envs:
        traj_dirs = [d for d in traj_dirs if osp.relpath(d.rstrip("/"), train_dir).split(os.sep)[0] in args.envs]
    print(f"{len(traj_dirs)} trajectories | thr={args.thr}px | gap={args.gap} | proc={PROC}x{PROC}")
    from tqdm import tqdm

    for d in tqdm(traj_dirs):
        name = osp.relpath(d.rstrip("/"), train_dir)
        done_flag = osp.join(d, "dynmask_raft", ".done")
        if osp.isfile(done_flag) and not args.overwrite:
            continue
        st = process_traj(d, model, tf, device, args.thr, gap=args.gap, save_vis=args.vis)
        if st.startswith("ok"):
            open(done_flag, "w").close()
        tqdm.write(f"  {name}: {st}")
    print("DONE.")


if __name__ == "__main__":
    main()
