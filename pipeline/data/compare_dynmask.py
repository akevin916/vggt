#!/usr/bin/env python3
"""Compare dynamic-mask variants on PointOdyssey and write RGB | GT seg | RAFT gap5 | mask panels.

--method old      : per-instance-COLOR decision, skips color 0 (black), fills whole color, + holefill
--method refined  : per connected-COMPONENT decision (no color-0 skip) + MINTRK guard + holefill

The 4th panel shows the chosen mask in white with RED where RAFT gap5 fires but the mask does not.

Run (from the repo root):
  python -m pipeline.data.compare_dynmask --method old                     # gap5, ITHR0.01, IFRAC0.1 defaults
  python -m pipeline.data.compare_dynmask --method old --ithr 0.02 --frames 3
"""
import os, os.path as osp, glob, argparse
import numpy as np, cv2, torch
from scipy import ndimage
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pipeline.data.motion_mask import compute_ego_flow
from pipeline.data.paths import data_path

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["old", "refined", "raft_snap", "all"], default="old",
                help="raft_snap = fill an instance/CC if enough of it is RAFT-positive (no GT tracks); "
                     "all = 5-panel compare gap5 | raft_snap | instance x GT-flow")
ap.add_argument("--snap_thr", type=float, default=0.2,
                help="raft_snap: fraction of a CC's pixels that must be RAFT-positive to fill it")
ap.add_argument("--po_dir", default=data_path("train", "point_odyssey"))
ap.add_argument("--split", default="test")
ap.add_argument("--seqs", nargs="*", default=["ALL"])
ap.add_argument("--frames", type=int, default=3)
ap.add_argument("--ig", type=int, default=5)
ap.add_argument("--ithr", type=float, default=0.01)
ap.add_argument("--ifrac", type=float, default=0.1)
ap.add_argument("--mintrk", type=int, default=8)
ap.add_argument("--mincc", type=float, default=0.001)
ap.add_argument("--maxcc", type=float, default=0.4,
                help="skip a connected component larger than this fraction of the image (background/over-merged blob)")
ap.add_argument("--open_iter", type=int, default=0,
                help="binary-opening iterations on each color BEFORE labeling, to sever thin bridges that merge bg+object into one CC")
ap.add_argument("--rthr", type=float, default=2.0, help="RAFT residual threshold (native px)")
ap.add_argument("--outdir", default="/home/cvml-75/Desktop/vggt/mask_compare")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
PW, PH, PANEL = 480, 272, (480, 270)
dev = "cuda"
w = Raft_Large_Weights.DEFAULT
raft = raft_large(weights=w).to(dev).eval()
tf = w.transforms()


def keymap(mask):
    m = mask.astype(np.int64)  # cast first (NumPy 2.x: uint8*256 overflows)
    return m[:, :, 0] * 65536 + m[:, :, 1] * 256 + m[:, :, 2]


def holefill(m):
    return (ndimage.binary_fill_holes(ndimage.binary_closing(m > 127, np.ones((3, 3)))) * 255).astype(np.uint8)


def tracks(anno, fid):
    t3, t2, val, vis = anno; F = t3.shape[0]; t2i = min(fid + args.ig, F - 1)
    good = val[fid] & val[t2i] & vis[fid]
    dm = np.linalg.norm(t3[t2i, good] - t3[fid, good], axis=1)
    return t2[fid, good], dm


def inst_old(mask, anno, fid):
    p, dm = tracks(anno, fid); H, W = mask.shape[:2]
    xs = np.clip(p[:, 0].astype(int), 0, W - 1); ys = np.clip(p[:, 1].astype(int), 0, H - 1)
    ak = keymap(mask); key = ak[ys, xs]; out = np.zeros((H, W), np.uint8)
    for k in np.unique(key):
        if k == 0:
            continue
        if (dm[key == k] > args.ithr).mean() > args.ifrac:
            out[ak == k] = 255
    return holefill(out)


def inst_refined(mask, anno, fid):
    p, dm = tracks(anno, fid); H, W = mask.shape[:2]
    xs = np.clip(p[:, 0].astype(int), 0, W - 1); ys = np.clip(p[:, 1].astype(int), 0, H - 1)
    ak = keymap(mask); tkey = ak[ys, xs]; out = np.zeros((H, W), np.uint8)
    for k in np.unique(ak):
        binm = (ak == k)
        if binm.mean() < args.mincc:
            continue
        binlab = ndimage.binary_opening(binm, np.ones((3, 3)), iterations=args.open_iter) if args.open_iter > 0 else binm
        lab, n = ndimage.label(binlab); cc_at = lab[ys, xs]
        for c in range(1, n + 1):
            cc = (lab == c)
            a = cc.mean()
            if a < args.mincc or a > args.maxcc:   # skip too-small and too-large (bg/over-merged) blobs
                continue
            sel = (tkey == k) & (cc_at == c)
            if sel.sum() < args.mintrk:
                continue
            if (dm[sel] > args.ithr).mean() > args.ifrac:
                out[cc] = 255
    return holefill(out)


def raft_snap(mask, g):
    """Fill an instance connected-component if enough of its pixels are RAFT-positive (no GT tracks)."""
    ak = keymap(mask); H, W = mask.shape[:2]; gpos = g > 127; out = np.zeros((H, W), np.uint8)
    for k in np.unique(ak):
        binm = (ak == k)
        if binm.mean() < args.mincc:
            continue
        binlab = ndimage.binary_opening(binm, np.ones((3, 3)), iterations=args.open_iter) if args.open_iter > 0 else binm
        lab, n = ndimage.label(binlab)
        for c in range(1, n + 1):
            cc = (lab == c); a = cc.mean()
            if a < args.mincc or a > args.maxcc:
                continue
            if gpos[cc].mean() > args.snap_thr:
                out[cc] = 255
    return holefill(out)


def raft_gap(s, rp, K, ext, t):
    natH, natW = cv2.imread(rp[0]).shape[:2]; sx, sy = PW / natW, PH / natH
    t2 = min(t + args.ig, len(rp) - 1)
    if t2 == t:
        return np.zeros((natH, natW), np.uint8)

    def rdi(p):
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(cv2.resize(im, (PW, PH))).permute(2, 0, 1).contiguous()
    b1, b2 = tf(rdi(rp[t])[None], rdi(rp[t2])[None])
    with torch.no_grad():
        flow = raft(b1.to(dev), b2.to(dev))[-1][0].cpu().numpy().transpose(1, 2, 0)
    fid = int(osp.basename(rp[t]).split("_")[1].split(".")[0])
    d = cv2.imread(s + f"depths/depth_{fid:05d}.png", cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535 * 1000
    d[d >= 1000] = 0; d = cv2.resize(d, (PW, PH), interpolation=cv2.INTER_NEAREST)
    Ks = K[t].copy(); Ks[0] *= sx; Ks[1] *= sy
    ego = compute_ego_flow(d, Ks, ext[t][:3], ext[t2][:3])
    res = flow - ego; res[..., 0] *= natW / PW; res[..., 1] *= natH / PH
    r = np.linalg.norm(res, axis=-1); r[d <= 0] = 0
    return cv2.resize((r > args.rthr).astype(np.uint8) * 255, (natW, natH), interpolation=cv2.INTER_NEAREST)


def panel(img, label=None):
    im = cv2.resize(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), PANEL)
    if label:
        cv2.rectangle(im, (0, 0), (PANEL[0], 22), (0, 0, 0), -1)
        cv2.putText(im, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


seqs = sorted(glob.glob(osp.join(args.po_dir, args.split, "*/")))
if args.seqs != ["ALL"]:
    seqs = [s for s in seqs if osp.basename(s.rstrip("/")) in args.seqs]
for s in seqs:
    name = osp.basename(s.rstrip("/"))
    rp = sorted(glob.glob(s + "rgbs/rgb_*.jpg"))
    an = np.load(s + "anno.npz", allow_pickle=True)
    if "trajs_3d" not in an or an["trajs_3d"].ndim != 3 or len(rp) < args.ig + 2:
        print(f"skip {name}"); continue
    K, ext = an["intrinsics"].astype(np.float32), an["extrinsics"].astype(np.float32)
    anno = (an["trajs_3d"].astype(np.float32), an["trajs_2d"].astype(np.float32),
            an["valids"].astype(bool), an["visibs"].astype(bool))
    rows = []
    for j, t in enumerate(np.linspace(len(rp) * 0.15, len(rp) * 0.85, args.frames).astype(int)):
        fid = int(osp.basename(rp[t]).split("_")[1].split(".")[0])
        rgb = cv2.imread(rp[t]); seg = cv2.imread(s + f"masks/mask_{fid:05d}.png")
        g = raft_gap(s, rp, K, ext, t)
        h = (j == 0)
        if args.method == "all":
            snap = raft_snap(seg, g); ref = inst_refined(seg, anno, fid)
            rows.append(np.concatenate([
                panel(rgb, f"{name[:14]} f{fid}" if h else None),
                panel(seg, "GT seg" if h else None),
                panel(g, "RAFT gap5" if h else None),
                panel(snap, f"raft_snap (thr{args.snap_thr})" if h else None),
                panel(ref, f"inst x GT-flow ITHR{args.ithr}" if h else None)], axis=1))
        else:
            fn = {"old": inst_old, "refined": inst_refined}.get(args.method)
            m = raft_snap(seg, g) if args.method == "raft_snap" else fn(seg, anno, fid)
            H, W = m.shape; dp = np.zeros((H, W, 3), np.uint8)
            dp[m > 127] = (255, 255, 255); dp[(g > 127) & (m <= 127)] = (0, 0, 255)
            rows.append(np.concatenate([
                panel(rgb, f"{name[:14]} {args.method} ITHR{args.ithr} IG{args.ig}" if h else None),
                panel(seg, "GT seg" if h else None),
                panel(g, "RAFT gap5" if h else None),
                panel(dp, "mask (red=gap5 not mask)" if h else None)], axis=1))
    cv2.imwrite(f"{args.outdir}/cmp_{args.method}_{name}.png", np.concatenate(rows, axis=0))
    print(f"{name} done")
print("DONE ->", args.outdir)
