#!/usr/bin/env python3
"""Precompute PointOdyssey DYNAMIC MASKS (m*_inst, dyn_vggt_method_v3 §5.3b) via instance x
GT-scene-flow (the 'refined' method).

Definition (motion-defined, robust; no RAFT / no ego / no depth):
  For frame t, a GT-tracked point is 'moving' if  ||trajs_3d[t+IG] - trajs_3d[t]|| > ITHR  (world metres).
  Split every instance-mask COLOR into connected components; a component is dynamic if
    >= MINTRK of its tracked points exist AND fraction(moving) > IFRAC,
    skipping components smaller than MINCC or larger than MAXCC (background / over-merged blobs).
  Fill the whole component, then morphological close(3x3) + fill-holes to solidify.

Why instead of RAFT flow-residual: a static world point has EXACTLY zero displacement at any gap,
so there is no static-background blow-up on fast-camera scenes and no gap fast/slow tension.

Output (drop-in for masks/, loaded via dataset dynamic_source="instance"):
  <seq>/dynmask_inst/dyn_{fid:05d}.png   uint8 {0,255}, native res   (+ a .done marker)

Run (from training/):
  python data/preprocess/po_instance_dynmask.py --split test           # small, do first
  python data/preprocess/po_instance_dynmask.py --split train          # full (background)
"""
import os, os.path as osp, glob, argparse
import numpy as np, cv2
from scipy import ndimage
from tqdm import tqdm


def keymap(mask):
    m = mask.astype(np.int64)  # cast first (NumPy 2.x: uint8*256 overflows)
    return m[:, :, 0] * 65536 + m[:, :, 1] * 256 + m[:, :, 2]


def blob_map(mask, mincc, maxcc, open_iter):
    """Global blob id per connected component of each instance color, filtered by area fraction."""
    ak = keymap(mask)
    H, W = ak.shape
    A = H * W
    bmap = np.zeros((H, W), np.int32)
    bid = 0
    for k in np.unique(ak):
        binm = (ak == k)
        if binm.mean() < mincc:
            continue
        b8 = binm.astype(np.uint8)
        if open_iter > 0:
            b8 = cv2.morphologyEx(b8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=open_iter)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b8, connectivity=8)
        for c in range(1, n):
            area = stats[c, cv2.CC_STAT_AREA] / A
            if area < mincc or area > maxcc:
                continue
            bid += 1
            bmap[lab == c] = bid
    return bmap, bid


def refined_mask(mask, anno, fid, ig, ithr, ifrac, mintrk, mincc, maxcc, open_iter):
    t3, t2, val, vis = anno
    F = t3.shape[0]
    H, W = mask.shape[:2]
    t2i = min(fid + ig, F - 1)
    if t2i == fid:
        return np.zeros((H, W), np.uint8)
    bmap, nb = blob_map(mask, mincc, maxcc, open_iter)
    if nb == 0:
        return np.zeros((H, W), np.uint8)
    good = val[fid] & val[t2i] & vis[fid]
    dm = np.linalg.norm(t3[t2i, good] - t3[fid, good], axis=1)
    p = t2[fid, good]
    xs = np.clip(p[:, 0].astype(int), 0, W - 1)
    ys = np.clip(p[:, 1].astype(int), 0, H - 1)
    tb = bmap[ys, xs]
    moving = dm > ithr
    dyn = []
    for b in range(1, nb + 1):
        sel = (tb == b)
        nt = int(sel.sum())
        if nt >= mintrk and moving[sel].mean() > ifrac:
            dyn.append(b)
    out = np.isin(bmap, dyn) if dyn else np.zeros((H, W), bool)
    out = ndimage.binary_fill_holes(ndimage.binary_closing(out, np.ones((3, 3))))
    return (out * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--po_dir", default="/media/cvml-75/ssd2t1/data/point_odyssey")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--ig", type=int, default=5)
    ap.add_argument("--ithr", type=float, default=0.01, help="world-delta threshold (metres)")
    ap.add_argument("--ifrac", type=float, default=0.10)
    ap.add_argument("--mintrk", type=int, default=8)
    ap.add_argument("--mincc", type=float, default=0.001)
    ap.add_argument("--maxcc", type=float, default=0.4)
    ap.add_argument("--open_iter", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seqs = sorted(glob.glob(osp.join(args.po_dir, args.split, "*/")))
    if args.seqs:
        seqs = [s for s in seqs if osp.basename(s.rstrip("/")) in args.seqs]
    print(
        f"{len(seqs)} seqs | IG={args.ig} ITHR={args.ithr} IFRAC={args.ifrac} "
        f"MINTRK={args.mintrk} MAXCC={args.maxcc} open={args.open_iter}"
    )

    for s in tqdm(seqs):
        name = osp.basename(s.rstrip("/"))
        out_dir = osp.join(s, "dynmask_inst")
        if osp.isfile(osp.join(out_dir, ".done")) and not args.overwrite:
            continue
        mps = sorted(glob.glob(osp.join(s, "masks", "mask_*.png")))
        anno_p = osp.join(s, "anno.npz")
        if not mps or not osp.isfile(anno_p):
            tqdm.write(f"  {name}: skip (no masks/anno)")
            continue
        an = np.load(anno_p, allow_pickle=True)
        if "trajs_3d" not in an or an["trajs_3d"].ndim != 3:
            tqdm.write(f"  {name}: skip (no trajs_3d)")
            continue
        anno = (
            an["trajs_3d"].astype(np.float32),
            an["trajs_2d"].astype(np.float32),
            an["valids"].astype(bool),
            an["visibs"].astype(bool),
        )
        os.makedirs(out_dir, exist_ok=True)
        for mp in mps:
            fid = int(osp.basename(mp).split("_")[1].split(".")[0])
            if fid >= anno[0].shape[0]:
                continue
            mask = cv2.imread(mp)
            if mask is None:
                continue
            m = refined_mask(mask, anno, fid, args.ig, args.ithr, args.ifrac, args.mintrk, args.mincc, args.maxcc, args.open_iter)
            cv2.imwrite(osp.join(out_dir, f"dyn_{fid:05d}.png"), m)
        open(osp.join(out_dir, ".done"), "w").close()
        tqdm.write(f"  {name}: ok ({len(mps)} frames)")
    print("DONE.")


if __name__ == "__main__":
    main()
