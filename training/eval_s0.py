# NEW: Dyn-VGGT S0 validation. Confirms the two new heads are usable BEFORE moving to S1.
#   V1 motion segmentation quality (IoU/F1/precision/recall/AUC, dyn-vs-static prob) on the val split
#   V2 flow improves dynamic geometry: err(X^can+mΔ) vs err(X^can) on dynamic pixels (normalised space)
#   V3 frozen geometry intact: S0 depth/world_points ≈ base VGGT-1B (frozen + temporal γ=0)
#   V5 qualitative overlays saved to logs/dyn_vggt_po_s0/eval/
#
# Run (from training/):
#   python eval_s0.py --ckpt logs/dyn_vggt_po_s0/ckpts/checkpoint_7.pt --n_clips 20 --img_per_seq 6
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # training/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np, torch
from types import SimpleNamespace

from vggt.models.vggt import VGGT
from data.datasets.pointodyssey import PointOdysseyDataset
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n_clips", type=int, default=20)
ap.add_argument("--img_per_seq", type=int, default=6)
ap.add_argument("--img_size", type=int, default=518)
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--base_ckpt", default="checkpoints/VGGT-1B.pt", help="base VGGT for V3 frozen check")
ap.add_argument("--save_dir", default="logs/dyn_vggt_po_s0/eval")
args = ap.parse_args()
dev = "cuda"; os.makedirs(args.save_dir, exist_ok=True)

def build_model(ckpt, temporal, motion, flow):
    m = VGGT(img_size=args.img_size, enable_camera=True, enable_depth=True, enable_point=True,
             enable_track=False, enable_temporal=temporal, enable_motion=motion, enable_flow=flow)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    miss, unexp = m.load_state_dict(sd, strict=False)
    return m.to(dev).eval()

print(f"loading S0 model from {args.ckpt}")
model = build_model(args.ckpt, temporal=True, motion=True, flow=True)

common = SimpleNamespace(img_size=args.img_size, patch_size=14,
    augs=SimpleNamespace(scales=None), rescale=True, rescale_aug=False, landscape_check=False,
    debug=False, training=False, get_nearby=True, load_depth=True,
    inside_random=False, allow_duplicate_img=False)
ds = PointOdysseyDataset(common_conf=common, split="test", min_num_images=args.img_per_seq)
print(f"val sequences: {ds.sequence_list_len}")

def collate(b):
    img = torch.from_numpy(np.stack(b["images"]).astype(np.float32)).permute(0,3,1,2).div(255)[None]
    def st(k): return torch.from_numpy(np.stack(b[k]).astype(np.float32))[None]
    pm = torch.from_numpy(np.stack(b["point_masks"]))[None]
    mm = torch.from_numpy(np.stack(b["motion_mask"]).astype(np.float32))[None]
    return img, st("depths"), st("extrinsics"), st("intrinsics"), st("cam_points"), st("world_points"), pm, mm

# accumulators
TP=FP=FN=TN=0.0
dyn_prob_sum=stat_prob_sum=0.0; dyn_n=stat_n=0
probs_sub=[]; labels_sub=[]
canon_err_dyn=assem_err_dyn=0.0; nd=0
canon_err_stat=assem_err_stat=0.0; ns=0

torch.manual_seed(0); np.random.seed(0)
for ci in range(args.n_clips):
    b = ds.get_data(seq_index=ci % ds.sequence_list_len, img_per_seq=args.img_per_seq, aspect_ratio=1.0)
    img, dep, extr, intr, cam, world, pm, mm = collate(b)
    extr_n, cam_n, world_n, dep_n, _ = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extr, cam_points=cam, world_points=world, depths=dep, point_masks=pm)
    img = img.to(dev)
    with torch.no_grad():
        pred = model(images=img)
    m = pred["motion_prob"][...,0].float().cpu()        # (1,S,H,W)
    gt = mm                                              # (1,S,H,W) {0,1}
    valid = pm                                           # (1,S,H,W) bool
    # ---- V1 motion ----
    pm_bin = (m >= args.thr)
    g = gt.bool()
    TP += float((pm_bin & g).sum()); FP += float((pm_bin & ~g).sum())
    FN += float((~pm_bin & g).sum()); TN += float((~pm_bin & ~g).sum())
    dyn_prob_sum += float(m[g].sum()); dyn_n += int(g.sum())
    stat_prob_sum += float(m[~g].sum()); stat_n += int((~g).sum())
    # subsample for AUC
    flatp = m.flatten().numpy(); flatl = g.flatten().numpy().astype(np.int8)
    idx = np.random.choice(len(flatp), min(20000, len(flatp)), replace=False)
    probs_sub.append(flatp[idx]); labels_sub.append(flatl[idx])
    # ---- V2 flow: err on dynamic vs static (normalised GT) ----
    canon = pred["world_points"].float().cpu()           # X^can
    assem = pred["world_points_dyn"].float().cpu()       # X^can + m*Δ
    gtw = world_n                                        # normalised GT world points
    ce = (canon - gtw).norm(dim=-1)                      # (1,S,H,W)
    ae = (assem - gtw).norm(dim=-1)
    dyn = g & valid; stat = (~g) & valid
    canon_err_dyn += float(ce[dyn].sum()); assem_err_dyn += float(ae[dyn].sum()); nd += int(dyn.sum())
    canon_err_stat += float(ce[stat].sum()); assem_err_stat += float(ae[stat].sum()); ns += int(stat.sum())

# ---- report ----
prec = TP/(TP+FP+1e-9); rec = TP/(TP+FN+1e-9); f1 = 2*prec*rec/(prec+rec+1e-9)
iou = TP/(TP+FP+FN+1e-9); acc = (TP+TN)/(TP+TN+FP+FN+1e-9)
print("\n================ V1: MOTION SEGMENTATION (val) ================")
print(f"  IoU={iou:.3f}  F1={f1:.3f}  precision={prec:.3f}  recall={rec:.3f}  pixel-acc={acc:.3f}")
print(f"  mean prob on DYNAMIC px = {dyn_prob_sum/max(dyn_n,1):.3f}  | on STATIC px = {stat_prob_sum/max(stat_n,1):.3f}")
print(f"  dynamic pixel fraction (GT) = {dyn_n/max(dyn_n+stat_n,1):.3f}")
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    P = np.concatenate(probs_sub); L = np.concatenate(labels_sub)
    print(f"  ROC-AUC={roc_auc_score(L,P):.3f}  AP={average_precision_score(L,P):.3f}  (subsampled {len(P)} px)")
except Exception as e:
    print(f"  (sklearn AUC/AP skipped: {e})")

print("\n================ V2: FLOW IMPROVES DYNAMIC GEOMETRY ============")
cd_=canon_err_dyn/max(nd,1); ad_=assem_err_dyn/max(nd,1)
cs_=canon_err_stat/max(ns,1); as_=assem_err_stat/max(ns,1)
print(f"  DYNAMIC px: err(X^can)={cd_:.4f} -> err(X^can+mΔ)={ad_:.4f}   improvement={100*(cd_-ad_)/max(cd_,1e-9):+.1f}%")
print(f"  STATIC  px: err(X^can)={cs_:.4f} -> err(X^can+mΔ)={as_:.4f}   (should be ~equal)")
print("  PASS if dynamic improvement is clearly positive and static ~unchanged.")

# ---- V3 frozen geometry intact ----
print("\n================ V3: FROZEN GEOMETRY INTACT ====================")
try:
    base = build_model(args.base_ckpt, temporal=False, motion=False, flow=False)
    b = ds.get_data(seq_index=0, img_per_seq=args.img_per_seq, aspect_ratio=1.0)
    img = collate(b)[0].to(dev)
    with torch.no_grad():
        ps0 = model(images=img); pb = base(images=img)
    dd = (ps0["depth"]-pb["depth"]).abs().max().item()
    dw = (ps0["world_points"]-pb["world_points"]).abs().max().item()
    dp = (ps0["pose_enc"]-pb["pose_enc"]).abs().max().item()
    print(f"  max|Δdepth|={dd:.2e}  max|Δworld_points|={dw:.2e}  max|Δpose|={dp:.2e}")
    print(f"  PASS if all ≈ 0 (frozen backbone + temporal γ≈0 → geometry unchanged).  -> {'PASS' if max(dd,dw,dp)<1e-2 else 'CHECK'}")
except Exception as e:
    print(f"  (V3 skipped: {e})")

# ---- V5 qualitative ----
try:
    import cv2
    b = ds.get_data(seq_index=0, img_per_seq=args.img_per_seq, aspect_ratio=1.0)
    img, *_ , mm = collate(b)
    with torch.no_grad():
        pr = model(images=img.to(dev))["motion_prob"][...,0,].float().cpu()[0]   # (S,H,W)
    rgb = (img[0].permute(0,2,3,1).numpy()*255).astype(np.uint8)                  # (S,H,W,3)
    for s in range(min(3, rgb.shape[0])):
        g = (mm[0,s].numpy()*255).astype(np.uint8)
        p = (pr[s].numpy()*255).astype(np.uint8)
        row = np.concatenate([cv2.cvtColor(rgb[s],cv2.COLOR_RGB2BGR),
                              cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),
                              cv2.applyColorMap(p,cv2.COLORMAP_JET)], axis=1)
        cv2.imwrite(f"{args.save_dir}/qual_frame{s}.png", row)
    print(f"\nV5: qualitative overlays (RGB | GT mask | pred m) saved to {args.save_dir}/")
except Exception as e:
    print(f"  (V5 skipped: {e})")
print("\nDONE.")
