"""Inventory the two private endoscopic sets before any reconstruction is run.

Answers the questions that decide the dump spec:
  * gastric -- where is the valid (non-letterbox) region, is it static across the clip,
    and how much camera motion is there between frames? A 60 fps clip can be degenerate
    (zero parallax) for long stretches, which no pose model can recover from.
  * lesion  -- how large is the specular-highlight fraction? Speculars are view-dependent,
    so an L->R warp cannot reproduce them and they dominate the PSNR residual.

Read-only on the datasets; everything lands in outputs/inspect_private/<set>/.
"""
import argparse, glob, json, os, sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils.paths import output_dir_for_exp
from data.paths import data_path

TOOL = "inspect_private"
DARK = 25.0          # gastric letterbox sits at luminance 15, tissue starts well above 25


def _lum(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def _best_shift(a, b, box, radius, step=2):
    """Coarse translation between two frames by brute-force SAD over a centre patch."""
    y0, y1, x0, x1 = box
    c = a[y0:y1, x0:x1]
    best = (np.inf, 0, 0)
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            w = b[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            if w.shape != c.shape:
                continue
            e = float(np.abs(c - w).mean())
            if e < best[0]:
                best = (e, dy, dx)
    return best


def inspect_gastric(root, out_dir, stride):
    frames = sorted(glob.glob(os.path.join(root, "frames", "0*.png")))
    assert frames, f"no frames under {root}"
    H, W = _lum(frames[0]).shape

    # --- valid mask: intersection of "bright" over a sample, so a transient specular
    # blob in the border cannot leak into the mask.
    probe = frames[::max(1, len(frames) // 20)]
    inter = np.ones((H, W), bool)
    union = np.zeros((H, W), bool)
    for p in probe:
        m = _lum(p) > DARK
        inter &= m
        union |= m
    ys, xs = np.nonzero(inter)
    bbox = dict(y0=int(ys.min()), y1=int(ys.max()), x0=int(xs.min()), x1=int(xs.max()))
    sub = inter[bbox["y0"]:bbox["y1"] + 1, bbox["x0"]:bbox["x1"] + 1]

    # --- motion profile: adjacent-frame residual + coarse shift, subsampled by `stride`
    box = (H // 3, 2 * H // 3, W // 3, 2 * W // 3)
    idx = list(range(0, len(frames) - 1, stride))
    prof = []
    for i in idx:
        a, b = _lum(frames[i]), _lum(frames[i + 1])
        e, dy, dx = _best_shift(a, b, box, radius=40, step=4)
        prof.append((i, e, float(np.hypot(dy, dx)), float(np.abs(a - b).mean())))
    prof = np.array(prof)

    stats = dict(
        n_frames=len(frames), resolution=[int(H), int(W)],
        valid_frac_intersection=float(inter.mean()), valid_frac_union=float(union.mean()),
        mask_static=bool(abs(inter.mean() - union.mean()) < 1e-3),
        bbox=bbox, bbox_size=[int(sub.shape[0]), int(sub.shape[1])],
        fill_inside_bbox=float(sub.mean()),
        motion_px_median=float(np.median(prof[:, 2])),
        motion_px_p90=float(np.percentile(prof[:, 2], 90)),
        frac_static_frames=float((prof[:, 2] < 1.0).mean()),
        raw_diff_median=float(np.median(prof[:, 3])),
    )

    Image.fromarray((inter * 255).astype(np.uint8)).save(os.path.join(out_dir, "gastric_mask.png"))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    rgb = np.asarray(Image.open(frames[len(frames) // 2]).convert("RGB"))
    ax[0].imshow(rgb); ax[0].set_title("frame (mid clip)")
    ov = rgb.copy(); ov[~inter] = (ov[~inter] * 0.25).astype(np.uint8)
    ax[1].imshow(ov)
    ax[1].add_patch(plt.Rectangle((bbox["x0"], bbox["y0"]), sub.shape[1], sub.shape[0],
                                  fill=False, ec="lime", lw=2))
    ax[1].set_title(f"valid mask {stats['valid_frac_intersection']:.1%}  bbox {sub.shape[0]}x{sub.shape[1]}")
    ax[2].plot(prof[:, 0], prof[:, 2], lw=1, label="coarse shift (px)")
    ax[2].plot(prof[:, 0], prof[:, 3], lw=1, alpha=.6, label="raw |I_t - I_t+1|")
    ax[2].axhline(1.0, color="r", ls="--", lw=.8, label="static threshold 1 px")
    ax[2].set_xlabel("frame"); ax[2].set_title("inter-frame motion"); ax[2].legend(fontsize=8)
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()
    p = os.path.join(out_dir, "gastric_overview.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return stats, [p, os.path.join(out_dir, "gastric_mask.png")]


def inspect_lesion(root, out_dir):
    per_folder, rows = {}, []
    for folder in sorted(glob.glob(os.path.join(root, "*/"))):
        name = os.path.basename(folder.rstrip("/"))
        ls = sorted(glob.glob(os.path.join(folder, "*_L.png")))
        spec, dark, shifts = [], [], []
        for p in ls:
            L = _lum(p); R = _lum(p.replace("_L.png", "_R.png"))
            spec.append(float((L > 250).mean()))
            dark.append(float((L < DARK).mean()))
            e, dy, dx = _best_shift(L, R, (156, 356, 156, 356), radius=150, step=2)
            shifts.append((dy, dx, e))
        sh = np.array(shifts, float)
        per_folder[name] = dict(
            n_pairs=len(ls), resolution=list(_lum(ls[0]).shape),
            specular_frac_mean=float(np.mean(spec)), specular_frac_max=float(np.max(spec)),
            dark_frac_mean=float(np.mean(dark)),
            disparity_px_median=float(np.median(np.abs(sh[:, 1]))),
            vertical_offset_px_median=float(np.median(np.abs(sh[:, 0]))),
        )
        rows.append((name, ls, spec))

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, ls, spec) in enumerate(rows):
        worst = int(np.argmax(spec))
        L = np.asarray(Image.open(ls[worst]).convert("RGB"))
        R = np.asarray(Image.open(ls[worst].replace("_L.png", "_R.png")).convert("RGB"))
        axes[r, 0].imshow(L); axes[r, 0].set_title(f"{name}  L (worst specular)", fontsize=9)
        axes[r, 1].imshow(R); axes[r, 1].set_title("R", fontsize=9)
        axes[r, 2].imshow(L.mean(-1) > 250, cmap="gray")
        axes[r, 2].set_title(f"specular mask {spec[worst]:.1%}", fontsize=9)
        axes[r, 3].plot(spec, marker="o", ms=3)
        axes[r, 3].set_title("specular frac per pair", fontsize=9)
        axes[r, 3].set_xlabel("pair"); axes[r, 3].grid(alpha=.3)
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout()
    p = os.path.join(out_dir, "lesion_overview.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return per_folder, [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gastric_root", default=data_path("eval", "gastric"))
    ap.add_argument("--lesion_root", default=data_path("eval", "lesion"))
    ap.add_argument("--stride", type=int, default=4, help="frame stride for the gastric motion profile")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or output_dir_for_exp("private", TOOL)
    os.makedirs(out_dir, exist_ok=True)

    g_stats, g_figs = inspect_gastric(args.gastric_root, out_dir, args.stride)
    l_stats, l_figs = inspect_lesion(args.lesion_root, out_dir)

    payload = dict(gastric=g_stats, lesion=l_stats)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nfigures:")
    for p in g_figs + l_figs:
        print(" ", p)
    print(" ", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
