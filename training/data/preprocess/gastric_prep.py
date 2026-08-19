"""Turn the raw gastric clip into sequences a reconstruction model can actually use.

The clip as extracted is not usable frame-by-frame, for three separate reasons, all measured
on the raw clip by the (since removed) ``diag/inspect_private.py``:

1. **Duplicated frames.** The container runs at 60 fps but the content only updates every
   third frame, so two of every three inputs carry no new information. Feeding them
   straight in makes triangulation degenerate -- this is not "small parallax", it is *no*
   parallax. De-duplication by stride is mandatory, not a tuning knob.
2. **Bursty motion.** Even after de-duplication ~29% of frames are still static, arranged
   as bursts separated by long stalls. Sampling uniformly across the clip therefore lands
   most frames in the stalls; sequences must be cut from the moving segments instead.
3. **Letterbox borders.** The endoscope image is a rounded-corner rectangle covering ~61%
   of the frame, static across the clip. The corners carry no texture, so depth there is
   unconstrained and grows fake geometry in the point cloud.

This script measures 1 and 2, writes a contact sheet so the segment choice is informed
rather than guessed, and exports the chosen segments cropped, masked and renumbered.

Prepared sequences land in ``<gastric_root>/prep/`` (they are data, and follow the same
convention as the ``po_*``/``*_dynmask`` preprocessors); the diagnostic figure and the
segment table land in ``outputs/gastric_prep/``.

Usage:
  python data/preprocess/gastric_prep.py --list              # measure + contact sheet only
  python data/preprocess/gastric_prep.py --export_top 3      # also write the 3 best segments
  python data/preprocess/gastric_prep.py --export 987 1143   # or an explicit frame range
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.paths import data_path
from eval_utils.media_io import write_video
from eval_utils.paths import output_dir_for_exp

TOOL = "gastric_prep"
DARK = 25.0            # the letterbox sits at luminance 15; tissue starts well above 25
MASK_ERODE = 3         # shave the 1-2 px intensity ramp at the border


def load_gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def valid_mask(frames, n_probe=20):
    """The endoscope's field of view as a *geometric* region, not a brightness test.

    Thresholding brightness per pixel is the obvious approach and it is wrong: deep lumen
    tissue is genuinely darker than the letterbox in some frames, so an intersection over
    frames punches holes through the middle of the image (it cost 3% of the frame here,
    scattered as speckle). Instead take the union over frames -- a pixel inside the FOV is
    bright in at least one of them -- and then fill: the FOV is a convex rounded rectangle,
    so filling each row and each column between their first and last lit pixel and
    intersecting the two recovers the region exactly, holes and all.
    """
    probe = frames[::max(1, len(frames) // n_probe)]
    lit = np.zeros(load_gray(probe[0]).shape, bool)
    for p in probe:
        lit |= load_gray(p) > DARK

    def fill(a):                             # per-row span fill
        out = np.zeros_like(a)
        for i, row in enumerate(a):
            nz = np.nonzero(row)[0]
            if len(nz):
                out[i, nz[0]:nz[-1] + 1] = True
        return out

    m = fill(lit) & fill(lit.T).T            # convex region -> exact
    if MASK_ERODE > 0:                       # shave the border ramp; min-filter, no scipy
        from PIL import ImageFilter
        im = Image.fromarray((m * 255).astype(np.uint8))
        m = np.asarray(im.filter(ImageFilter.MinFilter(2 * MASK_ERODE + 1))) > 127
    ys, xs = np.nonzero(m)
    bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    return m, bbox


def detect_period(frames, probe=(300, 480)):
    """Smallest stride at which consecutive frames actually differ.

    Measured on a moving stretch only: in a stall every stride looks identical and the
    period is undefined there.
    """
    lo, hi = probe
    g = [load_gray(p) for p in frames[lo:hi]]
    d = np.array([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])
    thresh = 2.0 * np.median(d)
    novel = np.nonzero(d > thresh)[0]
    if len(novel) < 3:
        return 1, d
    gaps = np.diff(novel)
    return int(np.median(gaps)), d


def motion_profile(frames, idx, bbox, downsample=4, radius=20):
    """Coarse per-step translation in original pixels, over the cropped valid region."""
    y0, y1, x0, x1 = bbox

    def get(i):
        return load_gray(frames[i])[y0:y1 + 1, x0:x1 + 1][::downsample, ::downsample]

    g = [get(i) for i in idx]
    H, W = g[0].shape
    box = (H // 4, 3 * H // 4, W // 4, 3 * W // 4)
    out = []
    for k in range(len(g) - 1):
        a, b = g[k], g[k + 1]
        ya, yb, xa, xb = box
        c = a[ya:yb, xa:xb]
        best = (np.inf, 0, 0)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                w = b[ya + dy:yb + dy, xa + dx:xb + dx]
                if w.shape != c.shape:
                    continue
                e = float(np.abs(c - w).mean())
                if e < best[0]:
                    best = (e, dy, dx)
        out.append(np.hypot(best[1], best[2]) * downsample)
    return np.array(out)


def find_segments(mv, idx, min_len, min_px):
    """Runs of at least ``min_len`` consecutive steps that all move at least ``min_px``."""
    ok = mv >= min_px
    segs, s = [], None
    for i, v in enumerate(list(ok) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_len:
                segs.append(dict(
                    i0=s, i1=i, n_frames=i - s + 1,
                    frame_start=int(idx[s]), frame_end=int(idx[i]),
                    motion_median=float(np.median(mv[s:i])),
                    motion_total=float(mv[s:i].sum()),
                ))
            s = None
    # Longest first: sequence length is what the "pair -> sequence" claim rests on.
    return sorted(segs, key=lambda d: -d["n_frames"])


def export_segment(frames, seg, mask, bbox, out_root, stride):
    y0, y1, x0, x1 = bbox
    name = f"seq_{seg['frame_start']:06d}_{seg['frame_end']:06d}"
    out_dir = os.path.join(out_root, name)
    img_dir = os.path.join(out_dir, "image")
    os.makedirs(img_dir, exist_ok=True)

    sub = mask[y0:y1 + 1, x0:x1 + 1]
    Image.fromarray((sub * 255).astype(np.uint8)).save(os.path.join(out_dir, "mask.png"))

    src_ids = list(range(seg["frame_start"], seg["frame_end"] + 1, stride))
    kept = []
    for j, fid in enumerate(src_ids):
        a = np.asarray(Image.open(frames[fid]).convert("RGB"))[y0:y1 + 1, x0:x1 + 1]
        a = (a * sub[..., None]).astype(np.uint8)    # black out the rounded corners
        Image.fromarray(a).save(os.path.join(img_dir, f"{j:06d}.png"))
        kept.append(a)
    # The clip ships with the frames so the segment can be judged without opening 80 PNGs,
    # and so whatever point cloud gets reconstructed from it has its footage alongside.
    write_video(os.path.join(out_dir, "segment.mp4"), kept, fps=8.0)

    meta = dict(source_frames=src_ids, stride=stride, crop_bbox=dict(y0=y0, y1=y1, x0=x0, x1=x1),
                size=[int(y1 - y0 + 1), int(x1 - x0 + 1)], n_frames=len(src_ids),
                mask_valid_frac=float(sub.mean()), **{k: seg[k] for k in
                ("motion_median", "motion_total")})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gastric_root", default=data_path("eval", "gastric"))
    ap.add_argument("--stride", type=int, default=None, help="default: auto-detected period")
    ap.add_argument("--min_len", type=int, default=8, help="min steps in a usable segment")
    ap.add_argument("--min_px", type=float, default=2.0, help="min per-step motion (px)")
    ap.add_argument("--list", action="store_true", help="measure and plot only")
    ap.add_argument("--export_top", type=int, default=0, help="export the N longest segments")
    ap.add_argument("--export", nargs=2, type=int, default=None, metavar=("START", "END"))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    src = os.path.join(args.gastric_root, "frames")
    frames = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    frames = [os.path.join(src, f) for f in frames]
    print(f"{len(frames)} frames in {src}")

    mask, bbox = valid_mask(frames)
    y0, y1, x0, x1 = bbox
    print(f"valid {mask.mean():.1%}  bbox y{y0}-{y1} x{x0}-{x1}  "
          f"({y1 - y0 + 1}x{x1 - x0 + 1})")

    period, adj = detect_period(frames)
    stride = args.stride or period
    print(f"detected duplicate period = {period} (adjacent diff median {np.median(adj):.2f})"
          f" -> stride {stride}, effective {len(frames) // stride} frames")

    idx = list(range(0, len(frames), stride))
    mv = motion_profile(frames, idx, bbox)
    segs = find_segments(mv, idx, args.min_len, args.min_px)
    print(f"\nmotion: median {np.median(mv):.1f}px  static(<{args.min_px}px) "
          f"{(mv < args.min_px).mean():.1%}")
    print(f"{'#':>2} {'frames':>16} {'n':>4} {'med px/step':>12} {'total px':>10}")
    for i, s in enumerate(segs):
        print(f"{i:>2} {s['frame_start']:6d}-{s['frame_end']:<9d} {s['n_frames']:>4} "
              f"{s['motion_median']:>12.1f} {s['motion_total']:>10.0f}")

    out_dir = args.out_dir or output_dir_for_exp("private", TOOL)
    os.makedirs(out_dir, exist_ok=True)

    n_show = min(6, len(segs))
    fig = plt.figure(figsize=(15, 4 + 2.6 * ((n_show + 2) // 3)))
    gs = fig.add_gridspec(2 + (n_show + 2) // 3, 3, height_ratios=[1.1, 1.1] +
                          [1] * ((n_show + 2) // 3))
    ax = fig.add_subplot(gs[0, :])
    ax.plot(idx[:-1], mv, lw=1)
    ax.axhline(args.min_px, color="r", ls="--", lw=.8, label=f"{args.min_px}px threshold")
    for s in segs[:n_show]:
        ax.axvspan(s["frame_start"], s["frame_end"], alpha=.18, color="tab:green")
    ax.set_xlabel("source frame"); ax.set_ylabel("px / step")
    ax.set_title(f"motion after stride-{stride} de-duplication (shaded = usable segments)")
    ax.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(adj, lw=1)
    ax2.set_title(f"raw adjacent-frame difference (period {period}: only every "
                  f"{period}rd frame is new)")
    ax2.set_xlabel("frame offset within probe window")

    for k, s in enumerate(segs[:n_show]):
        a = fig.add_subplot(gs[2 + k // 3, k % 3])
        mid = (s["frame_start"] + s["frame_end"]) // 2
        im = np.asarray(Image.open(frames[mid]).convert("RGB"))[y0:y1 + 1, x0:x1 + 1]
        a.imshow(im * mask[y0:y1 + 1, x0:x1 + 1][..., None])
        a.set_title(f"#{k}  {s['frame_start']}-{s['frame_end']}  "
                    f"{s['n_frames']}f  {s['motion_median']:.0f}px/step", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "gastric_segments.png")
    fig.savefig(fig_path, dpi=120); plt.close(fig)

    with open(os.path.join(out_dir, "segments.json"), "w") as f:
        json.dump(dict(n_frames=len(frames), period=period, stride=stride,
                       bbox=dict(y0=y0, y1=y1, x0=x0, x1=x1),
                       mask_valid_frac=float(mask.mean()),
                       motion_median=float(np.median(mv)), segments=segs), f, indent=2)
    print(f"\n{fig_path}\n{os.path.join(out_dir, 'segments.json')}")

    chosen = []
    if args.export:
        chosen = [dict(frame_start=args.export[0], frame_end=args.export[1],
                       motion_median=float("nan"), motion_total=float("nan"))]
    elif args.export_top:
        chosen = segs[:args.export_top]
    for s in chosen:
        d, meta = export_segment(frames, s, mask, bbox,
                                 os.path.join(args.gastric_root, "prep"), stride)
        print(f"exported {meta['n_frames']} frames -> {d}")


if __name__ == "__main__":
    main()
