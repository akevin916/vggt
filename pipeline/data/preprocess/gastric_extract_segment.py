"""Extract a time-range segment from a gastric video into a prep-compatible directory.

Usage:
    python -m pipeline.data.preprocess.gastric_extract_segment \\
        --video ../data/eval/gastric/"video1.mp4 的副本-001.mp4" \\
        --t_start 115 --t_end 140 --n_frames 30 \\
        --seg_name seq_video1_0115_0140
"""
import argparse, json, os, sys
import cv2
import numpy as np
from PIL import Image

from pipeline.data.paths import data_path

# ---------- geometry constants from gastric_prep (crop_left + endoscope bbox) ----------
CROP_LEFT = 640
Y0, Y1, X0, X1 = 108, 972, 162, 1156   # in post-crop coordinates
MASK_SRC = os.path.join(
    data_path("eval", "gastric"), "prep",
    "seq_000189_000432", "mask.png"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(
        data_path("eval", "gastric"), "video1.mp4 的副本-001.mp4"))
    ap.add_argument("--t_start", type=float, required=True, help="start time in seconds")
    ap.add_argument("--t_end",   type=float, required=True, help="end time in seconds")
    ap.add_argument("--n_frames", type=int, default=30, help="output frames to keep")
    ap.add_argument("--seg_name", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    seg_name = args.seg_name or (
        f"seq_video1_{int(args.t_start):04d}_{int(args.t_end):04d}"
    )
    out_root = args.out_root or os.path.join(data_path("eval", "gastric"), "prep")
    img_dir  = os.path.join(out_root, seg_name, "image")
    os.makedirs(img_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    f_start = int(args.t_start * fps)
    f_end   = int(args.t_end   * fps)
    total   = f_end - f_start
    stride  = max(1, total // args.n_frames)
    print(f"video fps={fps:.1f}  raw [{f_start}, {f_end}]  "
          f"total={total}  stride={stride}  n_out≈{total//stride}")

    mask = np.asarray(Image.open(MASK_SRC).convert("L")) > 127

    kept_frames, src_ids = [], []
    j = 0
    for raw_idx in range(f_start, f_end, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  [warn] could not read frame {raw_idx}")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, CROP_LEFT:]                        # remove left mono view
        roi = rgb[Y0:Y1 + 1, X0:X1 + 1]               # endoscope bbox
        roi = (roi * mask[..., None]).astype(np.uint8)  # mask corners
        Image.fromarray(roi).save(os.path.join(img_dir, f"{j:06d}.png"))
        kept_frames.append(roi)
        src_ids.append(raw_idx)
        j += 1
        if j >= args.n_frames:
            break
    cap.release()
    print(f"saved {j} frames -> {img_dir}")

    # write segment.mp4 preview
    from pipeline.eval.media_io import write_video
    write_video(os.path.join(out_root, seg_name, "segment.mp4"),
                [f.astype(np.float32) / 255 for f in kept_frames], fps=8.0)

    meta = dict(
        source=os.path.basename(args.video),
        t_start=args.t_start, t_end=args.t_end, native_fps=fps,
        source_frame_ids=src_ids, stride=stride, n_frames=j,
        crop_left=CROP_LEFT,
        crop_bbox=dict(y0=Y0, y1=Y1, x0=X0, x1=X1),
        size=[Y1 - Y0 + 1, X1 - X0 + 1],
        motion_median=float("nan"), motion_total=float("nan"),
    )
    with open(os.path.join(out_root, seg_name, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {os.path.join(out_root, seg_name)}")


if __name__ == "__main__":
    main()
