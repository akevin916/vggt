"""Cut contiguous stereo sequences out of the raw lesion videos (會動1/2/3.mkv).

WHY THIS EXISTS. The shipped lesion folders (病灶1/2/3) hold 12/12/19 frames sampled at
stride ~29 -- one frame per second. Measured median optical flow per step is 41.6 / 74.7 /
20.9 px on the 992-px eye (4.2% / 7.5% / 2.1% of frame width), so consecutive "frames" are
nearly independent views. Anything that reads the folder as a video is reading a slideshow.

SAMPLING, decided 2026-08-25 from the measured flow-vs-gap curve:

    gap (ms)      33    67   100   167   267   500
    病灶1 (px)   4.1   7.2  10.5  13.0  18.5  31.4
    病灶2 (px)   5.1  10.5  14.9  22.4  28.0  46.5
    病灶3 (px)   3.6   5.0   6.4   8.6  11.3  11.9

  * STRIDE 3 (10 fps), not a fixed frame count. A fixed RATE keeps dt identical across the
    three folders, which is what the temporal blocks' 1D RoPE actually sees; a fixed count
    would give each folder a different dt because the videos differ in length.
  * 80 FRAMES, because VGGT tops out near 80 at 518 px in one pass and any cross-frame
    metric computed over stitched chunks is dominated by the seams (see
    eval_utils.vggt_infer.infer_sequence_chunked). 80 x stride 3 = 240 raw frames = 8.0 s.
  * FROM FRAME 0, so index 0 still corresponds to the shipped 00000_L.png (verified: the
    video's first left half matches that file to within resize error).

EQUAL-MOTION SAMPLING (--sampling motion), added 2026-08-25 after measuring that uniform
time sampling spends most of the budget on nothing. Per-frame median flow over 會動1 shows
56.2% of its frames move less than 1 px; the motion arrives in bursts (3-4 s = 25.2% of the
whole clip's motion, 5-6 s = 20.3%, 8-9 s = 16.3%) separated by dead stretches (7-8 s runs
at 0.14 px/frame -- the endoscope is parked). Uniform stride 3 x 64 therefore covered only
77.5% / 56.6% / 39.3% of each clip's total motion while burning ~10 frames on a static
7-8 s. Equal-motion instead walks the CUMULATIVE flow curve and takes 64 points evenly
along it: static stretches collapse to a frame or two, bursts get sampled densely, and
every step carries the same parallax.

  This is defensible because the temporal RoPE is fed torch.arange(S) (aggregator.py:379) --
  the model sees ORDER, never elapsed time. What should be held constant across a step is
  therefore the motion, not the seconds. Side effect: input.mp4 plays back at a variable
  apparent speed, and the three folders no longer share a wall-clock span.

  Measured cost at 64 frames covering each clip end to end: 22.4 / 24.4 / 36.3 px per step
  (病灶1/2/3) against a total clip motion of 1410 / 1540 / 2288 px. 病灶3 is the coarsest
  because it is 18 s long against the others' 11 s.

Both eyes are written: the pair PSNR in benchmark/eval_lesion.py needs L and R at the same
frame index. Frames are stored at the native 992x992 per eye (the shipped set is a 512x512
downscale of exactly this), so nothing is thrown away here.

⚠️ The video is SIDE-BY-SIDE stereo, 1984x992. Left half = left eye -- verified per video by
comparing both halves against the shipped 00000_L.png (|diff| 0.7 vs 22-48).

Output goes to data/eval/lesion_seq/ -- a SIBLING of the shipped set, not inside it, because
benchmark/eval_lesion.py's list_folders() treats every directory under its root as a lesion
folder. Score it with:

    python benchmark/eval_lesion.py --lesion_root ../data/eval/lesion_seq --ckpts ...

Usage:
    cd training
    python data/preprocess/lesion_extract_seq.py --sampling motion --n_frames 64
    python data/preprocess/lesion_extract_seq.py --stride 1 --n_frames 80 --out_name dense

⚠️ 64 IS A HARD CEILING, measured not guessed: on these square 992x992 eyes (518x518 =
1369 patches, 23% more than SCARED's 518x420) the gate arms OOM at 72 and fit at 64 with a
28.26 GiB peak on the 32 GB card. scared_cam_vanilla -- no gate, no temporal -- fits 80.
All arms must share one frame count or their reconstructions cannot be put side by side.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.paths import data_path

# 會動N.mkv -> 病灶N. Correspondence verified by matching frame 0's left half against each
# folder's shipped 00000_L.png.
PAIRS = [("會動1.mkv", "病灶1"), ("會動2.mkv", "病灶2"), ("會動3.mkv", "病灶3")]


def motion_indices(video_path, n_frames, probe=256, f0=0, f1=None, max_raw_gap=None):
    """Frame ids spaced evenly along the CUMULATIVE optical-flow curve.

    Pass 1 of two: reads the whole clip at ``probe`` x ``probe`` grayscale (the full-res
    frames would be gigabytes) and integrates the per-frame median flow. Picking 64 points
    at equal cumulative flow makes every kept step carry the same parallax, which is what
    the model's arange-based temporal positions actually assume.
    """
    cap = cv2.VideoCapture(video_path)
    small, prev = [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h = f.shape[0]
        g = cv2.cvtColor(cv2.resize(f[:, :f.shape[1] // 2], (probe, probe)), cv2.COLOR_BGR2GRAY)
        small.append(g)
    cap.release()
    # A time window is applied HERE, before the cumulative curve is built, so the equal-motion
    # spacing is computed over the window's own motion rather than the whole clip's. Frame ids
    # returned are still absolute (offset by f0).
    f1 = len(small) if f1 is None else min(f1, len(small))
    f0 = max(0, min(f0, f1 - 1))
    small = small[f0:f1]
    n = len(small)
    if n < 2:
        return list(range(f0, f0 + n)), None

    step = []
    for i in range(n - 1):
        fl = cv2.calcOpticalFlowFarneback(small[i], small[i + 1], None,
                                          0.5, 3, 25, 3, 5, 1.2, 0)
        step.append(float(np.median(np.linalg.norm(fl, axis=-1))))
    cum = np.concatenate([[0.0], np.cumsum(step)])
    total = float(cum[-1])
    if total <= 0:
        return [f0 + i for i in range(0, n, max(1, n // n_frames))][:n_frames], None

    # Two-criterion greedy walk: take a frame when the accumulated flow since the last pick
    # reaches the target OR when max_raw_gap raw frames have gone by, whichever comes first.
    #
    # The gap criterion exists because median optical flow UNDER-reports what a near-static
    # stretch actually does. On 病灶3, equal-motion alone collapsed the 8-10 s lull into two
    # steps that each skipped 34 and 17 raw frames; their flow was an unremarkable ~30 px but
    # the picture jumps, because rotation and moving specular highlights change the image
    # without moving the median pixel much.
    #
    # The flow target is found by bisection so the count lands exactly on n_frames: raising it
    # can only remove picks, so the count is monotone in the target and the search is safe.
    # (Without the gap rule a plain linspace over the cumulative curve would do, which is what
    # this used to be -- see git history.)
    def walk(target):
        picks, last = [0], 0
        for i in range(1, n):
            if (cum[i] - cum[last]) >= target or (max_raw_gap and (i - last) >= max_raw_gap):
                picks.append(i)
                last = i
        return picks

    if max_raw_gap:
        lo, hi = 0.0, total
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if len(walk(mid)) > n_frames:
                lo = mid
            else:
                hi = mid
        ids = walk(hi)
        # bisection lands on <= n_frames; top up with the widest remaining gaps so the count
        # is exact rather than silently short.
        while len(ids) < n_frames:
            gaps = np.diff(ids)
            k = int(np.argmax(gaps))
            if gaps[k] < 2:
                break
            ids = sorted(set(ids + [ids[k] + int(gaps[k]) // 2]))
        ids = ids[:n_frames]
    else:
        # Monotone greedy walk over evenly spaced cumulative-flow targets. Inside a burst
        # several targets can land on the same raw frame (one 33 ms step can already exceed
        # the target spacing); forcing each pick to advance by at least one frame keeps the
        # count exact and the ids increasing.
        targets = np.linspace(0.0, total, n_frames)
        ids, prev = [], -1
        for k, t in enumerate(targets):
            j = int(np.abs(cum - t).argmin())
            j = max(j, prev + 1)
            j = min(j, n - (n_frames - k))
            ids.append(j)
            prev = j
    kept = np.array(ids)
    ids = [f0 + j for j in ids]
    per_step = np.diff(cum[kept])
    stats = dict(
        clip_total_flow_px=round(total, 2),
        step_flow_px_mean=round(float(per_step.mean()), 2),
        step_flow_px_min=round(float(per_step.min()), 2),
        step_flow_px_max=round(float(per_step.max()), 2),
        raw_gap_min=int(np.diff(kept).min()), raw_gap_max=int(np.diff(kept).max()),
        probe_size=probe, window_raw=[f0, f1], max_raw_gap=max_raw_gap,
    )
    return ids, stats


def extract(video_path, out_dir, stride, n_frames, sampling="uniform",
            t_start=None, t_end=None, max_raw_gap=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half = width // 2

    f0 = 0 if t_start is None else int(round(t_start * fps))
    f1 = None if t_end is None else int(round(t_end * fps))
    if sampling == "motion":
        wanted, flow_stats = motion_indices(video_path, n_frames, f0=f0, f1=f1,
                                            max_raw_gap=max_raw_gap)
    else:
        hi = min(total if f1 is None else f1, f0 + stride * n_frames)
        wanted, flow_stats = list(range(f0, hi, stride))[:n_frames], None
    want = set(wanted)
    os.makedirs(out_dir, exist_ok=True)

    written, raw_idx, out_idx = [], 0, 0
    while raw_idx <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if raw_idx in want:
            cv2.imwrite(os.path.join(out_dir, f"{out_idx:05d}_L.png"), frame[:, :half])
            cv2.imwrite(os.path.join(out_dir, f"{out_idx:05d}_R.png"), frame[:, half:])
            written.append(raw_idx)
            out_idx += 1
        raw_idx += 1
    cap.release()

    meta = dict(
        source_video=os.path.basename(video_path),
        video_frames=total, video_fps=fps, video_size=[width, height],
        eye_size=[half, height],
        sampling=sampling,
        stride=(stride if sampling == "uniform" else None),
        n_frames=len(written),
        effective_fps=(fps / stride if sampling == "uniform" else None),
        flow_stats=flow_stats,
        source_frame_ids=written,
        span_frames=(written[-1] - written[0] + 1) if written else 0,
        span_seconds=round(((written[-1] - written[0]) / fps), 3) if written else 0.0,
        contiguous_stride=True,
        note=(("equal-MOTION cut: frame ids are spaced evenly along the cumulative optical-flow "
               "curve, so wall-clock spacing is deliberately non-uniform (static stretches "
               "collapse, bursts get sampled densely). "
               if sampling == "motion" else "stride-%d uniform-time cut. " % stride)
              + "Index 0 == video frame 0 == the shipped set's 00000_L.png. "
                "Left half of the side-by-side frame is the LEFT eye."),
        created=datetime.now().isoformat(timespec="seconds"),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion_root", default=data_path("eval", "lesion"),
                    help="directory holding 會動N.mkv")
    ap.add_argument("--out_root", default=None,
                    help="default: <lesion_root>/../lesion_seq")
    ap.add_argument("--sampling", default="uniform", choices=["uniform", "motion"],
                    help="uniform = every --stride raw frames. motion = evenly spaced along "
                         "the cumulative optical-flow curve (see module docstring)")
    ap.add_argument("--stride", type=int, default=3,
                    help="raw frames between kept frames; 3 = 10 fps (see module docstring)")
    ap.add_argument("--n_frames", type=int, default=80,
                    help="frames per folder; 80 is VGGT's single-pass ceiling at 518 px")
    ap.add_argument("--folders", nargs="*", default=None,
                    help="restrict to these folder names, e.g. 病灶3")
    ap.add_argument("--t_start", type=float, default=None,
                    help="window start in seconds of the RAW video (default: 0)")
    ap.add_argument("--t_end", type=float, default=None,
                    help="window end in seconds of the RAW video (default: end of clip). "
                         "Use this when a clip's total motion is too large for --n_frames: "
                         "病灶3 spends 2288 px over 18 s, which at 64 frames is 36.3 px per "
                         "step and reconstructs badly, while 病灶2's 24.5 px per step is clean.")
    ap.add_argument("--max_raw_gap", type=int, default=None,
                    help="motion sampling only: never skip more than this many raw frames, "
                         "even through a stretch the flow metric calls static. Fixes the "
                         "visible jump equal-motion leaves in a lull (see module docstring)")
    ap.add_argument("--out_name", default=None,
                    help="sub-directory under out_root; default derives from stride/n_frames")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(os.path.dirname(args.lesion_root.rstrip("/")),
                                             "lesion_seq")
    if args.out_name:
        out_root = os.path.join(out_root, args.out_name)

    print(f"stride={args.stride} n_frames={args.n_frames} -> {out_root}\n")
    for video, folder in PAIRS:
        if args.folders and folder not in args.folders:
            continue
        vp = os.path.join(args.lesion_root, video)
        if not os.path.exists(vp):
            print(f"  SKIP {folder}: {vp} 不存在")
            continue
        m = extract(vp, os.path.join(out_root, folder), args.stride, args.n_frames,
                    sampling=args.sampling, t_start=args.t_start, t_end=args.t_end,
                    max_raw_gap=args.max_raw_gap)
        head = (f"  {folder:8s} {m['n_frames']:3d} 幀  "
                f"raw[{m['source_frame_ids'][0]}..{m['source_frame_ids'][-1]}]  "
                f"跨度 {m['span_seconds']:.1f}s")
        if m.get("flow_stats"):
            f = m["flow_stats"]
            probe_scale = m["eye_size"][0] / f["probe_size"]
            print(head + f"  每步光流 {f['step_flow_px_mean']*probe_scale:5.1f} px "
                         f"(min {f['step_flow_px_min']*probe_scale:4.1f} / "
                         f"max {f['step_flow_px_max']*probe_scale:5.1f})  "
                         f"raw gap {f['raw_gap_min']}-{f['raw_gap_max']}")
        else:
            print(head + f"  {m['effective_fps']:.1f} fps  "
                         f"單眼 {m['eye_size'][0]}x{m['eye_size'][1]}")
    print(f"\n完成。評分用：\n  python benchmark/eval_lesion.py "
          f"--lesion_root {os.path.relpath(out_root, os.path.join(os.path.dirname(__file__), '..', '..'))} --ckpts ...")


if __name__ == "__main__":
    main()
