#!/usr/bin/env python3
# Slim PointOdyssey anno.npz for transfer/training.
#
# Each PO anno.npz (~215 MB/seq, ~40 GB over the train split) stores six arrays:
#   trajs_2d, trajs_3d, valids, visibs  -> point-track annotation, ~99.7% of the bytes
#   intrinsics (N,3,3), extrinsics (N,4,4) -> the ONLY keys the training loader reads
#       (see training/data/datasets/pointodyssey.py: get_data() only touches
#        anno["intrinsics"] and anno["extrinsics"]).
# Repacking to just the camera arrays takes each file from ~215 MB to ~0.2 MB, i.e.
# the whole split's anno from ~40 GB to ~40 MB -- a big win before you tar + upload.
#
# By default this writes the slimmed copies into a PARALLEL output tree (source is left
# untouched), mirroring <seq>/anno.npz so you can tar it as a small standalone archive
# and extract it over the image/depth archive on the target machine. Pass --in-place to
# overwrite the source anno.npz instead (destructive: the track arrays are gone afterwards).
#
# Usage:
#   python po_slim_anno.py                       # train split -> ./_slim_anno tree, verify
#   python po_slim_anno.py --splits train test
#   python po_slim_anno.py --out /path/to/slim_anno
#   python po_slim_anno.py --in-place            # overwrite source (careful!)

import argparse
import os
import os.path as osp
import glob
import sys

import numpy as np

KEEP_KEYS = ("intrinsics", "extrinsics")


def slim_one(src_path: str, dst_path: str, compress: bool) -> tuple[int, int]:
    """Read src anno.npz, write only KEEP_KEYS to dst. Returns (src_bytes, dst_bytes)."""
    with np.load(src_path, allow_pickle=True) as a:
        missing = [k for k in KEEP_KEYS if k not in a.files]
        if missing:
            raise KeyError(f"{src_path} missing keys {missing} (has {a.files})")
        payload = {k: np.asarray(a[k]) for k in KEEP_KEYS}

    os.makedirs(osp.dirname(dst_path) or ".", exist_ok=True)
    saver = np.savez_compressed if compress else np.savez
    # write to a temp file then atomically rename, so an interrupted run never leaves
    # a half-written anno.npz (which the loader would choke on).
    tmp_path = dst_path + ".tmp"
    saver(tmp_path if tmp_path.endswith(".npz") else tmp_path, **payload)
    # np.savez appends .npz if the name lacks it; normalize.
    written = tmp_path if osp.exists(tmp_path) else tmp_path + ".npz"
    os.replace(written, dst_path)
    return os.path.getsize(src_path), os.path.getsize(dst_path)


def main():
    ap = argparse.ArgumentParser(description="Slim PointOdyssey anno.npz to camera-only arrays.")
    ap.add_argument("--po_dir", default="/media/cvml-75/ssd2t1/data/point_odyssey",
                    help="PointOdyssey root (contains train/ test/).")
    ap.add_argument("--splits", nargs="+", default=["train"], help="Splits to process.")
    ap.add_argument("--out", default=None,
                    help="Output root for the slimmed parallel tree "
                         "(default: <po_dir>/_slim_anno). Ignored with --in-place.")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite source anno.npz instead of writing a parallel tree (DESTRUCTIVE).")
    ap.add_argument("--no-compress", action="store_true",
                    help="Use np.savez (uncompressed) instead of np.savez_compressed.")
    ap.add_argument("--dry-run", action="store_true", help="List what would be done, write nothing.")
    args = ap.parse_args()

    compress = not args.no_compress
    out_root = args.out or osp.join(args.po_dir, "_slim_anno")

    tot_src = tot_dst = 0
    n_ok = n_skip = n_err = 0

    for split in args.splits:
        split_dir = osp.join(args.po_dir, split)
        if not osp.isdir(split_dir):
            print(f"[skip] split dir not found: {split_dir}")
            continue
        anno_paths = sorted(glob.glob(osp.join(split_dir, "*", "anno.npz")))
        print(f"[{split}] {len(anno_paths)} anno.npz found")

        for src in anno_paths:
            seq = osp.basename(osp.dirname(src))
            if args.in_place:
                dst = src
            else:
                dst = osp.join(out_root, split, seq, "anno.npz")

            if args.dry_run:
                print(f"  {seq}: {src} -> {dst}")
                n_ok += 1
                continue

            try:
                sb, db = slim_one(src, dst, compress)
                tot_src += sb
                tot_dst += db
                n_ok += 1
                print(f"  {seq}: {sb/1e6:7.1f} MB -> {db/1e6:5.2f} MB")
            except Exception as e:  # keep going; report at the end
                n_err += 1
                print(f"  [ERR] {seq}: {e}", file=sys.stderr)

    print("-" * 60)
    print(f"done: ok={n_ok} err={n_err}")
    if not args.dry_run and tot_src:
        print(f"anno bytes: {tot_src/1e9:.2f} GB -> {tot_dst/1e9:.3f} GB "
              f"({100*(1-tot_dst/tot_src):.1f}% smaller)")
    if not args.in_place and not args.dry_run:
        print(f"slimmed tree: {out_root}")
        print("  -> tar this tree and extract it OVER the image/depth archive on the target,")
        print("     so <seq>/anno.npz lands next to rgbs/ depths/ dynmask_inst/.")


if __name__ == "__main__":
    main()
