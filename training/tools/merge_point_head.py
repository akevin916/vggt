"""Splice a pretrained point_head into a camera-only checkpoint, producing a warm-start file.

WHY THIS EXISTS. Every SCARED arm so far ran with `model.enable_point: False`, so
`vggt.models.vggt.VGGT` never constructed a point_head and its checkpoints contain no
point_head tensors (arm 2: 1491 tensors = aggregator 1360 + camera_head 69 + depth_head 62).
Turning the head on with those weights would leave a randomly-initialised DPT head pushing
garbage gradients straight into the aggregator on step 0. VGGT-1B.pt still carries the
pretrained point_head, so this script takes the trained SCARED trunk and the pretrained head.

THE OUTPUT IS A BARE state_dict, ON PURPOSE. trainer._load_resuming_checkpoint restores
optimizer state, `steps` and `prev_epoch` whenever they are present in the file -- feeding it
arm 2's full checkpoint would start the new run at epoch 11 of a 10-epoch schedule (so it
would exit immediately) on optimizer state that does not even cover the new head's parameters.
Stripping everything but weights makes it a warm start rather than a resume.

Usage:
    python tools/merge_point_head.py \
        --src   logs/scared_cam_b16_gg_smooth_temporal/ckpts/best_ate.pt \
        --donor checkpoints/VGGT-1B.pt \
        --out   checkpoints/scared_arm2_pointhead.pt
"""

import argparse
import torch


def _state_dict(obj):
    """Accept both trainer checkpoints ({'model': sd, ...}) and bare state_dicts."""
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="checkpoint providing the trained trunk")
    ap.add_argument("--donor", required=True, help="checkpoint providing the missing module")
    ap.add_argument("--out", required=True, help="destination for the bare merged state_dict")
    ap.add_argument("--prefix", default="point_head",
                    help="module prefix copied from --donor (default: point_head)")
    args = ap.parse_args()

    src = _state_dict(torch.load(args.src, map_location="cpu", weights_only=False))
    donor = _state_dict(torch.load(args.donor, map_location="cpu", weights_only=False))

    donated = {k: v for k, v in donor.items() if k.startswith(args.prefix + ".")}
    if not donated:
        raise SystemExit(f"--donor has no tensors under '{args.prefix}.'")
    clash = sorted(set(donated) & set(src))
    if clash:
        raise SystemExit(
            f"--src already has {len(clash)} '{args.prefix}' tensors (e.g. {clash[0]}); "
            "refusing to overwrite trained weights with the donor's."
        )

    merged = dict(src)
    merged.update(donated)
    torch.save(merged, args.out)

    print(f"src   : {len(src):5d} tensors  <- {args.src}")
    print(f"donor : {len(donated):5d} tensors under '{args.prefix}.'  <- {args.donor}")
    print(f"out   : {len(merged):5d} tensors (bare state_dict)  -> {args.out}")


if __name__ == "__main__":
    main()
