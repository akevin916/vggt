# NEW: Dyn-VGGT warm-start verification. Loads pretrained VGGT-1B into the Dyn-VGGT model
#      (temporal + motion/flow heads enabled) with strict=False and checks that:
#        - missing keys  ⊆ {temporal_blocks.*, motion_head.*, flow_head.*}   (the NEW modules)
#        - unexpected keys ⊆ {track_head.*}                                  (track disabled here)
#      and that the temporal blocks warm-start as identity (LayerScale gamma == 0).
#
# Run:  python training/verify_warmstart.py [path/to/VGGT-1B.pt]
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import torch
from vggt.models.vggt import VGGT

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "training/checkpoints/VGGT-1B.pt"

model = VGGT(enable_camera=True, enable_depth=True, enable_point=True, enable_track=False,
             enable_temporal=True, enable_motion=True, enable_flow=True)

sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
missing, unexpected = model.load_state_dict(sd, strict=False)

def roots(keys):
    out = {}
    for k in keys:
        r = ".".join(k.split(".")[:2]) if k.split(".")[0] in ("aggregator",) else k.split(".")[0]
        out[r] = out.get(r, 0) + 1
    return out

print(f"ckpt: {ckpt_path}")
print(f"#missing={len(missing)}  #unexpected={len(unexpected)}")
print("missing by module   :", roots(missing))
print("unexpected by module:", roots(unexpected))

NEW = ("aggregator.temporal_blocks", "motion_head", "flow_head")
bad_missing = [k for k in missing if not k.startswith(NEW)]
bad_unexpected = [k for k in unexpected if not k.startswith("track_head")]

print("\nmissing outside NEW modules     :", bad_missing[:10] or "None  ✅")
print("unexpected outside track_head   :", bad_unexpected[:10] or "None  ✅")

# Confirm temporal blocks are identity at init (gamma == 0)
gammas = [float(b.ls1.gamma.abs().sum()) + float(b.ls2.gamma.abs().sum()) for b in model.aggregator.temporal_blocks]
print("temporal LayerScale gamma all zero:", all(g == 0 for g in gammas), f"({len(gammas)} blocks)")

ok = (not bad_missing) and (not bad_unexpected) and all(g == 0 for g in gammas)
print("\nWARM-START VERIFY:", "PASS ✅" if ok else "FAIL ❌")
