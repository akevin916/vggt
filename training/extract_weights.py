# NEW: Extract pure model state_dict from a trainer checkpoint (which also contains optimizer/epoch/scaler).
#      Used to pass weights between training stages (S0→S1→S2) without hitting the upstream trainer
#      resume bug (trainer.py:216 self.optims.optimizer on a list) and to start each stage with a
#      fresh optimizer (which is the correct curriculum behavior anyway).
#
# Usage:
#   python extract_weights.py --src logs/inst_g/ckpts/epoch_15.pt --dst checkpoints/inst_g.pt
import argparse, torch, os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="trainer checkpoint (contains 'model', 'optimizer', 'epoch', ...)")
ap.add_argument("--dst", required=True, help="output path for pure state_dict")
args = ap.parse_args()

ck = torch.load(args.src, map_location="cpu", weights_only=False)
if isinstance(ck, dict) and "model" in ck:
    sd = ck["model"]
    epoch = ck.get("epoch", "?")
    print(f"extracted model state_dict from trainer checkpoint (epoch {epoch})")
    extra_keys = [k for k in ck if k != "model"]
    print(f"  dropped keys: {extra_keys}")
else:
    sd = ck
    print("checkpoint is already a raw state_dict, copying as-is")

print(f"  #params: {len(sd)}")
os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
torch.save(sd, args.dst)
print(f"  saved to {args.dst} ({os.path.getsize(args.dst)/1e9:.2f} GB)")
