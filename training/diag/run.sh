#!/usr/bin/env bash
# Gate-bias ablation sweep over frame budgets on Sintel, WITH temperature scaling
# (predicted_x{T}) to split gate "under-confidence" from "wrong pattern".
# Run from the training/ directory:  bash diag/run.sh
set -euo pipefail

CKPT="logs/dyn_vggt_v3_s1_inst/ckpts/best.pt"        # clean run 1, epoch 15
# One dir per frame budget: a sweep legitimately needs the setting in the path, since
# gate_bias_ablation.py always writes results.json.
OUT_ROOT="../outputs/dyn_vggt_v3_s1_inst/gate_sweep_scales"

for F in 4 8 16 32; do
    echo "=== max_frames=chunk_size=${F} ==="
    python diag/gate_bias_ablation.py --dataset sintel \
        --ckpt "${CKPT}" --all_seqs \
        --max_frames "${F}" --chunk_size "${F}" --require_gate \
        --scales 3.0 10.0 \
        --out_dir "${OUT_ROOT}/f${F}"
done
