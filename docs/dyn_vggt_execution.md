# Dyn-VGGT：執行手冊

> 所有指令在 `training/` 目錄下執行，conda env `vggt-dyn`。
> Stage 定義與設計見 [method_v3.md](dyn_vggt_method_v3.md) §8；方法沿革見 [history](dyn_vggt_history.md)。

---

## 0. 環境

```bash
conda activate vggt-dyn
cd /home/cvml-75/Desktop/vggt/training

# 確認 vggt 指向本 repo
pip show vggt | grep Editable

# 若不對：
cd /home/cvml-75/Desktop/vggt && pip install -e . --no-deps
```

---

## 1. 訓練

```bash
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_s
```

---

## 2. 抽權重

```bash
python extract_weights.py --src logs/<exp>/ckpts/checkpoint_N.pt --dst checkpoints/dyn_vggt_<name>.pt
```

> 不可直接 resume 含 `optimizer` key 的 trainer checkpoint（會 crash）。跨 stage 或 crash-recovery 都必須先抽純權重。

---

## 3. 評測

### Sintel benchmark（pose + depth）

依賴：`pip install evo opencv-python`

數據：`/home/cvml-75/Desktop/3D-repo/data/sintel/training`（`final/` + `depth/` + `camdata_left/`）

```bash
# 只需指定 checkpoint；輸出自動寫入 logs/<exp>/eval_sintel/results.json
python benchmark/eval_sintel.py --ckpt logs/<exp>/ckpts/checkpoint.pt

# 抽出的純權重 → logs/train/<name>/eval_sintel/
python benchmark/eval_sintel.py --ckpt checkpoints/dyn_vggt_s0.pt

# 單 seq smoke 加 --seq_list alley_2
```

### Gate 診斷

**數字**走 `diag/gate_eval.py`（2026-08-19 合併了舊的 `gate_quality.py` + `gate_bias_ablation.py`，
兩者共用同一次 forward）；**畫面**走 `diag/vis/gate_gif.py`。舊的 `diag/vis/gate.py` /
`gate_temporal.py` 已刪除。

```bash
# 預設：quality（AUC/F1/calibration）與 pose（off/predicted/oracle 的 ATE）都跑
python diag/gate_eval.py --ckpt checkpoints/inst_g.pt --all_seqs

# 只看 gate 準不準
python diag/gate_eval.py --ckpt logs/inst_gts/ckpts/epoch_30.pt --metrics quality --all_seqs

# PointOdyssey（只支援 pose）
python diag/gate_eval.py --ckpt checkpoints/inst_g.pt --dataset po --metrics pose

# 整段序列的 σ(g) 動畫（絕對 0..1 色階）
python diag/vis/gate_gif.py --ckpt checkpoints/inst_g.pt
```

輸出：`outputs/gate_quality/<exp>/quality_f<N>.json`、`outputs/gate_bias_ablation[_po]/<exp>/results.json`、
`outputs/gate_gif/<exp>/`。

**判讀 gate 品質一律用 AUC/F1，不要用 BCE** —— BCE 的目標是 average-pool 後的軟標籤，
boundary patch 有不可約的 floor，val BCE 會看似 overfit 但與高 AUC 並存。

### 目前結果

| 指標 | VGGT-1B | S0 | S1a | S1b | MonST3R |
|---|---|---|---|---|---|
| Depth AbsRel ↓ | 0.2747 | 0.2747 | **0.2552** | 0.2790 | 0.3450 |
| Depth δ<1.25 ↑ | 0.6832 | 0.6832 | **0.6859** | 0.6919 | 0.5620 |
| Pose ATE ↓ | 0.1714 | 0.1714 | 0.1710 | **0.1692** | 0.1080 |
| Pose RPE-trans ↓ | 0.0617 | 0.0617 | 0.0691 | 0.0770 | 0.0420 |
| Pose RPE-rot ↓ | 0.4706 | 0.4706 | **0.4792** | 0.5502 | 0.7320 |

---

## 5. 中斷與恢復

```bash
# 中斷
kill $(pgrep -f "launch.py --config <config>")

# 恢復（必須抽純權重，不能直接 resume trainer checkpoint）
python extract_weights.py --src logs/<exp>/ckpts/checkpoint_N.pt --dst checkpoints/temp_resume.pt
# 改 config 的 resume_checkpoint_path 指向 temp_resume.pt，重新啟動
torchrun --nproc_per_node=1 launch.py --config <config>
```

epoch 計數器會重置為 0，但權重是連續的。

---

## 6. 已知坑

| 問題 | 解法 |
|---|---|
| `accum_steps>1` 空 chunk crash | 設 `accum_steps: 1` |
| S1a OOM `img_nums=[8,8]` | 用 `[2,6]`（peak ~25 GB） |
| S1b OOM `img_nums≥[2,6]` | 用 `[2,4]`（peak ~27 GB） |
| trainer resume crash（含 optimizer） | 用 `extract_weights.py` 抽純權重 |
| `F.binary_cross_entropy` AMP 不允 | 已改手寫 BCE（loss.py） |
| TartanAir 無 `loss_motion` | 預期行為（靜態集無 motion_mask） |
