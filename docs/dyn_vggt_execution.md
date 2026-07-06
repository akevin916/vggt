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

# OOM 時加 --chunk_size 32；單 seq smoke 加 --seq_list alley_2
```

### Gate 診斷（視覺化為主）

主入口 `diag/vis_gate.py`：輸出 PO 的 `m_gt | m*_patch | g` 拼圖，或 Sintel 的 `m*_raft_patch`（flow-residual）對照。

```bash
# PO in-domain（預設）
python diag/vis_gate.py --ckpt logs/dyn_vggt_v3_s1/ckpts/checkpoint.pt

# Sintel cross-domain
python diag/vis_gate.py --ckpt checkpoints/dyn_vggt_v3_s1.pt --dataset sintel

# 兩者都跑，並寫入最小 summary.json
python diag/vis_gate.py --ckpt checkpoints/dyn_vggt_v3_s1.pt --dataset all --metrics
```

輸出目錄：`logs/<exp>/vis_gate/`（`po/`、`sintel/` 子目錄）。

終端會印簡短診斷：`σ(g)` 在 dynamic/static patch 的分離度（gap）、acc@0.5。加 `--metrics` 才寫 `summary.json`。

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
