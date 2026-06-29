# Dyn-VGGT：執行手冊

> 所有指令均在 `training/` 目錄下執行，使用 `vggt-dyn` conda 環境。
> 本文件讓你可以自行啟動、重啟、驗證、以及推進各訓練階段，不需要 Claude。

---

## 0. 環境前置

```bash
# 切換環境
conda activate vggt-dyn

# 確認 vggt 指向正確 repo（應顯示 /home/cvml-75/Desktop/vggt）
pip show vggt | grep Editable

# 若指向錯誤：
cd /home/cvml-75/Desktop/vggt && pip install -e . --no-deps

# 工作目錄（所有後續指令都在此執行）
cd /home/cvml-75/Desktop/vggt/training
```

---

## 1. 檔案清單

| 類型 | 路徑 | 說明 |
|---|---|---|
| **Config** | `config/dyn_vggt_po_s0.yaml` | S0: 凍 backbone，只訓 motion/flow；`val_metrics: false` |
| | `config/dyn_vggt_s1.yaml` | **S1 正式**：凍 aggregator、5 頭 + 7 loss、PO+TA |
| | `config/dyn_vggt_s1_smoke.yaml` | S1 smoke（20 batch，新策略） |
| | `config/dyn_vggt_s2.yaml` | **S2**：temporal+global+5 頭、PO+TA+Waymo |
| | `config/dyn_vggt_s2_smoke.yaml` | S2 記憶體 smoke（750M @ max_img=4） |
| | `config/dyn_vggt_s3.yaml` | S3 可選（patch_embed 精修） |
| | `config/dyn_vggt_po.yaml` | ⚠️ legacy 舊 S1；改用 `dyn_vggt_s1.yaml` |
| | `config/dyn_vggt_po_s0_smoke.yaml` | S0 極短 smoke test（3 batch × 1 epoch） |
| **權重** | `checkpoints/VGGT-1B.pt` | 預訓練 VGGT-1B（原始 warm-start） |
| | `checkpoints/dyn_vggt_s0.pt` | S0 訓練完的純權重（epoch 7，已驗證） |
| **腳本** | `verify_warmstart.py` | 驗證 checkpoint 載入（missing/unexpected keys + γ=0） |
| | `eval_s0.py` | S0 驗證（motion IoU/F1 + flow 改善 + 凍結完好 + 質性圖） |
| | `eval_sintel.py` | Sintel pose（ATE/RPE）+ depth（AbsRel/δ）評估 + S0 vs baseline gate |
| | `eval/` | Sintel I/O、VGGT 推理、pose/depth 指標模組 |
| | `extract_weights.py` | 從 trainer checkpoint 抽純權重（跨 stage 傳遞用） |
| **訓練輸出** | `logs/<exp_name>/` | **每個 config 獨立目錄**（由 `exp_name` 決定） |
| | `logs/<exp_name>/log.txt` | trainer 主 log（`model.txt`、`frozen.txt`、`trainable.txt` 同目錄） |
| | `logs/<exp_name>/tensorboard/` | TensorBoard |
| | `logs/<exp_name>/ckpts/` | 每 epoch checkpoint |
| | `logs/<exp_name>/run.log` | 建議 nohup 重定向路徑（手動） |
| | `logs/dyn_vggt_po_s0/eval/` | S0 驗證質性圖（`eval_s0.py --save_dir`） |
| **數據** | `/media/cvml-75/ssd2t1/data/point_odyssey` | PointOdyssey（train 131 seq / test 13 seq） |
| | `/media/cvml-75/ssd2t1/data/tartanair` | TartanAir V1 train（S1 靜態混入，無 motion_mask） |

---

## 2. S0：對齊新頭（已完成 ✅）

### 2.1 訓練

```bash
# 正式訓練（12 epoch，每 epoch 存 checkpoint，~10 min/epoch）
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_po_s0

# 極短 smoke test（確認能跑通，3 batch × 1 epoch）
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_po_s0_smoke

# 背景執行（過夜）— stdout 也寫進該實驗目錄
nohup torchrun --nproc_per_node=1 --master_port=29606 launch.py --config dyn_vggt_po_s0 \
  > logs/dyn_vggt_po_s0/run.log 2>&1 &

# 監控（trainer 結構化 log 在 log.txt；nohup 捕獲的 stdout 在 run.log）
tail -f logs/dyn_vggt_po_s0/log.txt | grep -E "Train Epoch|Val Epoch|Saving|Error"
```

### 2.2 驗證

```bash
# 完整 S0 驗證（V1 motion IoU/F1 + V2 flow 改善 + V3 凍結完好 + V5 質性圖）
python eval_s0.py --ckpt logs/dyn_vggt_po_s0/ckpts/checkpoint_7.pt --n_clips 20 --img_per_seq 6

# 可換任意 checkpoint（比較不同 epoch）
python eval_s0.py --ckpt logs/dyn_vggt_po_s0/ckpts/checkpoint_3.pt --n_clips 20
```

**Pass bar（S0 已達標）**：
- V1: IoU ≳ 0.5（實際 0.733）、動態均值 ≫ 靜態均值（0.849 vs 0.221）
- V2: 動態區誤差下降（實際 +3.8%）、靜態區不變
- V3: max|Δ| ≈ 0（實際 0.00）

### 2.3 抽權重（給 S1 用）

```bash
# 從 trainer checkpoint 抽出純 state_dict（去掉 optimizer/scaler/epoch）
python extract_weights.py \
  --src logs/dyn_vggt_po_s0/ckpts/checkpoint_7.pt \
  --dst checkpoints/dyn_vggt_s0.pt
```

### 2.4 Sintel pose + depth 評估（S0 regression gate）

在 PointOdyssey 上驗證 motion/flow 後，用 **Sintel 14 seq** 量測 pose / depth 是否與 VGGT-1B 一致（S0 凍結幾何頭，預期相同）。

**依賴**（eval 環境需有 `evo`）：

```bash
pip install evo
```

**數據**：`/home/cvml-75/Desktop/3D-repo/data/sintel/training`（`final/` + `depth/` + `camdata_left/`）

**執行**（在 `training/` 下）：

```bash
# 1) VGGT-1B baseline（14 seq，~40s）
python eval_sintel.py \
  --ckpt checkpoints/VGGT-1B.pt \
  --variant vggt_base \
  --out_dir logs/dyn_vggt_po_s0/eval_sintel

# 2) Dyn-VGGT S0
python eval_sintel.py \
  --ckpt checkpoints/dyn_vggt_s0.pt \
  --variant s0 \
  --out_dir logs/dyn_vggt_po_s0/eval_sintel

# 3) Regression gate（S0 vs baseline）
python eval_sintel.py \
  --compare logs/dyn_vggt_po_s0/eval_sintel/results_vggt_base.json \
            logs/dyn_vggt_po_s0/eval_sintel/results_s0.json \
  --out_dir logs/dyn_vggt_po_s0/eval_sintel

# 4) 可選：逐位元檢查（1 seq × 6 frames）
python eval_sintel.py \
  --ckpt checkpoints/dyn_vggt_s0.pt --variant s0 \
  --bitwise_ckpt checkpoints/VGGT-1B.pt --bitwise_variant vggt_base \
  --seq_list alley_2
```

**輸出**：

| 檔案 | 內容 |
|---|---|
| `eval_sintel/results_vggt_base.json` | baseline per-seq + mean |
| `eval_sintel/results_s0.json` | S0 per-seq + mean |
| `eval_sintel/summary_*.md` | markdown 表 |
| `eval_sintel/compare_gate.json` / `.md` | S0 vs baseline PASS/FAIL |

**Pass bar（S0 已達標）**：

- Pose：`\|ATE_s0 − ATE_base\| / ATE_base < 1%` 或 abs < 0.01 m
- Depth：`\|AbsRel_s0 − AbsRel_base\| < 0.001`
- Bitwise：`max|Δdepth|`, `max|Δpose_enc| < 1e-2`（實測 0.00）

**實測結果（2026-06-26，14 seq mean）**：

| Method | ATE | RPE-trans | RPE-rot | AbsRel | δ<1.25 |
|--------|-----|-----------|---------|--------|--------|
| VGGT-1B | 0.1714 | 0.0617 | 0.4706 | 0.2747 | 0.6832 |
| Dyn-VGGT S0 | 0.1714 | 0.0617 | 0.4706 | 0.2747 | 0.6832 |

Regression gate：**PASS**（S0 與 baseline 完全一致）。

> OOM 時加 `--chunk_size 32` 分段推理。單 seq smoke：`--seq_list alley_2`。

---

## 3. S1：運動解耦（凍 backbone，5 頭 + 全 loss）

### 3.1 執行前確認

```bash
# 確認 S0 純權重存在
ls -lh checkpoints/dyn_vggt_s0.pt

# 確認 S1 config 的 resume 指向 S0 純權重（不是 trainer checkpoint！）
grep resume_checkpoint_path config/dyn_vggt_s1.yaml
# 應顯示：checkpoints/dyn_vggt_s0.pt

# warm-start 驗證（可選）
python verify_warmstart.py checkpoints/dyn_vggt_s0.pt

# 確認 TartanAir 路徑存在（S1 train mix 需要）
ls /media/cvml-75/ssd2t1/data/tartanair/train | head
```

### 3.2 S1 config 要點（`dyn_vggt_s1.yaml`）

| 項目 | S0 | S1（現行） |
|---|---|---|
| 解凍 | 只 motion/flow 頭（65M） | **5 頭全解凍**（~150–200M） |
| 凍結 | 全 aggregator + 原三頭 | **全 aggregator**（embed + frame + temporal + global） |
| 啟用 loss | motion + flow | **全 7 項**（camera + depth + point + motion + flow + reproj + tsmooth） |
| `mask_dynamic` | off | **on** |
| `val_metrics` | **false** | **true**（depth/pose 在此 stage 才有意義） |
| resume | `VGGT-1B.pt` | **`dyn_vggt_s0.pt`** |
| 數據 | PointOdyssey | **PO ~70% + TartanAir ~30%**（`len_train` 100000:43000） |
| val | PO test | **僅 PO test** |
| `max_img_per_gpu` | 10 | **6**（smoke：`[8,8]` OOM，`[6,6]` peak ~25GB） |
| `img_nums` | [2, 10] | **[2, 6]** |
| `accum_steps` | 1 | **1** |
| `limit_train_batches` | 800 | **2000** |
| `max_epochs` | 8–12 | **20** |

**TartanAir batch 行為**：無 `motion_mask` → `L_motion` 自動跳過；其餘 6 項仍監督（靜態 m≈0、Δ≈0）。

### 3.3 Smoke test（✅ 已通過）

```bash
# 管線 + 記憶體確認（~1–2 分鐘，20 train batch）
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_s1_smoke
```

Pass bar：`img_nums=[6,6]` peak **~25GB/31GB**；`[8,8]` OOM。20 train step + val + `val_metrics` 正常。

### 3.4 正式訓練

```bash
# 正式訓練（20 epoch，limit 2000 train batch/epoch）
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_s1

# 背景執行
nohup torchrun --nproc_per_node=1 --master_port=29610 launch.py --config dyn_vggt_s1 \
  > logs/dyn_vggt_s1/run.log 2>&1 &

# 監控 loss
tail -f logs/dyn_vggt_s1/log.txt | grep -E "loss_motion|loss_reproj|loss_camera|Error"

# 監控 val 指標
grep "Val Epoch.*metrics" logs/dyn_vggt_s1/log.txt
```

### 3.5 S1 完成後：抽權重給 S2

```bash
python extract_weights.py \
  --src logs/dyn_vggt_s1/ckpts/checkpoint_BEST.pt \
  --dst checkpoints/dyn_vggt_s1.pt
```

---

## 4. S2：時序建模（temporal + global）

Config：`config/dyn_vggt_s2.yaml`（✅ 已建立）

| 項目 | S1 | S2 |
|---|---|---|
| 凍結 | 全 aggregator | **patch_embed + frame_blocks** |
| 解凍 | 5 頭 | **temporal + global + 5 頭**（~750M） |
| loss | 7 項 + `mask_dynamic` | 同 S1 |
| 數據 | PO + TA | **PO + TA + Waymo**（Spring 待 loader） |
| `max_img_per_gpu` | 6 | **4** |
| `img_nums` | [2, 6] | **[2, 4]** |
| resume | `dyn_vggt_s0.pt` | **`dyn_vggt_s1.pt`** |

```bash
# S2 smoke（記憶體 gate）
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_s2_smoke

# 正式 S2
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_s2
```

Gate：時序一致↑；Sintel eval（`eval_sintel.py`）；靜態 benchmark 無退化。

---

## 5. 通用工具

### 5.1 warm-start 驗證

```bash
# 驗證任意 checkpoint 載入 Dyn-VGGT 模型（檢查 missing/unexpected keys + temporal γ=0）
python verify_warmstart.py [path/to/checkpoint.pt]

# 預設路徑：checkpoints/VGGT-1B.pt
python verify_warmstart.py
```

### 5.2 抽權重（跨 stage 傳遞）

```bash
# 通用：從任何 trainer checkpoint 抽出純 state_dict
python extract_weights.py --src <trainer_ckpt.pt> --dst <output.pt>
```

### 5.3 查看訓練進度（不中斷訓練）

```bash
# 最近幾個 batch 的 loss（trainer 寫入）
tail -20 logs/<exp_name>/log.txt | grep "Train Epoch"

# 每 epoch 的 val loss 趨勢
grep "Val Epoch.*loss_motion" logs/<exp_name>/log.txt | tail -12

# Val depth/pose 指標摘要（AbsRel、ATE 等，需 val_metrics.enabled: true）
grep "Val Epoch.*metrics" logs/<exp_name>/log.txt
```

TensorBoard：`Metrics/val/abs_rel`、`Metrics/val/ate` 等（每 val epoch 一個點）。

```bash
# 已存 checkpoint
ls -lt logs/<exp_name>/ckpts/

# GPU 記憶體
nvidia-smi

# 進程是否存活
pgrep -af "launch.py --config"
```

### 5.4 中斷與恢復

```bash
# 中斷訓練
kill $(pgrep -f "launch.py --config dyn_vggt_s1")

# 恢復（重要：不能直接 resume trainer checkpoint，會撞 optimizer bug）
# 步驟：
#   1. 找到最後一個好的 checkpoint
ls -lt logs/<exp_name>/ckpts/
#   2. 抽純權重
python extract_weights.py --src logs/<exp_name>/ckpts/checkpoint_N.pt --dst checkpoints/temp_resume.pt
#   3. 改 config 的 resume_checkpoint_path 指向它
#   4. 重新啟動（會從 epoch 0 開始，但權重是已訓練的）
torchrun --nproc_per_node=1 launch.py --config <config_name>
```

> ⚠️ **已知限制**：upstream trainer 的 `_load_resuming_checkpoint` 在載入含 `"optimizer"` key 的 checkpoint 時會 crash（`self.optims.optimizer` 是 list）。所以**跨 stage 或 crash-recovery 都必須先 `extract_weights.py` 抽純權重**，再用新 config warm-start。epoch 計數器會重置為 0，但權重是連續的。

---

## 6. 已知坑與修正記錄

| 問題 | 觸發條件 | 修正 |
|---|---|---|
| `accum_steps>1` 空 chunk crash | `batch_size=1`（`max_img=4`, S=2–4）+ `accum_steps>1` | 設 `accum_steps: 1` |
| S1 OOM @518² | 5 頭 @ `img_nums=[8,8]` 或 S>6 | 用 `max_img_per_gpu: 6`、`img_nums: [2,6]`（smoke peak ~25GB） |
| S2 OOM @518² | 750M trainable + S≥6 | `max_img_per_gpu: 4`、`img_nums: [2,4]`；凍 frame_blocks |
| OOM @518²（S0） | 4 個 DPT head + `max_img_per_gpu≥16` | 降 `max_img_per_gpu`（S0 實測 10 → ~25GB） |
| trainer resume crash | 載入含 `"optimizer"` key 的 checkpoint | 用 `extract_weights.py` 抽純權重再 warm-start |
| `F.binary_cross_entropy` AMP 不允 | AMP autocast + sigmoid 輸出 | 已改手寫 BCE（loss.py） |
| loss 分支 `**None` crash | config 設 `null` 但原 code 無 guard | 已加 `self.X is not None` 守衛（loss.py） |
| TartanAir batch 無 `loss_motion` | 靜態集故意不提供 `motion_mask` | 預期行為；PO batch 仍監督 motion |

---

## 7. 訓練階段總覽（快速參考）

```
VGGT-1B.pt ──warm-start──> S0 (凍 aggregator+原三頭, 只訓 motion/flow, PO)
                              │ extract_weights.py
                              ▼
                      dyn_vggt_s0.pt ──warm-start──> S1 (凍 aggregator, 5頭+7loss, PO+TA)
                                                       │ extract_weights.py
                                                       ▼
                                               dyn_vggt_s1.pt ──warm-start──> S2 (temporal+global+5頭, PO+TA+WA)
                                                                                │
                                                                                ▼
                                                                         eval / S3? / P4 / P5
```

每個箭頭 = `extract_weights.py` 抽純權重 + 新 config warm-start（全新 optimizer）。
