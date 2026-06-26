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
| **Config** | `config/dyn_vggt_po_s0.yaml` | S0: 凍 backbone，只訓 motion/flow |
| | `config/dyn_vggt_po.yaml` | S1: 解凍 temporal+原三頭，全 7 項 loss |
| | `config/dyn_vggt_po_s0_smoke.yaml` | S0 極短 smoke test（3 batch × 1 epoch） |
| **權重** | `checkpoints/VGGT-1B.pt` | 預訓練 VGGT-1B（原始 warm-start） |
| | `checkpoints/dyn_vggt_s0.pt` | S0 訓練完的純權重（epoch 7，已驗證） |
| **腳本** | `verify_warmstart.py` | 驗證 checkpoint 載入（missing/unexpected keys + γ=0） |
| | `eval_s0.py` | S0 驗證（motion IoU/F1 + flow 改善 + 凍結完好 + 質性圖） |
| | `extract_weights.py` | 從 trainer checkpoint 抽純權重（跨 stage 傳遞用） |
| **訓練 log** | `logs/dyn_vggt_po_s0/run.log` | S0 訓練 log |
| | `logs/dyn_vggt_po_s0/ckpts/` | S0 每 epoch checkpoint（checkpoint_0.pt – checkpoint_7.pt） |
| | `logs/dyn_vggt_po_s0/eval/` | S0 驗證質性圖輸出 |
| **數據** | `/media/cvml-75/ssd2t1/data/point_odyssey` | PointOdyssey（train 264 seq / test 28 seq） |

---

## 2. S0：對齊新頭（已完成 ✅）

### 2.1 訓練

```bash
# 正式訓練（12 epoch，每 epoch 存 checkpoint，~10 min/epoch）
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_po_s0

# 極短 smoke test（確認能跑通，3 batch × 1 epoch）
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_po_s0_smoke

# 背景執行（過夜）
nohup torchrun --nproc_per_node=1 --master_port=29606 launch.py --config dyn_vggt_po_s0 \
  > logs/dyn_vggt_po_s0/run.log 2>&1 &

# 監控
tail -f logs/dyn_vggt_po_s0/run.log | grep -E "Train Epoch|Val Epoch|Saving|Error"
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

---

## 3. S1：運動解耦（下一步）

### 3.1 執行前確認

```bash
# 確認 S0 純權重存在
ls -lh checkpoints/dyn_vggt_s0.pt

# 確認 S1 config 的 resume 指向 S0 純權重（不是 trainer checkpoint！）
grep resume_checkpoint_path config/dyn_vggt_po.yaml
# 應顯示：checkpoints/dyn_vggt_s0.pt（或絕對路徑）

# warm-start 驗證（可選——確認載入正確）
python verify_warmstart.py checkpoints/dyn_vggt_s0.pt
```

### 3.2 S1 config 要點（相對 S0 的差異）

| 項目 | S0 | S1 |
|---|---|---|
| 解凍 | 只 motion/flow 頭 | temporal 塊 + 原三頭 + 兩新頭（凍 DINOv2 patch_embed） |
| 啟用 loss | motion + flow | **全 7 項**（+ camera + depth + point + reproj + tsmooth） |
| `mask_dynamic` | off | **on**（point loss 用 (1−m) 屏蔽動態） |
| resume | `VGGT-1B.pt` | **`dyn_vggt_s0.pt`**（S0 純權重） |

### 3.3 訓練

```bash
# 正式訓練
torchrun --nproc_per_node=1 launch.py --config dyn_vggt_po

# 背景執行
nohup torchrun --nproc_per_node=1 --master_port=29610 launch.py --config dyn_vggt_po \
  > logs/dyn_vggt_po/run.log 2>&1 &

# 監控（S1 多了更多 loss 項）
tail -f logs/dyn_vggt_po/run.log | grep -E "loss_motion|loss_reproj|loss_camera|Error"
```

### 3.4 S1 完成後：抽權重給 S2

```bash
python extract_weights.py \
  --src logs/dyn_vggt_po/ckpts/checkpoint_BEST.pt \
  --dst checkpoints/dyn_vggt_s1.pt
```

---

## 4. S2：全網精修（待做）

S2 config 尚未建立，要點：
- `resume_checkpoint_path: checkpoints/dyn_vggt_s1.pt`
- 全解凍（`frozen_module_names` 清空）、小 lr（1e-5）
- 全 7 項 loss、最終配比
- 混入靜態數據（~30%）防遺忘 → **需先補 TartanAir loader**
- 防遺忘 gate：S2 結束後在純靜態 benchmark 驗不退化

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
# 最近幾個 batch 的 loss
tail -20 logs/<exp_name>/run.log | grep "Train Epoch"

# 每 epoch 的 val loss 趨勢
grep "Val Epoch.*loss_motion" logs/<exp_name>/run.log | tail -12

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
kill $(pgrep -f "launch.py --config dyn_vggt_po")

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
| `accum_steps=2` 空 chunk crash | `batch_size=1`（大 `img_num` 時）+ `accum_steps>1` | 設 `accum_steps: 1` |
| OOM @518² | 4 個 DPT head + `max_img_per_gpu≥16` | 降 `max_img_per_gpu`（S0 實測 10 → 22GB） |
| trainer resume crash | 載入含 `"optimizer"` key 的 checkpoint | 用 `extract_weights.py` 抽純權重再 warm-start |
| `F.binary_cross_entropy` AMP 不允 | AMP autocast + sigmoid 輸出 | 已改手寫 BCE（loss.py） |
| loss 分支 `**None` crash | config 設 `null` 但原 code 無 guard | 已加 `self.X is not None` 守衛（loss.py） |

---

## 7. 訓練階段總覽（快速參考）

```
VGGT-1B.pt ──warm-start──> S0 (凍backbone, 只訓motion/flow)
                              │ extract_weights.py
                              ▼
                      dyn_vggt_s0.pt ──warm-start──> S1 (解凍temporal+原三頭, 7項loss)
                                                       │ extract_weights.py
                                                       ▼
                                               dyn_vggt_s1.pt ──warm-start──> S2 (全網, 混靜態)
                                                                                │
                                                                                ▼
                                                                         eval / P4 / P5
```

每個箭頭 = `extract_weights.py` 抽純權重 + 新 config warm-start（全新 optimizer）。
