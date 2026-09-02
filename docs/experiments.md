# 實驗登記表

> **建立 2026-08-31。每個跑過的 run 一列：身分、下場、以及失敗的話病因是什麼。**
> 這裡不放完整數字表（那在 [results/natural.md](results/natural.md) / [results/medical.md](results/medical.md)），
> 也不下跨 run 的結論（那在 [status.md](status.md)）。這份回答的是
> **「這個 ckpt 是誰、從哪來、能不能用」**。
>
> checkpoint 的舊名對照與已刪除清單見 [archive/checkpoints.md](checkpoints.md)。

## 讀這份表之前

- **`best.pt` 在自然場景線不可信**。那些 run `pose_eval.enabled=false`，`best.pt` 由 noisy
  windowed val ATE 選，與 full-seq 不對齊。醫學線有 `pose_eval`，`best_ate.pt` 才有意義。
- **ATE 欄分兩種，不可混排**：`val ATE` 是訓練中 channel B 每 epoch 算的（選 ckpt 用）；
  `test` 是 benchmark 數字。單位在 SCARED / C3VD 是 **mm**。
- **單次比較的 2σ 雜訊門檻是 16.9%**（自然場景線）。低於此的 Δ 不要當結果讀。
- **跨幀指標一律 `chunk_size=0`**（整段一次 forward）。分塊拼接會讓接縫主宰 ATE。

---

## 1. 自然場景線（Sintel / PointOdyssey）

warm-start 鏈：`VGGT-1B` → `inst_gate_init` → `inst_g` → `inst_gts`。
全部 `enable_gate=True + enable_temporal=True` 建模（run 1 的 temporal 是 γ=0 身分且凍住），
所以後續 run 能直接載前一個而不會 missing key。

| run | config | warm start | 排程 | 代表 ckpt | Sintel ATE | 下場 |
|---|---|---|---|---|---|---|
| `inst_gate_init` | — | `VGGT-1B` | S0，只訓 gate_predictor | `checkpoints/inst_gate_init.pt` | — | ✅ 用完（當 `inst_g` 的起點） |
| **`inst_g`（run 1）** | `inst_g` | `inst_gate_init` | 50 ep × 3000 步，`img_nums [4,16]` | **`epoch_15`** | **0.1533** | ✅ 完成。gate + camera 的 base |
| **`inst_gts`（run 2）** | `inst_gts` | `inst_g` best | 同上 | **`epoch_30`** | **0.1343** | ✅ 完成。**目前自然場景最佳**，但因子未拆 |
| `inst_g_hard` | `inst_g_hard` | `inst_gate_init` | 只跑 3 ep | `epoch_2` | 0.1641 | ⚪ null result（見 §6.1） |
| `inst_gts_photo`（run 3） | `inst_g_photo` | `inst_g` | — | `epoch_10` | 0.1583 | ❌ 負結果（見 §6.2） |
| `inst_gts_egoflow` 系（4 個 run） | `inst_gts_egoflow*` | `inst_gts` | 20 ep | — | 0.1325–0.1346 | ❌ 陰性收束，見 [ego_flow.md](topics/ego_flow.md) |
| `inst_gtsp_buginit`（run 4 舊版） | — | **buggy-init** | 48 ep | — | 0.1743 | ❌ lineage 汙染，已棄（見 §6.3） |
| `inst_g_buggy` | — | — | — | — | — | 🔒 刻意保留作 bug 世代對照 |

**`inst_gts` 的逐 ckpt 掃描**（full-seq，2026-08-03）：

| ckpt | ep10 | ep20 | **ep30** | ep35 | ep40 | ep45 | ep50(=`last`) |
|---|---|---|---|---|---|---|---|
| ATE | 0.1876 | 0.1796 | **0.1343** | 0.1370 | 0.1340 | 0.1356 | 0.1360 |
| ATE(12) | — | — | **0.0500** | — | 0.0597 | — | 0.0588 |

ep30 後平台化在 0.134–0.136。ep40 的 ATE 看似平手但 ATE(12) 差 19%——它的持平是靠
`cave_2`/`temple_3` 兩個離群序列換來的。**取 `epoch_30.pt`。**

⚠️ `inst_gts` 的 windowed `best.pt`（ep17）full-seq ATE = 0.1651，比 ep30 差 23%。**不要用。**

### 1.1 gate 本身的評估（不是 run，是對 `inst_gts` ep30 的診斷）

| 產出 | 內容 |
|---|---|
| `outputs/gate_quality/inst_gts/quality_f50.json` | macro AUC 0.844 / F1 0.177；micro AUC 0.823 / F1 0.228；calibration 顯示欠自信 |
| `outputs/gate_bias_ablation/inst_gts/baseline/` | off / predicted / oracle / hard@τ / top-k 的 ATE（見 [status.md](status.md) A.2） |
| `outputs/gate_bias_ablation/inst_gts/leaky1.0/` | `gate_leaky=1.0` 變體 |
| `outputs/gate_bias_ablation/inst_gts/zero_ref/` | `gate_bias_zero_ref=True` 變體 |

全部同一支入口：`diag/gate_eval.py`。

---

## 2. 醫學線 — SCARED

全部從 `checkpoints/inst_g.pt` 暖啟動（`vanilla` 除外），`max_epochs: 10`，
train = keyframe1/2、val = keyframe3、test = keyframe4。`val ATE` = channel B full-seq SCARED（mm）。

| arm | config | 與前一個的差別 | val ATE（best / last） | 下場 |
|---|---|---|---|---|
| **baseline** | `scared_cam_vanilla` | warm start 改 `VGGT-1B`、`enable_gate/temporal=False` | 0.9705 / 1.0069 | ✅ 完成。**這是對照組**，不是 pretrained |
| `b16` | `scared_cam_b16` | warm start `inst_g`，但 gate 凍結 + temporal γ=0 ≈ 原架構 | 0.9814 / 1.0099 | ✅ 完成。用來拆「lineage vs 架構」 |
| `b2` | `scared_cam_b2` | 只差 batch 組成 | — | ⚪ 未跑完（log 無 pose eval） |
| `b16_gg` | `scared_cam_b16_gg` | gate 解凍通電 | 0.9464 / 0.9464 | ⚠️ **ep8 被 SIGTERM**，`best_ate.pt` 是 8-epoch 產物 |
| **`smooth_temporal`（arm 2）** | `..._smooth_temporal` | temporal 解凍 + `camera_smooth`；`img_nums [4,16]→[4,12]` | **0.9108** / 0.9306 | ✅ 完成。**test pose / depth 單張最佳** |
| **`wide`（arm 3）** | `..._wide` | 唯一 delta：`nearby_expand_range: 120` | **0.8150** / 0.8240 | ✅ 完成。**val ATE 與 depth 多張最佳** |
| `depth` | `..._depth` | `loss.depth` 開 + `depth_head` 解凍 | 0.8566 / 0.9246 | ❌ **負結果**（見 §6.4） |
| `depth_lowalpha` | `..._depth_lowalpha` | 同上但 `alpha: 0.2 → 0.02` | — | ⬜ **尚未跑** |

**逐 epoch val ATE**（channel B，full-seq SCARED）：

| arm | ep0 | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `vanilla` | 1.5590 | 1.1383 | 1.0454 | **0.9705** | 1.0812 | 1.0598 | 0.9977 | 1.0349 | 1.0228 | 1.0069 |
| `b16` | 1.4538 | 1.4608 | 1.2478 | 1.0879 | 1.2032 | 1.0571 | 1.0117 | **0.9814** | 1.0095 | 1.0099 |
| `b16_gg` | 1.1753 | 1.3068 | 1.2980 | 1.0819 | 1.0703 | 0.9617 | 0.9739 | **0.9464** | — | — |
| `smooth_temporal` | 1.2871 | 1.1897 | 0.9315 | 0.9800 | 1.1799 | 1.0848 | 0.9315 | **0.9108** | 0.9353 | 0.9306 |
| `wide` | 1.1159 | 1.1767 | 1.0472 | 0.9973 | 0.9355 | 0.8716 | 0.8426 | **0.8150** | 0.8234 | 0.8240 |
| `depth` | 1.3271 | 1.1123 | 1.0902 | 0.9514 | 1.1050 | 0.9805 | **0.8566** | 0.9064 | 0.9352 | 0.9246 |

> ⚠️ **選 ckpt 的指標與 test 不一致**：`wide` 的 val ATE 最好（0.8150）但 test pose 較差
> （0.0705 vs `smooth_temporal` 的 0.0597）。這件事沒解決，見 [status.md](status.md) B.2。

test 數字（depth / pose，含輸入張數與協定）一律看 [results/medical.md](results/medical.md) §1，
對外的 published 方法對照看 [results/sota.md](results/sota.md)。

---

## 3. 醫學線 — C3VD

| arm | config | warm start | val ATE（逐 epoch） | 下場 |
|---|---|---|---|---|
| `c3vd_cam_vanilla` | `c3vd_cam_vanilla` | `VGGT-1B` | **0.4021** / 0.4118 / 0.4401 | ✅ 對照組 |
| `c3vd_cam_gts` | `c3vd_cam_gts` | `inst_g` | **0.3589** / 0.4296 / 0.4194 / 0.4218 / 0.4715 / 0.4230 | ✅ 完成（ep6 後暫停） |

零樣本 `VGGT-1B` 同設定 0.4991。

⚠️ **兩個 run 的 `best_ate.pt` 都是 epoch 0**，之後再沒更新過——「我們的 C3VD 版本」
**就是第一個 epoch 的權重**。test 也同向（ep0 0.4329 < ep5 0.4642）。
這是一個需要解釋的現象，目前沒查。

⚠️ `c3vd_cam_gts` 的 `best_loss.pt` 在 2026-08-25 08:29 被 ep6 覆寫，
[results/medical.md](results/medical.md) §2.1 那一列**重跑不可復現**。

---

## 4. 照明穩健性線

| arm | config | warm start | val ATE（逐 epoch） | 下場 |
|---|---|---|---|---|
| `scared_point_inf` | `scared_point_inf` | `scared_arm2_pointhead.pt` | 4.31 / **2.03** / 3.57 / 3.91 / 6.22 / 5.53 / 4.92 / 3.93 / 3.32 / 3.24 | ❌ 失敗（見 §6.5） |

warm start 是 `tools/merge_point_head.py` 把 arm 2 的 trunk 與 `VGGT-1B` 的 pretrained
`point_head` 拼成的**裸 state_dict**（裸的很重要——帶 optimizer state 會讓這個 run 從
10-epoch 排程的第 11 個 epoch 起跑然後直接結束）。

**尚未跑的必要對照**：
- `loss.influence.weight: 0` 的 augmentation-only arm。沒有它就不能主張任何 L_inf 的增益。
- 修正後（quantile filter + weight 10）的重跑。

---

## 5. 已棄與汙染 lineage（不要引用）

| 對象 | 為什麼不能用 |
|---|---|
| `inst_gtsp_buginit` | 從 buggy-init 暖啟動而非 run 1 |
| `outputs/train/*/`、`archive/logs/v3_bug/` | gate 前向 bug 修正（commit `e064087`）之前的世代 |
| `inst_gts` 的 `best.pt`（ep17） | windowed 選擇，full-seq 差 23% |
| `test_0f/results.json` 裡 `b16` 的 snippet ATE 0.6148 | 守衛加入前的失效值 |
| `c3vd_cam_gts` 的 `best_loss.pt` 那一列 | 檔案已被 ep6 覆寫 |

---

## 6. 病因檔案

失敗的 run 值錢的地方在這裡。每一條都有可查的量測依據。

### 6.1 `inst_g_hard` — BCE boundary floor 不是 gate 欠自信的病因

**假說**：`m*_patch` 是 average-pool 的軟標籤，boundary patch 有不可約的 BCE floor，
這個 floor 讓 gate 學不到自信。
**做法**：hard label（≥0.7→1、≤0.3→0）+ 中間 band 不算 loss，砍掉 floor。
**結果**：`signed_err` 只降 6%，pose 沒變化。**假說證偽。** 欠自信另有原因。

### 6.2 `_photo`（static-photo loss）— 越訓越差

full-seq 最佳 `ep10` = 0.1583，**比它自己的 warm-start 起點 0.1533 還差**，
ep10 → ep20 單調惡化。疊上 smooth_temporal 的版本更差（0.1743）。
**這條線已收掉**，機制描述保留在 [method.md](method.md) §6。

### 6.3 `inst_gtsp_buginit` — lineage 汙染

從 buggy-init 而非 run 1 暖啟動，ATE 0.1743 比 base 還差。
**這個數字不能當「全開組合沒用」的證據**——乾淨 lineage 的全開組合至今未跑。

### 6.4 `..._depth` — confidence loss 的 `−α·log(c)` 項劫持了目標

test depth 反而比**凍結**的頭差（AbsRel 0.0434 → 0.0468，SqRel 與 RMSE 甚至低於零樣本頭），
pose 退 18%（snippet ATE 0.0597 → 0.0707）。epoch 5 的 loss 分解說明了原因：

| 項 | 值 | 權重 | 貢獻 |
|---|---|---|---|
| camera | 0.0063 | ×5.0 | +0.032 |
| camera_smooth | 0.0008 | ×3.0 | +0.002 |
| **conf_depth** | **−0.3190** | ×1.0 | **−0.319** ← camera 項的九倍，而且是負的 |
| reg_depth | 0.0165 | | +0.017 |
| grad_depth | 0.0122 | | +0.012 |
| **objective** | | | **−0.256** |

`L_conf = γ·l_reg·c − α·log(c)`，最佳解 `c* = α/(γ·l_reg)` 會在 `l_reg → 0` 時把
`L_conf` 推向 −∞。這對從頭訓的 head 無害（`l_reg` 大），但我們的 head 是**訓練得很好的
pretrained head**，起手 `l_reg` 只有 0.0165 → `−α·log(c)` 從第 0 步就主宰 → 最便宜的下降
方向是「灌大信心」而不是「深度預測得更準」。depth_head 與 camera_head 共用 aggregator，
所以這個梯度重塑了 trunk，把 pose 一起帶壞。

**修正**：`alpha: 0.2 → 0.02`（`scared_cam_b16_gg_smooth_temporal_depth_lowalpha`）。
該 config 自帶可證偽的預測：objective 必須轉正、`conf_depth` 落在 −0.02~−0.05、
`reg_depth` 過 ep1 仍在降。**尚未跑。**

### 6.5 `scared_point_inf` — L_inf 被幾十個失控像素綁架

`loss_inf_point` 前 5 個 epoch 平在 ~0.0013，之後 0.0351(ep7) → 0.1538(ep9) → 0.636(末)。
weight 30 之下該項變成整個 objective（4.62 of 4.60），run 的 ATE 反向。

`diag/influence_scale_probe.py` 把兩個假說分開量：

| 假說 | 特徵 | 實測 |
|---|---|---|
| (A) 正規化回饋（我寫的 bug） | raw 平、scale 降 → norm 升 | — |
| (B) 真的發散（方法問題） | raw 本身升 | — |

**實測結果**：ep10 時 point map 的 median 0.2684、p99 1.33（**都與暖啟動時相同**），
但 **max 到了 8.9e4**。~2M 像素裡的幾十個扛走了幾乎整個 mean。而**行為正常的那 99%
方向是對的**（median drift 0.00082 → 0.00024）——那才是這個 loss 該讀的訊號。

**根因是兩個 loss 對 outlier 的定義不一致**：`compute_point_loss` 的 `valid_range=0.98`
會丟掉殘差最大的 2%，所以那些失控像素**從來沒被監督過**、在它背後無人看管地長大；
而 L_inf 照單全收它們的原始值。

**修正**（已進 code，未重跑）：
1. L_inf 套用同一個 quantile filter（`valid_range=0.98`）。
2. 除數從 `mean` 改 `median`（teacher 的 mean |X| 是 12.62 而 median 只有 0.2684，
   mean 自己就被 outlier 主宰，會靜默把 loss 放大 ~30 倍）。
3. weight 30 → 10，錨定在 camera 項的量級上。
4. `w_pose: 1.0 → 12.0`（兩個 channel 尺度差一個數量級，等權時 pose channel 是啞的）。

---

## 7. 尚未跑但已定義的 run

| run | 要回答什麼 | 為什麼還沒跑 |
|---|---|---|
| temporal-only / smooth-only | 拆 `inst_gts` 的 −12% 是結構還是 loss | 一直被醫學線的 deadline 排擠 |
| 乾淨 lineage 的全開組合（真正的 run 4） | ①+②+③ 是否正交相加 | ② 已收掉，需重新定義「全開」是什麼 |
| `..._depth_lowalpha` | §6.4 的修正是否有效 | 待跑 |
| L_inf `weight: 0`（augmentation-only） | L_inf 的增益是不是只來自「看過壞幀」 | 待跑 |
| L_inf 修正後重跑 | §6.5 的修正是否有效 | 待跑 |
| run 3'（預測深度的 static-photo） | pose+depth 聯合光度精修 | ② 線已收掉，優先序最低 |
