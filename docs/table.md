# Dyn-VGGT v3 — 評估數據表

> 建立 2026-07-15。本檔只放**已經跑出來的數字**與其實驗設定，不放結論推論。
> 每張表開頭的「設定」區塊是重跑該表所需的完整資訊。

## 世代地圖（讀任何數字前先確認）

gate 前向 bug 修正 = commit `e064087`（2026-07-09 12:17）。**修正是 code 層，任何在此之後啟動的 run 自動生效。**

| 位置 | 世代 | 內容 |
|---|---|---|
| `training/logs/dyn_vggt_v3_s1_inst/` | ✅ **clean** | run 1，24 epoch。`gate_sweep`/`gate_sweep_scales`/`gate_quality`/`gate_bias_ablation`/`pose_eval` |
| `training/logs/dyn_vggt_v3_s1_inst_hard/` | ✅ clean | 只有 3 epoch |
| `training/logs/train/*/` | ✅ clean | 07-13 評測 harness：`VGGT-1B` / `dyn_vggt_v3_s0_inst` / `dyn_vggt_v3_s1_inst` / `s1_buggy` |
| `training/logs/dyn_vggt_v3_s1_inst_buggy/` | ❌ buggy | 專門保留作對照 |
| `outputs/train/*/` | ❌ **buggy** | photo / smooth_temporal / oracle_cam 的 Sintel 數字全在這 |
| `archive/logs/v3_bug/` | ❌ buggy | |
| `training/logs/dyn_vggt_v3_s1_inst_photo_smooth_temporal/` | ❌ buggy-init | lineage 不乾淨，已棄 |

**checkpoint 更名**：`dyn_vggt_v3_s1.pt` == `archive/checkpoints/dyn_vggt_v3_s1_inst.pt`（歸檔時更名）。
同一模式：`dyn_vggt_v3_oracle.pt` ← `oracle_camera_only` run。

**2026-07-15 清理**：`archive/logs/v3_bug/*/ckpts/` 全數刪除（55 GB，archive 100 GB → 45 GB）。
刪除前已逐一驗證 `archive/checkpoints/` 的 model-only 匯出檔涵蓋對應 `best.pt` 的**全部 model key 且張量值相同**：

| run | best.pt | 匯出檔 | 狀態 |
|---|---|---|---|
| `s1_inst_photo` | ep19 | `dyn_vggt_v3_s1_inst_photo.pt` | ✅ 1491 keys 全等 |
| `oracle_camera_only` | ep19 | `dyn_vggt_v3_oracle.pt` | ✅ 1423 keys 全等 |
| `s1_inst_smooth_temporal` | **ep2** | `dyn_vggt_v3_s1_smooth_temporal.pt` | ✅ 1491 keys 全等 |
| `s1_inst_smooth_temporal_v2` | ep0 | 無 | 已棄（從未評測） |

**已知損失**：`s1_inst_smooth_temporal/ckpts/last.pt`（**epoch 4**）未匯出，已隨刪除消失。
該 run 的 best 在 ep2 → ep3/ep4 更差，且屬 buggy 世代已棄 lineage，判定無分析價值。
其餘 run 的 `last.pt` 與 `best.pt` 為同一 epoch，無額外損失。
`tensorboard/`、`log.txt`、`config/`、eval json 全部保留。

## Δ% 的基準（全檔通則）

**除表 2.3 外，所有 Δ% 一律以「同一列自己的 `off`」為分母**：

```
Δ%  =  (該模式 ATE − 同列 off ATE) / 同列 off ATE × 100
```

同 ckpt、同 `max_frames`、同序列 —— 只有 gate 模式不同。**負值 = ATE 下降 = 變好。**

**表 2.3 是唯一例外**：該表比較的是**跨版本的 `off` 本身**，分母是被比較的那個版本（`buggy_off` 或 `base_off`）。詳見該表設定區塊。

每張表的設定區塊都會重述其 Δ% 基準，以免單獨引用時誤讀。

---

## 表 1 — Sintel 全序列全指標 × 版本

### 設定

| 項目 | 值 |
|---|---|
| script | `training/benchmark/eval_sintel.py` |
| 序列 | 14 seq（Sintel training split 全部） |
| `chunk_size` | **0 = 完整序列**（非分塊） |
| 實際幀數 | mean **45.9** frames/seq |
| `max_depth` | 80.0 |
| pose 指標 | `ate`, `rpe_trans`, `rpe_rot` |
| depth 指標 | `abs_rel`, `sq_rel`, `rmse`, `log_rmse`, `delta_1/2/3` |
| ATE(12) | 14 seq 去掉 `cave_2`、`temple_3` 兩個離群值後的平均 |
| **Δ% 基準** | **本表無 Δ%，全部為絕對值**。要比較請自行以 `VGGT-1B base` 列為分母 |
| 資料來源 | `outputs/train/*/eval_sintel/`、`training/logs/*/pose_eval/epoch_*/`、`archive/logs/v1_train/` |

### 表

| 版本 | 世代 | ATE | ATE(12) | RPE-t | RPE-r | AbsRel | sqRel | RMSE | logRMSE | δ1 | δ2 | δ3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **VGGT-1B base** | base | 0.1714 | 0.0618 | 0.0617 | 0.4706 | 0.2747 | 2.348 | 5.827 | 0.4217 | 0.683 | 0.810 | 0.881 |
| v1 S0 | v1 | 0.1714 | 0.0618 | 0.0617 | 0.4706 | 0.2747 | 2.348 | 5.827 | 0.4217 | 0.683 | 0.810 | 0.881 |
| v1 S1 | v1 | 0.1710 | 0.0651 | 0.0691 | 0.4792 | 0.2552 | 1.924 | 5.384 | 0.3827 | 0.686 | 0.815 | 0.881 |
| v1 S1b | v1 | 0.1692 | 0.0680 | 0.0770 | 0.5502 | 0.2790 | 2.999 | 5.733 | 0.4006 | 0.692 | 0.821 | 0.880 |
| v2 S1_v2 | v2 | 0.1732 | 0.0688 | 0.0685 | 0.4857 | 0.2558 | 1.943 | 5.472 | 0.3847 | 0.689 | 0.817 | 0.881 |
| v1 S2a | v1 | 0.2035 | 0.1113 | 0.0896 | 0.8909 | 0.2840 | 2.431 | 4.974 | 0.3620 | 0.704 | 0.837 | 0.905 |
| v3bug s1_inst | bug | 0.1790 | 0.0706 | 0.0828 | 0.5200 | 0.2892 | 2.646 | 5.586 | 0.4198 | 0.677 | 0.822 | 0.887 |
| v3bug photo | bug | 0.1620 | 0.0666 | 0.0813 | 0.5474 | 0.2732 | **6.210** | 6.489 | 0.4014 | 0.684 | 0.808 | 0.883 |
| v3bug smooth_temp ⚠️ | bug | 0.1568 | 0.0535 | 0.0720 | 0.4634 | 0.2458 | 2.082 | 5.133 | 0.3812 | 0.680 | 0.825 | 0.898 |
| v3bug oracle_cam | bug | 0.1757 | 0.0685 | 0.0649 | 0.4769 | **0.5631** | 4.535 | 7.694 | 0.6945 | **0.430** | 0.630 | 0.734 |
| **v3CLEAN run1 ep15** ★ | clean | **0.1533** | **0.0517** | 0.0671 | 0.4923 | **0.2136** | **1.776** | **5.109** | **0.3522** | **0.714** | **0.851** | **0.910** |
| v3CLEAN run1 ep24 | clean | 0.1713 | 0.0590 | 0.0665 | 0.4733 | 0.2428 | 3.116 | 5.709 | 0.3732 | 0.699 | 0.840 | 0.898 |
| v3CLEAN hard ep2 | clean | 0.1641 | 0.0557 | 0.0710 | **0.4447** | 0.2607 | 2.095 | 5.340 | 0.3947 | 0.684 | 0.826 | 0.894 |
| **MonST3R（目標）** | ref | **0.1080** | — | **0.0420** | 0.7320 | 0.3450 | — | — | — | 0.562 | — | — |

### 附註

- ⚠️ **`v3bug smooth_temp` 遠未收斂 —— 該列是 epoch 2 的數字**。已由 checkpoint metadata（`prev_epoch`）確認：
  - 被評測的 `dyn_vggt_v3_s1_smooth_temporal.pt` = 該 run 的 `best.pt` = **epoch 2**（train steps 6003）
  - `last.pt` = epoch 4；log 在 epoch 5 進行中時中斷（07-06 18:54 → 22:54）
  - config 設定為 **20 epoch**

  **所以 0.1568 是「20 epoch 計畫中第 2 個 epoch」的快照**，不能當作 smooth+temporal 的收斂值。同組 `s1_inst_smooth`（Ep 0 step 36）與 `_v2`（**Ep 0**，train steps 2001）死得更早。**smooth 這條線至今沒有任何跑完的 run，也沒有任何 run 的 checkpoint 超過 epoch 4。**
- **v1 S0 與 VGGT-1B 逐位元相同** —— `enable_*` flag 全關時與 pretrained 權重相容的回歸證據。
- **`run1 ep15` 全面最好**：ATE / ATE(12) / AbsRel / sqRel / RMSE / logRMSE / δ1 / δ2 / δ3 **九項全贏 base**，只輸 RPE-t、RPE-r。
- `v3bug photo` 的 sqRel **6.210** 是全場最差（base 2.348）而 AbsRel 0.2732 看似正常 → 少數 pixel 巨大誤差。
- `oracle_cam` 的 depth 全面崩潰（AbsRel 0.563、δ1 0.430），其 pose 數字須在此前提下讀。該 run config 的 loss block 含 `point`/`track` 且 `enable_gate=False`。
- **MonST3R 只在 ATE 與 RPE-t 領先**；RPE-r（0.732 vs 0.492）、AbsRel（0.345 vs 0.214）、δ1（0.562 vs 0.714）我方大幅領先。**缺口是純平移的。**
- 所有版本 depth 的 `num_frames` 均為 45.9 → 協定一致，可直接比較。

---

## 表 2.1 — 全序列 off / on / oracle

### 設定

| 項目 | 值 |
|---|---|
| script | `training/diag/gate_bias_ablation.py --dataset sintel` |
| `max_frames` | **50**（≈ 完整序列，Sintel mean 45.9） |
| `chunk_size` | 0 = 完整序列 |
| `k` | 30.0 — oracle/off 的 logit 幅度，bias ≈ `−softplus(k)` |
| `motion_thr` | 2.0 px — GT flow-residual oracle mask 閾值 |
| 序列 | **14 seq** |
| **Δ% 基準** | **同一列自己的 `off`**（同 ckpt、同 f50、同序列）。例：VGGT-1B oracle `+2.3%` = (0.1752 − 0.1713) / 0.1713。**不同列之間的 Δ% 分母不同，不可相減** |
| 資料來源 | `training/logs/train/VGGT-1B/`、`training/logs/dyn_vggt_v3_s1_inst/gate_sweep/f50/`、`training/logs/dyn_vggt_v3_s1_inst_buggy/gate_sweep/f50/` |

**模式定義**
- `off` — gate bias 關閉（`enable_gate` 前向不施加 bias）
- `on` — = `predicted`，用訓練出來的 gate predictor 輸出的 `σ(g)`
- `oracle` — `gate_logits_override`，直接由 GT motion mask 造 logit（eval-only 上界）

### 表

| 版本 | 世代 | n | off | **on (predicted)** | Δ% | **oracle** | Δ% |
|---|---|---|---|---|---|---|---|
| VGGT-1B (base) | base | 14 | 0.1713 | 0.1715 | **+0.1** | 0.1752 | **+2.3** |
| s1_inst | clean | 14 | **0.1537** | 0.1533 | **−0.3** | 0.1548 | **+0.7** |
| s1_inst | buggy | 14 | 0.1793 | 0.1789 | −0.2 | 0.1907 | **+6.4** |

**13-seq 口徑**（去 `alley_2`，與表 2.2 對齊）：

| 版本 | n | off | on | Δ% | oracle | Δ% |
|---|---|---|---|---|---|---|
| VGGT-1B (base) | 13 | 0.1838 | 0.1840 | +0.1 | 0.1880 | +2.3 |
| s1_inst (clean) | 13 | **0.1642** | 0.1637 | −0.3 | 0.1653 | +0.7 |
| s1_inst (buggy) | 13 | 0.1922 | 0.1918 | −0.2 | 0.2044 | +6.4 |

### 附註

- **去掉 `alley_2` 只改變絕對值，Δ% 一模一樣**（+0.1 / +2.3 / −0.3 / +0.7 / −0.2 / +6.4 完全不動）。原因：`alley_2` 三個模式的 ATE 幾乎相同（clean: off 0.0182 / oracle 0.0182 / predicted 0.0181），它只是稀釋分母。**所以本表的相對結論對序列組成不敏感。**
- **VGGT-1B 的 oracle = +2.3%（反向）** 是 sanity check：未經 gate 訓練的 camera path 吃到 gate bias 是純傷害，符合預期。
- **在完整序列上，三個版本的 oracle 都沒有頭空**（+2.3 / +0.7 / +6.4）。clean 的 +0.7% 是三者中最接近 0 的，代表 bug fix 把「oracle 主動傷害 pose」修成了「oracle 無害但也無益」。
- `on` 在三個版本都是 −0.3 ~ +0.1%，**完整序列上 gate 等同於關閉**。

---

## 表 2.2 — clean 版本在不同長度下的 off / on / oracle

### 設定

| 項目 | 值 |
|---|---|
| script | `training/diag/gate_bias_ablation.py --dataset sintel` |
| ckpt | `training/logs/dyn_vggt_v3_s1_inst/ckpts/best.pt`（= run1 ep15） |
| `chunk_size` | 0 = 完整序列（`max_frames` 只截斷序列長度，不分塊） |
| `k` | 30.0 |
| `motion_thr` | 2.0 px |
| 序列 | **f4~f32 均為 13 seq（缺 `alley_2`）；f50 為 14 seq** |
| **Δ% 基準** | **同一列自己的 `off`**（同 ckpt、**同 `max_frames`**、同序列）。例：f12 oracle `−15.0%` = (0.0351 − 0.0413) / 0.0413。**不同 frames 之間的 Δ% 分母不同（off 隨長度從 0.0138 漲到 0.1537），只能比方向與有無，不可相減** |
| 資料來源 | `gate_sweep/f{4,8,16,32,50}/`（scales=[]）、`gate_bias_ablation/`（f12, scales=[3,10]） |

### 表

| frames | n | off | **on** | Δ% | **oracle** | Δ% | ×3 Δ% | ×10 Δ% |
|---|---|---|---|---|---|---|---|---|
| 4 | 13 | 0.0138 | 0.0139 | +0.7 | 0.0123 | **−10.8** | +1.5 | −0.0 |
| 8 | 13 | 0.0268 | 0.0265 | −1.0 | 0.0246 | **−8.0** | +0.3 | +1.4 |
| **12** | 13 | 0.0413 | 0.0406 | −1.5 | 0.0351 | **−15.0** | **−5.4** | **−7.1** |
| 16 | 13 | 0.0438 | 0.0437 | −0.2 | 0.0421 | **−3.8** | −1.1 | −2.9 |
| 32 | 13 | 0.1204 | 0.1191 | −1.0 | 0.1072 | **−11.0** | −1.0 | −1.1 |
| 50 | 14 | 0.1537 | 0.1533 | −0.3 | 0.1548 | **+0.7** | — | — |

（f50 的 13-seq 口徑見表 2.1；Δ% 不變。）

### 附註

- **`on` 在所有長度都是 −1.5 ~ +0.7%**，沒有任何例外。**gate 訓練出來的預測值，在任何序列長度都等同於關閉。**
- **oracle 頭空存在但非單調**：f4 −10.8 → f8 −8.0 → **f12 −15.0** → **f16 −3.8** → f32 −11.0 → f50 +0.7。
- ⚠️ **f12 與 f16 之間 11 個百分點的跳動已排除下列解釋**：兩者**同一支 script**（`gate_bias_ablation.py`）、**同 13 seq**、**同 `k=30`**、**同 `motion_thr=2.0`**、**同 `chunk_size=0`**。唯一差異是 f12 帶 `scales=[3,10]`（只影響額外的 ×3/×10 欄，不影響 off/oracle/predicted 的計算）。**此跳動目前無法解釋，是本表最大的未解問題。**
- **f50 是唯一 14 seq 的格子**，但表 2.1 已證實序列組成不影響 Δ%，故「f50 歸零」成立。
- **×3/×10 只在 f12/f16 有效**（f12 ×10 −7.1%，吃到 oracle −15.0% 的約一半）。f4/f8 放大反而變差，f32 完全無反應。**放大策略只在窄的長度窗成立。**

---

## 表 2.3 — clean vs buggy vs base（修復後架構的訓練收益）

### 設定

| 項目 | 值 |
|---|---|
| script | `training/diag/gate_bias_ablation.py --dataset sintel`，取 `off` 模式 |
| 比較對象 | `off` 模式 = 不施加 gate bias 的裸 pose，用以隔離「架構修復 + 訓練」的收益，排除 gate 本身 |
| clean ckpt | `dyn_vggt_v3_s1_inst/ckpts/best.pt`（post-`e064087` 重訓，warm from `s0_inst.pt`） |
| buggy ckpt | `dyn_vggt_v3_s1_inst_buggy`（pre-fix 訓練） |
| base ckpt | `checkpoints/VGGT-1B.pt` |
| 序列 | f4~f32 為 13 seq；f50 為 14 seq |
| **Δ% 基準** | ⚠️ **全檔唯一例外 —— 此表為跨版本比較，分母是被比較的那個版本**：<br>`clean vs buggy` = (clean_off − buggy_off) / **buggy_off**<br>`clean vs base` = (clean_off − base_off) / **base_off**<br>例：f50 `−14.3%` = (0.1537 − 0.1793) / 0.1793 |
| ⚠️ 限制 | **base 只有 f50 一個資料點**，f4~f32 無 base 對照 |

### 表

| frames | base off | buggy off | **clean off** | clean vs buggy | clean vs base |
|---|---|---|---|---|---|
| 4 | — | 0.0194 | **0.0138** | **−28.9%** | — |
| 8 | — | 0.0394 | **0.0268** | **−32.0%** | — |
| 16 | — | 0.0620 | **0.0438** | **−29.4%** | — |
| 32 | — | 0.1532 | **0.1204** | **−21.4%** | — |
| 50 | 0.1713 | 0.1793 | **0.1537** | **−14.3%** | **−10.3%** |

oracle 模式的世代對比（同格子）：

> ⚠️ **此小表回到通則：每個 Δ% 是該版本 oracle 相對「該版本自己的 off」**（buggy 除以 buggy_off，clean 除以 clean_off）。
> **兩欄分母不同，不可相減。** clean 的 off 本來就比 buggy 小 21~32%，所以同樣的絕對改善在 clean 欄會顯示成更大的百分比。
> 本小表只能讀**頭空的方向與有無**，不能讀「頭空增加了幾個百分點」。

| frames | buggy oracle Δ%<br>(÷ buggy_off) | clean oracle Δ%<br>(÷ clean_off) | 變化 |
|---|---|---|---|
| 4 | −6.0 | **−10.8** | 頭空放大 |
| 8 | −12.2 | −8.0 | 頭空縮小 |
| 16 | **+1.6** | **−3.8** | **反向 → 正向** |
| 32 | −0.5 | **−11.0** | **無效 → 有效** |
| 50 | **+6.4** | **+0.7** | **傷害 → 無害** |

### 附註

- **「bug fix 買到 base pose −20~30%」證實**：clean 相對 buggy 在 f4~f32 是 −21% ~ −32%，f50 收斂到 −14.3%。**短序列收益最大。**
- **唯一有 base 對照的格子是 f50**：clean 0.1537 vs base 0.1713 = **−10.3%**。這是 v3 訓練相對 pretrained VGGT 的真實淨收益（在 gate 關閉的前提下）。
- ⚠️ **f4~f32 缺 base 對照是本表最大缺口**。沒有它，就無法把「bug fix 的收益」與「短序列本身較容易」這兩個效應分離 —— buggy 在短序列爛，可能是 bug 在短序列傷害更大，也可能只是 buggy run 整體較差而短序列放大了差距。**補這五格 base 是成本最低、資訊量最大的一次 eval。**
- **oracle 的世代對比是機制修復最直接的證據**：buggy 在 f16 (+1.6)、f50 (+6.4) 上 oracle **主動傷害** pose；clean 全部轉為無害或有益（f32 從 −0.5 變 −11.0）。**這證明修的是 forward 的正確性，不是調參。**

---

## 表 3 — Gate 品質 → Pose 影響（clean, per-seq）

### 設定

| 項目 | 值 |
|---|---|
| 品質來源 | `training/logs/dyn_vggt_v3_s1_inst/gate_quality/quality_f16.json` |
| pose 來源 | `training/logs/dyn_vggt_v3_s1_inst/gate_sweep_scales/f16/results.json` |
| ckpt | `dyn_vggt_v3_s1_inst/ckpts/best.pt`（clean run1 ep15） |
| `max_frames` | **16**（兩份對齊） |
| `motion_thr` | 2.0 px — GT mask 由 flow-residual 導出 |
| `label_thr` | 0.5 — patch label 二值化閾值（pooled mask > 0.5 → dynamic） |
| `k` | 30.0 |
| **Δ% 基準** | **同一列（該序列）自己的 `off`**。例：`market_5` 的 ×10 `−16.7%` = 除以它自己的 off 0.0888，**不是**除以全體平均。**各列分母不同（off 從 0.0025 到 0.1232 差 50 倍），跨序列的 Δ% 不可相加或平均** |
| join | quality 有 14 seq、sweep 有 13 seq（缺 `alley_2`）→ **交集 13 seq**；相關係數用 11 seq（再去 `sleeping_1/2` 的 NaN） |

**指標意義**

| 欄 | 意義 | 讀法 |
|---|---|---|
| `dyn_frac` | GT 動態 patch 佔比 | 場景有多少在動 |
| **`AUC`** | ROC-AUC，gate logit 對 GT label 的**排序**能力 | **與閾值無關** → 純測「gate 知不知道哪裡在動」。1.0 = 完美排序 |
| **`F1`** | 在固定閾值 0.5 下的 F1 | **與閾值有關** → 測「gate 敢不敢在 0.5 以上表態」。**AUC 高但 F1 低 = 排序對但機率被壓低（欠自信）** |
| `p_dyn` | GT 動態 patch 上的平均 `σ(g)` | gate 對「真的在動」的地方給多少機率。**< 0.5 = 欠自信** |
| `p_stat` | GT 靜態 patch 上的平均 `σ(g)` | 誤報程度 |
| `gap` | `p_dyn − p_stat` | 兩類的分離度 |
| `off` | 該序列 gate 關閉時的 ATE | 基準 |
| **`oracle Δ`** | 用 GT mask 時 ATE 的變化 | **該序列的頭空上界。與 gate 品質無關**（用的是 GT，不是預測） |
| **`pred Δ`** | 用訓練出的 gate 時 ATE 的變化 | 實際兌現了多少 |
| `×3 / ×10 Δ` | logit 乘 3/10 後的 ATE 變化 | 人工放大信心的診斷 |

> ⚠️ **不要用 BCE (`loss_gate`) 判斷 gate 品質。** BCE 的 label 是 `adaptive_avg_pool2d` 出來的 soft 值，boundary patch 有不可約 floor → val BCE 會看似 overfit 卻與高 AUC 並存。一律用 AUC / F1。

### 表

| seq | dyn_frac | AUC | F1 | p_dyn | gap | off | **oracle Δ** | **pred Δ** | ×3 Δ | ×10 Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| temple_3 | 0.136 | 0.989 | 0.836 | 0.681 | 0.647 | 0.0885 | −9.7 | +0.6 | +3.9 | −2.3 |
| temple_2 | 0.030 | 0.972 | 0.302 | 0.294 | 0.257 | 0.0268 | −3.4 | +1.3 | +0.0 | −2.1 |
| **ambush_5** | 0.332 | 0.968 | 0.323 | 0.334 | 0.296 | 0.0252 | **−27.2** | −0.7 | −4.4 | −5.9 |
| market_2 | 0.218 | 0.962 | 0.432 | 0.379 | 0.318 | 0.0110 | +0.4 | **+4.3** | −0.8 | −1.1 |
| shaman_3 | 0.042 | 0.943 | 0.481 | 0.597 | 0.425 | 0.0025 | −0.8 | −2.3 | −6.6 | −8.3 |
| **market_5** | 0.371 | 0.932 | 0.554 | 0.442 | 0.364 | 0.0888 | **−22.3** | −3.1 | **−12.5** | **−16.7** |
| market_6 | 0.068 | 0.909 | 0.148 | 0.232 | 0.185 | 0.0299 | −1.3 | −0.5 | −0.1 | +1.4 |
| **cave_2** | 0.407 | 0.863 | 0.073 | 0.173 | 0.142 | 0.0536 | **+50.7** | +2.3 | +2.3 | +0.5 |
| ambush_6 | 0.325 | 0.814 | 0.184 | 0.267 | 0.166 | 0.1232 | −10.3 | −0.6 | −0.0 | −0.6 |
| **cave_4** | 0.516 | **0.632** | 0.087 | 0.140 | 0.058 | 0.0220 | **−26.7** | +0.1 | +1.0 | +1.5 |
| **ambush_4** | 0.364 | **0.618** | 0.030 | 0.101 | 0.033 | 0.0867 | +5.0 | +0.3 | +1.1 | +1.8 |
| sleeping_1 | 0.000 | nan | nan | nan | nan | 0.0068 | +0.0 | +1.9 | +1.5 | **+11.0** |
| sleeping_2 | 0.000 | nan | nan | nan | nan | 0.0039 | +0.0 | −2.0 | +0.0 | +0.0 |

**總計**：macro **AUC 0.883 / F1 0.347 / p_dyn 0.346 / p_stat 0.065 / gap 0.283**
micro（132608 patch）：AUC 0.839 / F1 0.289 / p_dyn 0.274 / p_stat 0.065 / dyn_frac 0.202

### 相關係數（Pearson, n=11）

| | vs oracle Δ | vs pred Δ | vs ×10 Δ |
|---|---|---|---|
| AUC | **−0.006** | +0.029 | −0.466 |
| F1 | −0.317 | −0.163 | **−0.577** |
| p_dyn | −0.237 | −0.254 | **−0.563** |
| gap | −0.203 | −0.144 | −0.510 |
| dyn_frac | −0.052 | −0.000 | +0.058 |

### 附註

- **`corr(AUC, oracle Δ) = −0.006`** 是正確的 sanity check：oracle 用 GT mask，本就該與預測品質無關。**它同時說明「該序列有多少頭空」與「gate 有多好」是兩個獨立的量。**
- **`corr(F1/p_dyn, ×10 Δ) ≈ −0.57`**：放大只在 gate 本來就半自信的序列有效（`market_5` F1 0.554 → ×10 **−16.7%**；`shaman_3` p_dyn 0.597 → ×10 −8.3%）。**對真正不自信的序列（`cave_2` p_dyn 0.173）放大無效。**
- **`p_dyn` 無任何序列超過 0.7**（最高 `temple_3` 0.681）；12 個有效序列中 **9 個 < 0.5**。「機率被壓在 0.5 以下」證實。
- **AUC 與 `dyn_frac` 反相關**：`dyn_frac ≥ 0.32` 的六個序列 AUC 全部 ≤ 0.932，最差兩名 `cave_4`(0.516→AUC 0.632)、`ambush_4`(0.364→AUC 0.618)。**動態佔比越高，gate 排序越差。**
- **`ambush_5` 是「欠自信」最乾淨的單一證據**：AUC 0.968（排序幾乎完美）+ oracle **−27.2%**（全場最大頭空）+ pred **−0.7%**（幾乎沒兌現）。
- **`cave_4` 是頭空/品質落差最大的序列**：AUC 0.632（排序最差之一）但 oracle **−26.7%**（頭空第二大）。**頭空大而 gate 完全抓不到。**
- **`cave_2` 是唯一的 oracle 災難**：oracle **+50.7%** —— 用了完美 mask 反而讓 ATE 惡化一半。它的 AUC 0.863 並不低，F1 0.073 / p_dyn 0.173 則是全場最不自信之一。**oracle 傷害與 gate 品質在此完全脫鉤，指向 GT mask 或機制本身的問題，而非 gate 學得好不好。**
- **`market_2` 的 pred +4.3%** 是全場最差的 predicted，與其 AUC 0.962 矛盾。
- **`sleeping_1` 全靜態（dyn_frac=0）卻在 ×10 吃到 +11.0%** —— 全靜態序列被放大的 gate 主動傷害，是放大策略的風險上界。
- `sleeping_1/2` 的 `dyn_frac = 0` → AUC/F1 為 NaN，macro 僅用 12 seq、相關係數僅用 11 seq。

---

## 已知資料缺口

| # | 缺口 | 影響 |
|---|---|---|
| 1 | **base (VGGT-1B) 只有 f50 一個長度**（表 2.3） | 無法分離「bug fix 收益」與「短序列本身較易」 |
| 2 | **f12 vs f16 的 11 個百分點跳動無法解釋**（表 2.2） | 「短序列武器」的論述基礎不穩 |
| 3 | **`gate_quality` 只有 f16 一個長度**（表 3） | f16 恰是 oracle 頭空的凹陷點（−3.8%），拿它解釋 f12/f32 有風險 |
| 4 | **clean 世代沒有任何 photo / smooth / temporal 的 run** | 該線的貢獻完全未知；表 1 中相關數字全是 buggy 世代 |
| 5 | **無成本 / 效率資料** | feed-forward vs TTO 是核心宣稱，但 wall-clock、VRAM、參數量、是否需 per-scene optimization 全部沒有記錄 |
| 6 | **無標籤品質對照**（inst / m_geo / RAFT 三來源的 dyn_frac） | `cave_2` 的 oracle +50.7% 無法判斷是機制錯還是標籤錯 |
| 7 | **訓練雜訊帶未正式量化** | run1 的 24 個 epoch pose_eval 在 0.1533~0.2225 間跳動，ep24 (0.1713) ≈ ep2 (0.1715)，無收斂趨勢。**雜訊帶約 ±0.02（±13%）** → 任何小於此的「改善」不可信 |
