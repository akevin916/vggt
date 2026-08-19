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
| 資料來源 | `outputs/train/*/eval_sintel/`、`training/logs/*/pose_eval/epoch_*/`、`archive/logs/v1_train/`；v3CLEAN photo/smooth_temp 列（‡）來自 `outputs/<run>/eval_sintel/<ckpt>/`（`training/eval_three_runs.sh`，**max_depth=70**） |

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
| **v3CLEAN smooth_temp `ep30`** ▲ | clean | **0.1343** | **0.0500** | 0.0633 | **0.3943** | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ |
| v3CLEAN smooth_temp `ep40` | clean | 0.1340 | 0.0597 | 0.0642 | 0.4092 | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ |
| v3CLEAN smooth_temp `ep50`(=`last`) | clean | 0.1360 | 0.0588 | 0.0627 | 0.4046 | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ |
| v3CLEAN photo `ep10`(=best) | clean | 0.1583 | 0.0496 | 0.0569 | 0.4562 | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ |
| v3CLEAN photo_smooth_temp | clean | 0.1743 | 0.0779 | 0.0917 | 0.6096 | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ |
| **MonST3R（目標）** | ref | **0.1080** | — | **0.0420** | 0.7320 | 0.3450 | — | — | — | 0.562 | — | — |

### 附註

- ⚠️ **`v3bug smooth_temp` 遠未收斂 —— 該列是 epoch 2 的數字**。已由 checkpoint metadata（`prev_epoch`）確認：
  - 被評測的 `dyn_vggt_v3_s1_smooth_temporal.pt` = 該 run 的 `best.pt` = **epoch 2**（train steps 6003）
  - `last.pt` = epoch 4；log 在 epoch 5 進行中時中斷（07-06 18:54 → 22:54）
  - config 設定為 **20 epoch**

  **所以 0.1568 是「20 epoch 計畫中第 2 個 epoch」的快照**，不能當作 smooth+temporal 的收斂值。同組 `s1_inst_smooth`（Ep 0 step 36）與 `_v2`（**Ep 0**，train steps 2001）死得更早。**smooth 這條線至今沒有任何跑完的 run，也沒有任何 run 的 checkpoint 超過 epoch 4。**
- **v1 S0 與 VGGT-1B 逐位元相同** —— `enable_*` flag 全關時與 pretrained 權重相容的回歸證據。
- **`run1 ep15` 是 depth 最好**：AbsRel / sqRel / RMSE / logRMSE / δ1 / δ2 / δ3 七項全贏 base。（原本標「全面最好」，但 pose 已被 `smooth_temp ep30` 超越 —— 見下。）
- ▲ **`smooth_temp ep30` 是目前 pose 最好（全 clean 家族最低 ATE）**：ATE **0.1343**、ATE(12) **0.0500**、RPE-r **0.3943** 三項全表最低（ref MonST3R 除外），ATE 比 warm-start `run1 ep15`（0.1533）低 **12%**、比 base（0.1714）低 **22%**。距 MonST3R 本機實測（0.1110，表 4）仍差 **+21%**。
- ⚠️ **【2026-08-03 更正】此 run 已跑完 50 epoch 並收斂，先前記錄的 `last` = 0.1253 已失效。** 舊記錄寫於 `last.pt` 還停在 ~ep32 時的暫態快照，該檔案已被後續 epoch 覆蓋（現 `last` == `epoch_50` == 0.1360）。**全 ckpt full-seq 掃描：**

  | ckpt | ep10 | ep20 | **ep30** | ep35 | ep40 | ep45 | ep50(=`last`) |
  |---|---|---|---|---|---|---|---|
  | ATE | 0.1876 | 0.1796 | **0.1343** | 0.1370 | 0.1340 | 0.1356 | 0.1360 |
  | ATE(12) | — | — | **0.0500** | — | 0.0597 | — | 0.0588 |

  **ep30 之後平台化在 0.134~0.136（帶寬 ±0.002）** —— 「晚期單調下降、仍在訓練」的敘述已被否證，**沒有「再訓久一點就追上 MonST3R」的空間**。
- **ep40 的 ATE（0.1340）與 ep30（0.1343）看似平手，但 ATE(12) 差 19%（0.0597 vs 0.0500）** → ep40 的持平是靠 `cave_2`/`temple_3` 兩個離群序列改善換來的，其餘 12 個序列反而變差。**取 `ep30` 為此 run 的代表 ckpt。**
- （`best_loss.pt` full-seq ATE 0.1344 ≈ ep30，屬同一平台，非獨立資訊。）
- ⚠️ **這四列的 checkpoint 選擇要看清楚 —— windowed `best.pt` 不可信**：這些 run `pose_eval.enabled=false`，`best.pt` 由 noisy windowed val ATE（channel A，~0.017 scale）選，與 full-seq 不對齊。`smooth_temp` 的 windowed `best.pt`（ep17）full-seq ATE = **0.1651**，遠差於真正最佳的 `epoch_30`（0.1343，差 23%）。**要用這條線一律取 `epoch_30.pt`，不要用 `best.pt`，也不要用 `last.pt`（=ep50，ATE(12) 較差）。**
- **`photo` 越訓越差**：full-seq 最佳 0.1583（=`ep10`）**比它自己的 warm-start 起點 0.1533 還差**，且 ep10→ep20 一路劣化（static-photo loss 傷 pose）。
- **`photo_smooth_temp` 比 base 還爛**（0.1743，且 ep10 0.199 → last 0.207 單調惡化）→ lineage 不乾淨、確認棄用。
- ‡ **這四列的 depth 欄（AbsRel…δ3）省略**：`eval_three_runs.sh` 用 `max_depth=70`（eval_sintel 預設），與本表 max80 協定不同（max70 裁掉遠景會使 RMSE/AbsRel 假性變好，不可同欄比）。pose 欄不受 max_depth 影響，可直接比。要補齊 depth 需以 `--max_depth 80` 重評這幾個 ckpt。
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

## 表 4 — MonST3R 協議對齊對照（Dyn-VGGT vs MonST3R）

> ⚠️ **本表與表 1 的 depth 數字不可混比。** 表 1 用 14-seq / `max_depth=80` / per-frame median scale；
> 本表改用 **MonST3R 原生協議**：depth 在 **23-seq**、**lad2 scale+shift**（整段 pool）、`max_depth=70`+post-clip 70、
> 跨序列 **valid-pixel 加權**。pose 協議與表 1 相同（14-seq、sim3-aligned），故本表 Dyn-VGGT 的 ATE 0.1533
> 與表 1「v3CLEAN run1 ep15」的 0.1533 一致（互為 sanity check）。

### 設定

| 項目 | 值 |
|---|---|
| script | `training/benchmark/eval_sintel.py`（2026-07-22 改版，對齊 MonST3R depth 協議） |
| Dyn-VGGT ckpt | `training/logs/dyn_vggt_v3_s1_inst/ckpts/best.pt`（= clean run1 ep15） |
| pose 序列 | **14 seq**（`SINTEL_EVAL_SEQUENCES`） |
| depth 序列 | **23 seq**（`final/` 全部，MonST3R `--full_seq` 口徑；共 1064 幀） |
| depth 對齊 | **lad2** = scale+shift，Adam L1，lr=1e-4 / 1000 iters（`absolute_value_scaling2`，自 MonST3R verbatim port） |
| depth 粒度 | **整段序列 pool**：全幀堆疊後擬合單一 (s,t)，metric 在全部 valid pixel 上算 |
| `max_depth` / post-clip | **70 / 70** |
| 跨序列平均 | **valid-pixel 加權**（`np.average(weights=valid_pixels)`） |
| pose 指標 | evo：ATE=`ape(trans, align, correct_scale)`；RPE=`rpe(δ=1 frame, all_pairs)` —— 與 MonST3R `eval_metrics` 逐項相同 |
| MonST3R 版本 | `MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth`，`launch.py --mode=eval_pose`（pose 預設；depth 加 `--full_seq --no_crop`） |
| MonST3R 數字 | **本機實測**（非論文抄錄），`reference/monst3r/results/{sintel_pose,sintel_video_depth}/` |
| **Δ% 基準** | 本表無 Δ%，全為絕對值 |

### 表

**Camera Pose — 14 seq**

| 方法 | ATE ↓ | RPE-trans ↓ | RPE-rot ↓ |
|---|---|---|---|
| MonST3R | **0.1110** | **0.0440** | 0.7803 |
| **Dyn-VGGT** (v3 s1_inst/best) | 0.1533 | 0.0673 | **0.4937** |

**Video Depth — 23 seq**

| 方法 | AbsRel ↓ | sqRel ↓ | RMSE ↓ | logRMSE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|---|---|---|---|---|---|---|---|
| MonST3R | 0.3330 | 3.507 | 4.501 | 0.4286 | 0.5909 | 0.8010 | 0.8863 |
| **Dyn-VGGT** | **0.1731** | **1.019** | **3.753** | **0.3136** | **0.7495** | **0.8740** | **0.9424** |

### 附註

- **Depth：Dyn-VGGT 七項全勝**（AbsRel 0.173 vs 0.333、δ1 0.750 vs 0.591）。feed-forward 的 point/depth head 對上 MonST3R 的 per-scene 優化仍全面領先。
- **Pose 分歧**：MonST3R 的 **ATE / RPE-trans 較低**（平移軌跡較準，得益於 per-seq global-alignment 優化），但 Dyn-VGGT 的 **RPE-rot 明顯更好**（0.494 vs 0.780）。**缺口是純平移的**，與表 1、表 2.3 附註一致。
- **本機實測的 MonST3R 數字與表 1「MonST3R（目標）」列（論文抄錄：ATE 0.108 / RPE-t 0.042 / RPE-r 0.732 / AbsRel 0.345 / δ1 0.562）有小幅差異**，屬實測 vs 論文的正常落差；本表以實測為準，因為它與 Dyn-VGGT 走完全相同的序列集與對齊碼。
- **殘留協議差異（未對齊）**：Dyn-VGGT 推論用 crop@518，MonST3R depth 用 `--no_crop`。此為模型輸入前處理、非評估方法；強制 no_crop 可能反使 VGGT 吃虧（原生 crop 訓練），故保留。
- 對齊碼改動：`eval_utils/metrics_depth.py`（新增 `absolute_value_scaling2` + lad2/pool/加權）、`data/sintel_io.py`（`list_sintel_full_sequences`）、`benchmark/eval_sintel.py`（depth-23 / pose-14 拆分、`max_depth` 預設 70）。trainer 線上 val（`metrics_val.py`）仍走 per-frame median，未受影響。

---

## 表 5 — MonST3R flow_loss 可行性探針

> 表 5 本體是 PointOdyssey（訓練分布）；跨資料集見 **表 5.1**、脫鉤測試見 **表 5.2**。

### 設定

| 項目 | 值 |
|---|---|
| script | `training/diag/flow_loss_probe.py --dataset po` |
| ckpt | `logs/dyn_vggt_v3_s1_inst_smooth_temporal/ckpts/epoch_40.pt` |
| 資料 | PointOdyssey **train** split，20 clips × 16 幀，`dynamic_source=instance` |
| 幀對 | clip 內相鄰對，**Δt ≤ 5**（對齊 `dynmask_inst` 的 gap-5 基準），雙向，n = 388 rows |
| mask | `dynmask_inst`（靜態）∧ `point_masks`（GT depth 有效）；平均覆蓋 67.3% 像素 |
| target | RAFT-large（torchvision DEFAULT 權重，20 iters），跑在模型輸入解析度上（518 zero-pad 至 8 的倍數後裁回） |
| ego-flow | `warp_by_disp` 視差形式，monst3r `goem_opt.py:196` 逐行移植 |
| loss | `smooth_L1_loss_fn`（beta=1.0, per_pixel_thre=50），monst3r `optimizer.py:18` 逐行移植 |
| 資料來源 | `outputs/dyn_vggt_v3_s1_inst_smooth_temporal/flow_loss_probe/results_po.json` |

**變體定義**(ego-flow 用什麼幾何量生成;mask 與 target 三者共用)

- `gt` — 全部 GT pose/depth/K。**這是地板**:pose 完美時剩下的殘差 = RAFT 誤差 + mask 誤差 + 離散化
- `pred` — 全部模型預測。**訓練時實際會看到的值**
- `pose_only` / `depth_only` — 混合版本。⚠️ 混合前必須先對齊尺度:ego-flow 的平移項是 `視差 × t`,VGGT 在自己正規化的尺度、GT 在公尺,直接混合量到的是單位不符而非幾何誤差(未對齊時 `pose_only` 假性劣化 5×)。此處以 `median(GT depth / pred depth)` 逐 clip 換算

### 表

**per-pixel 差異 `|ego_flow − raft_flow|`（靜態+有效像素，單位 = 像素）**

| 變體 | p50 | p90 | p99 | loss 中位數 |
|---|---|---|---|---|
| `gt`（地板） | **0.09** | 0.28 | 0.91 | 0.04 |
| `depth_only` | 0.12 | 0.36 | 0.93 | 0.05 |
| `pose_only` | 0.15 | 0.40 | 0.93 | 0.06 |
| `pred` | **0.16** | 0.40 | 0.93 | 0.06 |

**整項熔斷率**(`flow_loss_fwd + flow_loss_bwd > 閾值` → 該步整項丟棄)

| 閾值 | 5 | 10 | 20 | **50**(monst3r 原值) | 100 | 200 |
|---|---|---|---|---|---|---|
| `gt` | 3.6% | 1.0% | 0.5% | **0.0%** | 0.0% | 0.0% |
| `pred` | 3.1% | 0.5% | 0.5% | **0.0%** | 0.0% | 0.0% |

NaN 率 0.0%;`per_pixel_thre=50` 的保留率 99.4~99.8%。

**按幀距拆解(loss 中位數,單向)**

| Δt | n | `gt` | `pred` |
|---|---|---|---|
| 1 | 122 | 0.03 | 0.03 |
| 2 | 76 | 0.03 | 0.04 |
| 3 | 88 | 0.06 | 0.07 |
| 4 | 46 | 0.08 | 0.08 |
| 5 | 56 | 0.08 | 0.10 |

### 附註

- **熔斷不是問題（陰性結果，可直接引用）**：在 monst3r 原值 50 上觸發率 **0%**。「照抄常數會讓 loss 靜默變成 no-op」這個顧慮**已排除**。NaN 率 0%，monst3r `nan > thre` 為 False 的那個漏洞未被觸發。
- **管線正確性的旁證**：`gt` 的 p50 僅 **0.09 px** —— GT 導出的 ego-flow 與 RAFT 在靜態像素上中位差不到十分之一像素。座標系、padding、mask 對齊、視差形式若有任一處接錯，此值不可能這麼小。
- **PO 上沒有可優化的訊號**：`pred`(0.16 px)僅比地板(0.09 px)高 0.07 px，**殘差的多數是 RAFT/mask 雜訊地板而非模型誤差**。按 Δt 拆解也一致 —— 誤差隨幀距成長(0.03 → 0.10)，但 `gt` 全程貼著 `pred` 一起長，**成長的是地板(遮蔽、大位移下 RAFT 變難)，不是模型劣化**。
- **depth 與 pose 的貢獻**：相對地板的增量為 `depth_only` +0.03、`pose_only` +0.06。pose 約為 depth 的兩倍，但**兩者皆為次像素等級**。→ 「flow loss 需解凍 depth_head」的結構性理由(深度與位移在式子中相乘、會互相補償)仍成立，但**在 PO 上 depth 不是瓶頸**。
- ⚠️ **本表不能外推到 Sintel**。PO train 是模型的訓練分布(且 `epoch_40` 已在其上收斂)，殘差小是預期的。ATE 的缺口在 Sintel(域外)。表 4 顯示 **RPE-trans 我方輸 53%**(0.0673 vs MonST3R 0.0440) —— 那是逐幀、局部的誤差，正是逐對 flow loss 可能修的東西。**決定性測量見下方表 5.2。**

---

### 表 5.1 — 跨資料集(Δt=1，唯一地板可信的格子)

| 資料集 | 地板 `gt` | `pred` | **比值** | 靜態+有效像素 |
|---|---|---|---|---|
| PointOdyssey (train) | 0.06 | 0.11 | 1.9× | 67% |
| TartanAir | 2.02 | 2.93 | 1.5× | 91% |
| Waymo | 0.54 | 1.02 | 1.9× | **15%** |
| Spring | 0.12 | 0.25 | 2.0× | 84% |
| **Sintel**(域外) | 0.10 | 0.46 | **4.6×** | 76% |

**RAFT 地板隨 Δt 崩壞**(`gt` 中位數，px)：

| Δt | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| PointOdyssey | 0.06 | 0.09 | 0.13 | 0.13 | 0.11 |
| TartanAir | **2.02** | 5.22 | 6.76 | 8.55 | **11.31** |
| Waymo | 0.54 | 1.15 | 2.46 | 3.78 | 1.54 |
| Spring | 0.12 | 0.68 | 1.93 | 1.21 | 1.24 |

- **四個訓練集全部飽和(1.5–2.0×)，只有域外的 Sintel 是 4.6×。** 原本預期取樣配額最低的 Spring(5%)可能未擬合 —— 否證，它是 2.0×。
- **PO 是唯一在 Δt≤5 全程地板都穩的資料集**。TartanAir 是無人機快速飛行，Δt=5 時地板 11.31 px —— 該格量到的 100% 是 RAFT 在崩，與模型無關。→ **任何 flow-based loss 只能用 Δt=1**；先前基於 `dynmask_inst` gap-5 基準推出的「Δt ≤ 5」是不夠緊的。
- Waymo 的靜態+有效像素只有 **15%**(LiDAR 稀疏)，其絕對值不可與其他資料集直接比。

### 表 5.2 — 脫鉤測試：flow 殘差不是 ATE 的代理

同一 run 的兩個 ckpt，ATE 差 21.6%，Sintel(Δt=1)的 flow 殘差幾乎不動：

| | VGGT-1B | ep30 | 變化 |
|---|---|---|---|
| **full-seq ATE** | 0.1714 | 0.1343 | **−21.6%** |
| Sintel `gt`(地板) | 0.100 | 0.100 | 0.0% |
| Sintel `pred` p50 | 0.488 | 0.464 | −4.9% |
| Sintel `pred` p90 | 1.353 | 1.338 | −1.1% |
| PO `pred` p50 | 0.113 | 0.111 | −1.5% |

- **地板兩邊完全相同 → 不是量測漂移。**
- 此 run 加的是 `camera_smooth`(軌跡層級)與解凍 `temporal_blocks`(跨幀通路)，**兩者都在 pair 層級之上** —— 與「逐對殘差不動」一致。
- ⚠️ 此表證明的是「flow 殘差不能當 ATE 的代理指標」與「至今的 ATE 收益不來自逐對幾何」。它**不**證明壓低殘差對 ATE 無用 —— 那由表 6 直接干預測得。
- ⚠️ ep30 的 Sintel `pred` **尾巴變差**(p99 3.758 → 4.142，loss p90 2.844 → 4.426)。訓練把軌跡拉好的同時讓少數區域的局部幾何更糟，成因未查。
- 跨 clip 的 `GT/pred depth scale` 中位數 4.699（p10 3.59 / p90 7.77）。此散布是 **預期且無害** 的：VGGT 對每個場景各自正規化，而 loss 與 ATE 均在單一序列內運作、ego-flow 輸出又是像素單位，clip 內自洽即足夠。此值的唯一用途是讓混合變體可比。**clip 內部的尺度漂移本表未量測。**

---

## 表 6 — flow 殘差的 ATE 頭空(TTO 上界)與正規化掃描

### 設定

| 項目 | 值 |
|---|---|
| script | `training/diag/flow_pose_headroom.py` |
| ckpt | `logs/dyn_vggt_v3_s1_inst_smooth_temporal/ckpts/epoch_30.pt`(ATE 0.1343) |
| 做什麼 | **網路凍結**，只把它輸出的 pose 當變數微調(每幀 6 個數，初值 0)，最小化對 RAFT 的 flow 殘差 |
| 序列 | Sintel 14 seq，每序列前 50 幀，**Δt=1** |
| 迭代 | 300 步 Adam，lr 1e-3，每 25 步評一次 ATE |
| sanity check | iter 0 的 ATE **必須**等於 benchmark per-seq 值 —— 實測 `market_5` 0.1398 完全吻合 |
| 目標函數 | `w_flow·flow + w_anchor·anchor + w_smooth·smooth + w_dprior·dprior` |
| `anchor` | `dw² + (dt/軌跡長度)²` —— **取代 monst3r `(li+lj)` 的「錨定」角色**(拼接角色 VGGT 不需要) |
| ⚠️ 用到 GT | 目標函數零 GT，但**遮罩用 GT 導出的 `m_geo`** 與 GT 深度有效性 → 數字偏樂觀 |
| 資料來源 | `outputs/<run>/flow_pose_headroom/epoch_30/results_*.json` |

### 表

**基準(iter 0)**:ATE(12) **0.0500** / ATE(14) **0.1343**;MonST3R 本機實測 14seq = **0.1110**

| 配置 | depth | w_anchor | w_smooth | w_dprior | **ATE(12)** | Δ% | ATE(14) | Δ% | flow Δ% |
|---|---|---|---|---|---|---|---|---|---|
| **`a100`** | − | **100** | 0.01 | 0 | **0.0414** | **−17.0%** | **0.1266** | **−5.7%** | −61.1% |
| `a100_dil4` | − | 100 | 0 | 0 | 0.0413 | −17.2% | 0.1267 | −5.7% | −64.5% |
| `a100_dil8` | − | 100 | 0 | 0 | 0.0414 | −17.1% | 0.1268 | −5.6% | −65.5% |
| `a10` | − | 10 | 0.01 | 0 | 0.0439 | −12.2% | 0.1328 | −1.1% | −69.9% |
| `a100+depth+dprior` | **Y** | 100 | 0.01 | **1.0** | 0.0453 | −9.3% | 0.1307 | −2.7% | **−72.2%** |
| `a1000` | − | 1000 | 0.01 | 0 | 0.0466 | −6.6% | 0.1313 | −2.2% | −51.1% |
| `a1` | − | 1 | 0.01 | 0 | 0.0480 | −3.8% | 0.1522 | +13.4% | −72.2% |
| **純 flow** | − | **0** | 0 | 0 | **0.0502** | **+0.6%** | **0.1630** | **+21.4%** | −73.1% |
| 純 flow + depth | Y | 0 | 0 | 0 | 0.0512 | +2.6% | 0.1616 | +20.3% | −81.8% |
| `a10 + smooth 1.0` | − | 10 | **1.0** | 0 | 0.0698 | +39.7% | 0.1505 | +12.1% | **+75.6%** |

**純 flow 的收斂軌跡**(所有序列同一步數，非 oracle):

| iter | 0 | 50 | **100** | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| flow | 2.131 | 0.854 | 0.705 | 0.643 | 0.611 | 0.573 |
| ATE(12) | 0.0500 | 0.0453 | **0.0446** | 0.0467 | 0.0487 | 0.0502 |
| Δ% | — | −9.3% | **−10.8%** | −6.4% | −2.5% | +0.6% |

### 附註

- **主結果:`w_anchor=100` 給出 ATE(12) −17.0%，而且是穩定的**(終值 0.0414 ≈ 路徑最佳 0.0410，不需早停)。且**優於純 flow 版早停撈到的最好值**(0.0446) —— 正規化不只防止衝過頭，它找到了更好的解。ATE(14) 也首次改善(−5.7%)。
- **anchor 權重有清楚的 U 型最佳點**:0 → 0.0502、1 → 0.0480、10 → 0.0439、**100 → 0.0414**、1000 → 0.0466(拉太緊，回到初值)。
- **flow 降幅與 ATE 改善反向**:`a100` 只降 61.1% 卻是最好的；純 flow 降 73.1%、`+depth` 降 81.8% 反而更差。**殘差降得多 ≠ pose 變準**，此表出現三次。
- ❌ **「退化方向是深度、需 depth prior」假說被推翻**:放開 depth + prior 為 0.0453，比凍結 depth 的 0.0414 **更差**。逐序列看收益被稀釋約一半(`market_5` −35.2% → −20.0%、`market_6` −36.8% → −13.6%)。多給的自由度讓優化器**把殘差吸收進深度而非修正 pose**。
- ❌ **「遮罩漏遮是離群崩壞的成因」假說被推翻**:膨脹 4/8 px 後 ATE(12) 為 0.0413/0.0414，與未膨脹的 0.0414 **無差別**。原本崩壞的 `cave_2`(+12.6% → −2.5%)、`temple_3`(+65.1% → +3.3%)是**錨定**救回來的。
- ❌ **monst3r 的一階平滑無貢獻**:`w_smooth=0.01` 與完全不加**逐位元相同**(0.0502 vs 0.0502)；`w_smooth=1.0` 是災難(+39.7%，flow 殘差還升 75.6%)。
- **純 flow 版的 `alley_2` 是最乾淨的單一反例**:flow 單調 −87.7%(0.244 → 0.030)、ATE 單調 **+151.2%**(0.0159 → 0.0401)。平滑、單調 —— 不是優化失敗，是**該殘差的極小值不在正確的 pose 上**。
- **`w_flow : w_anchor = 1 : 100` 與 monst3r 的 `flow 0.01 : 骨幹 1.0` 同量級**，兩邊獨立掃出同一個比例。
- ⚠️ 這是 **TTO 上界**。訓練 loss 只能逼近，且本表的遮罩用了 GT。0.1266 仍高於 MonST3R 的 0.1110(它另外優化 depth、focal，且有點雲骨幹)。

---

## 表 7 — 光流對相機位姿的觀測度(逐序列)

### 設定

| 項目 | 值 |
|---|---|
| script | `training/diag/flow_observability.py` |
| ckpt | 同表 6(`epoch_30`) |
| 做什麼 | 擾動預測位姿，量 **ego_flow 變了多少**(px)。旋轉與平移分開擾動 |
| 擾動幅度 | 旋轉 0.005 rad/幀；平移 = **軌跡長度的 1%**/幀。隨機方向 20 次取平均 |
| 像素集 | 與 loss 相同(靜態 ∧ GT 深度有效)，Δt=1 |
| **目標無關** | 只比 ego_flow 對 ego_flow，**不用 RAFT 也不用 GT flow** → 結論同時適用 RAFT-target 與 GT 導出的 `L_ego_flow` |
| `視差量` | `中位視差 × 軌跡長度` —— **無量綱**。單獨的視差在 VGGT 的逐序列正規化尺度下不可跨序列比 |
| 資料來源 | `outputs/<run>/flow_observability/epoch_30/results.json` |

### 表

| seq | 旋轉敏感度 | **平移敏感度** | 平移/旋轉 | 視差量 | 靜態% | RPE-t | 表 6 `a100` Δ% |
|---|---|---|---|---|---|---|---|
| market_2 | 3.788 | **0.068** | 0.018 | 0.008 | 81.3% | 0.006 | −0.9% |
| cave_4 | 2.657 | 0.335 | 0.126 | 0.058 | 63.6% | 0.036 | −3.4% |
| **cave_2** | 2.943 | **0.455** | 0.155 | 0.058 | 63.8% | **0.322** | **−2.5%** |
| sleeping_2 | 2.023 | 0.466 | 0.230 | 0.127 | 100.0% | 0.004 | +0.7% |
| temple_3 | 4.238 | 0.604 | 0.143 | 0.085 | **7.8%** | 0.196 | +3.3% |
| ambush_4 | 2.402 | 0.625 | 0.260 | 0.067 | 69.1% | 0.065 | −8.8% |
| alley_2 | 2.032 | 0.665 | 0.327 | 0.197 | 98.3% | 0.011 | +19.9% |
| sleeping_1 | 3.089 | 0.919 | 0.298 | 0.174 | 98.2% | 0.005 | −20.8% |
| shaman_3 | 3.540 | 0.936 | 0.265 | 0.136 | 94.1% | 0.003 | −16.0% |
| ambush_5 | 3.259 | 1.343 | 0.412 | 0.205 | 57.2% | 0.022 | −6.5% |
| ambush_6 | 2.915 | 1.415 | 0.485 | 0.182 | 68.8% | 0.087 | −10.3% |
| temple_2 | 2.378 | 2.079 | 0.874 | 0.441 | 58.8% | 0.031 | +0.3% |
| **market_6** | 2.828 | **3.086** | 1.091 | 0.513 | 82.5% | 0.034 | **−36.8%** |
| **market_5** | 1.796 | **5.799** | 3.230 | 0.715 | 76.5% | 0.064 | **−35.2%** |

**相關係數**(Pearson, n=14):

| | 平移敏感度 | 平移/旋轉 | 視差量 | 光流殘差 | 動態程度 | ATE |
|---|---|---|---|---|---|---|
| vs 表 6 `a100` Δ% | **−0.710** | −0.630 | −0.621 | −0.42 | −0.10 | +0.11 |

`corr(視差量, 平移敏感度)` = **+0.957**　　`corr(視差量, 旋轉敏感度)` = −0.492

### 附註

- **完整的因果鏈**:`靜態參考物有多近 × 相機走多遠` **─(+0.957)→** `平移在光流裡有多明顯` **─(−0.710)→** `光流優化能改善多少`。**平移敏感度是目前最強的預測指標**，勝過光流殘差(−0.42)。
- **平移敏感度跨序列差 85 倍**(0.068 → 5.799)，**旋轉敏感度只差 2.3 倍**(1.8 → 4.2)。這正是幾何預測:平移項乘上視差、旋轉項不含深度。旋轉是有效的對照組。
- **`cave_2` 之謎解決**:它的 RPE-t 是全場最差(0.322，平移確實錯得嚴重)，但平移敏感度只有 0.455 —— **光流物理上看不見那個誤差**，所以殘差小(表 5 的 0.50 px)、優化推不動(−2.5%)。**不是方法沒調好，是該序列的平移不可觀測。**
- **先前四個假設全數否證**(皆為間接代理指標):累積漂移(被 RPE 反駁 —— RPE 是局部指標且 `cave_2` 最差)、深度多樣性不足(`cave_2` 的 p90/p10 = 13.0，優於多數)、絕對視差過小(corr −0.27)、紋理不足(`cave_2` 的靜態像素梯度 11.0 **高於** `market_5` 的 8.0)。**直接量測觀測度才得到正解。**
- **可預測的限制**:低視差量的序列，**任何以光流為目標的 loss 都幫不上忙** —— 因為本測試目標無關，此限制對 `L_ego_flow` 同樣成立。
- ⚠️ `corr(視差量, 旋轉敏感度)` = −0.492 而非預測的 ≈0。旋轉敏感度仍受**預測焦距**影響(焦距越長，轉一度畫面動越多)，各序列焦距不同。但其變化幅度(2.3×)比平移(85×)小一個數量級，對照組作用仍成立。
- ⚠️ **`temple_3` 的靜態像素只有 7.8%** —— `m_geo` 把 92% 畫面標成動態。這是獨立的遮罩問題，會讓它在本表與表 5/6 的所有數字都不可靠。**未查。**

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
| 8 | ~~flow_loss 探針只有 PointOdyssey~~ | ✅ **已補**（表 5.1，五個資料集）。結論：訓練集全部飽和 1.5–2.0×，只有域外 Sintel 是 4.6× |
| 9 | **`temple_3` 的 `m_geo` 把 92% 畫面標成動態**（表 7 靜態% = 7.8%） | 它在表 5/6/7 的所有數字都不可靠，且它是 ATE(14) 的兩個離群值之一。是遮罩壞了還是該序列真的幾乎全動態，未查 |
| 10 | **表 6/7 的遮罩用 GT 導出的 `m_geo`** | −17.0% 偏樂觀。可部署版本的遮罩必須來自 `gate_predictor`；換成 σ(g) 重跑是**目前唯一能把 gate 與 ATE 直接連起來的實驗**（現有 gate 因果鏈繞經 attention，f50 上 predicted −0.3%／oracle +0.7%） |
| 11 | **ep30 的 Sintel flow 殘差尾巴比 VGGT-1B 差**（表 5.2：p99 3.758 → 4.142） | 訓練改善軌跡的同時讓少數區域的局部幾何變糟，成因未查。可用 per-seq 資料定位到是哪些序列 |
| 7 | **訓練雜訊帶未正式量化** | run1 的 24 個 epoch pose_eval 在 0.1533~0.2225 間跳動，ep24 (0.1713) ≈ ep2 (0.1715)，無收斂趨勢。**雜訊帶約 ±0.02（±13%）** → 任何小於此的「改善」不可信 |
