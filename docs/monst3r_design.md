# MonST3R vs Dyn-VGGT —— loss 逐項對照

## 為什麼有這份文件

**原始意圖**：複製 MonST3R 從靜態遷移到動態的成功 —— 把它 TTO 用的 `flow_loss` 與
`temporal_smoothing_loss` 最直接地搬進 VGGT 的 finetune 訓練。

**實際做出來的東西**：兩項都只有**概念上部分一致**，計算上並不相同。而且這兩處落差都不是疏漏，
是當初為了搬進訓練而自行決定的調整：

| 項 | MonST3R 的 | 我們實作成 | 結果 |
|---|---|---|---|
| flow | 對 **RAFT 光流**的 2D 位移殘差 | 對**影像像素**的 RGB 強度殘差（`static_photo`） | **完全失敗** —— full-seq 最佳 0.1583，比自己的 warm-start 起點 0.1533 還差，且越訓越糟 |
| 平滑 | **一階**（罰運動本身） | **二階**（罰加速度）+ Δt 正規化 | **有效但不足** —— 0.1714 → 0.1343 後平台化，未超越 MonST3R 的 0.1110 |

這份文件記錄那些落差**具體差在哪**。

## 收錄範圍

只收**兩邊都有對應物**的三項。MonST3R 的 `(li+lj)` 點雲對齊骨幹與 `depth_prior` 在 Dyn-VGGT
沒有對應物 —— 前者是把 pairwise 預測拼成全域解的機器，而 VGGT 一次前向就是全域解；後者是因為
MonST3R 的深度是每像素自由變數才需要拉回初值。兩者皆不列入。

本檔只比對**目標函數本身**（公式、變數、梯度流向），不討論 TTO 與監督訓練的差異，也不放實驗數字
（那些在 [table.md](table.md)）。每一項附 `file:line`，可自行核對。

> **通則：這三項對得上名字，對不上作用。**

---

## 1. flow / photometric —— 影像觀測項

### MonST3R `flow_loss`
[optimizer.py:780-802](../reference/monst3r/dust3r/cloud_opt/optimizer.py#L780-L802)、
[smooth_L1_loss_fn:18-24](../reference/monst3r/dust3r/cloud_opt/optimizer.py#L18-L24)、
[warp_by_disp](../reference/monst3r/dust3r/utils/goem_opt.py#L196-L237)

```
ego_flow = normalize( K'R_rel K⁻¹·ũ  +  d·(K'·t_rel) ) − ũ   d = 1/depth（視差形式）

raw    = smoothL1( ego_flow ⊙ m , flow_RAFT ⊙ m , beta=1.0 )
keep   = (raw < pxl_thre) ⊙ m                                pxl_thre = 50
L_i    = Σ(raw ⊙ keep) / Σ(keep)

L_flow = L_i(i→j) + L_j(j→i)
if L_flow > flow_loss_thre:  L_flow = 0                      ← 太大就整段不用，thre = 20
```

- 幾何量：預測 pose + **預測 depth** + 預測 K，**三者皆為變數**
- 遮罩 `m` = 自算動態遮罩取反（見第 3 節）
- 第 10% iteration 之後才啟用（`flow_loss_start_epoch=0.1`）
- ⚠️ **超參要看 argparse，不是 class 簽名**。兩邊不一樣，實際生效的是 argparse 那組：

  | | class 簽名 `optimizer.py:37-38` | 實際跑的 [training.py:108-114](../reference/monst3r/dust3r/training.py#L108-L114) |
  |---|---|---|
  | `flow_loss_thre`（整段不用的門檻） | 50 | **20** |
  | `flow_loss_start_epoch` | 0.15 | **0.1** |
  | `translation_weight` | 0.1 | **1.0** |
  | `pxl_thre`、`motion_mask_thre` | 50、0.35 | 相同 |

  Sintel pose eval 走 `pose_eval.py`，吃的是 argparse。門檻 20 比 50 嚴格得多。
- ⚠️ `flow_loss_flag` 設了但全 repo 無人讀；`OccMask` 算了但從未進 loss —— 兩者皆 dead code
- ⚠️ 「整段不用」是 Python `>` 比較，而 `nan > thre` 為 False → **NaN 擋不住**（分母為 0 時可達）

### Dyn-VGGT `compute_static_photo_loss`
[loss.py:488-591](../training/loss.py#L488-L591)

```
X_t     = K_gt⁻¹·ũ · D_gt                                    ← GT 深度反投影（相機 t 座標系）
R_rel   = R_{t+1} R_tᵀ ,  t_rel = T_{t+1} − R_rel·T_t        ← 皆由預測 pose_enc 組成
u'      = π( R_rel·X_t + t_rel )                             ← 預測相對位姿 warp
L       = Huber( I_{t+1}(u') − I_t(u) , δ=0.1 )              ← RGB 強度差
```

程式碼分成 cam_t → world → cam_{t+1} 兩步走（[loss.py:555-556](../training/loss.py#L555-L556)），
和上式等價。**真正影響 loss 的只有相對位姿** —— 整段軌跡一起平移旋轉，這個 loss 不會變。

- 幾何量：**只有 pose 是預測的**，depth / K / 影像皆為常數
- 遮罩：`point_masks ∧ (motion_mask < 0.5) ∧ in_bounds ∧ 目標落點亦為靜態`
- 單向 t→t+1，逐 pair 累加後除 `n_pairs`

### 差異

| | MonST3R | Dyn-VGGT |
|---|---|---|
| 殘差量 | **2D 位移向量** | **RGB 強度** |
| 觀測來源 | RAFT 光流 | 原始影像像素 |
| 梯度流向 | pose + depth + focal | **只有 pose** |
| 無紋理區 | 不受影響（比的是 correspondence） | **梯度 = 0** |
| 亮度變化 | 免疫 | 直接汙染 |
| 方向 | 雙向 | 單向 |
| 離群處理 | 逐像素丟棄 + 整段不用 | 無 |

**兩者名義上對應，實際上沒有共同機制。** MonST3R 用 RAFT 把 photometric 換成 correspondence，
正是為了規避無紋理／亮度／local-minimum 三個坑，而 `static_photo` 把三個都踩回去。

---

## 2. 平滑先驗

### MonST3R `relative_pose_loss`
[optimizer.py:1014-1027](../reference/monst3r/dust3r/cloud_opt/optimizer.py#L1014-L1027)

```
RT_rel = RT_t⁻¹ · RT_{t+1}
L_temp = Σ_t [ ‖R_rel − I‖_F  +  w_t·‖t_rel‖₂ ]              w_t = translation_weight = 1.0
總 loss 內再乘 temporal_smoothing_weight = 0.01
```

**一階** —— 懲罰相鄰幀的**運動本身**，隱含先驗是「相機幾乎不動」，等速運動亦受罰。

### Dyn-VGGT `compute_camera_smooth_loss`
[loss.py:594-681](../training/loss.py#L594-L681)

```
v    = (T_{t+1} − T_t) / Δt                     Δt 由 batch["ids"] 給（真實幀號差）
a_T  = v_{t+1} − v_t
q̂    = 四元數半球符號修正後的序列
a_R  = 同樣的二階差分

L    = Σ_stages γ^(n−1−s) [ mean|a_T|·w_T + mean|a_R|·w_R ] / n
```

**二階** —— 懲罰**加速度**，等速運動免費，只罰速度突變。

### 差異

| | MonST3R | Dyn-VGGT |
|---|---|---|
| 階數 | 一階（罰運動） | **二階**（罰加速度） |
| 隱含先驗 | 相機不動 | 相機等速 |
| 表示空間 | SE(3)，Frobenius | pose-encoding 逐分量 L1 + 四元數符號修正 |
| 取樣假設 | 等間隔 | **Δt 正規化**，排除 Δt=0 的重複幀 |
| reduction | `.sum()`（值隨序列長度成長） | mask 加權平均 + 多 stage γ 加權 |

**Δt 正規化是 Dyn-VGGT 這側必要的加項**：訓練 clip 由 `get_nearby_ids(expand_ratio=2.0)`
取樣，幀距不規則（Δt 分佈 0~12+，其中 13% 為重複幀），不能像 MonST3R 那樣假設等間隔。

---

## 3. 動態遮罩

### MonST3R —— 推導出來的副產品，**沒有 loss**
[get_motion_mask_from_pairs:294-366](../reference/monst3r/dust3r/cloud_opt/optimizer.py#L294-L366)

```
err  = ‖ego_flow(粗估 pose/depth) − flow_RAFT‖₂              (H,W)
err  ← (err − min) / (max − min)                             每個 pair 各自 min-max 正規化
mask = mean_over_pairs(err) > 0.35
mask = mask ∨ SAM2_mask                                      ← 預設開啟，取聯集
```

- 粗估來自 `PairViewer`，**不是** GT，也不是優化中的變數
- `requires_grad_(False)`，是常數 buffer，**不參與梯度**
- 唯一用途：當 flow loss 的像素權重
- ⚠️ **SAM2 預設是開的**（`sam2_mask_refine=True`），而且做的是取聯集、不是修正：
  `dynamic_masks[i] |= sam2_dynamic_masks[i]`
  （[optimizer.py:443](../reference/monst3r/dust3r/cloud_opt/optimizer.py#L443)）。
  也就是說預設跑出來的遮罩 = **幾何殘差 ∪ SAM2 分割**。說 MonST3R 的遮罩「只是幾何副產品」時要留意這點。
- ⚠️ min-max 正規化讓**每個 pair 一定有像素等於 1.0**。單 pair 的幀因此必定被標出動態區；
  多 pair 平均後才有機會全部低於 0.35。全靜態場景還是容易憑空生出動態區。

### Dyn-VGGT `compute_gate_loss` —— 可學模組，且回饋進架構
[loss.py:702-757](../training/loss.py#L702-L757)

```
m*_patch = adaptive_avg_pool2d( motion_mask , patch grid )
L_gate   = BCE_with_logits( g , m*_patch )

# C-1 選項：hard label + ignore band
keep     = (m*_patch ≤ lo) ∨ (m*_patch ≥ hi)
target   = (m*_patch ≥ hi)
L_gate   = Σ(BCE ⊙ keep) / Σ(keep)
```

- `gate_logits` **不 detach** → 梯度回流到 `GatePredictor`
- σ(g) 同時送進 aggregator：對 camera/register query row 的 patch key 加 `−softplus(g)` bias
- 標籤 `m*` 來自離線預算的 `dynmask_inst`（instance × GT scene-flow）

### 差異

| | MonST3R | Dyn-VGGT |
|---|---|---|
| 來源 | 從幾何殘差推導 | **網路預測** |
| 有 loss | **沒有** | `BCE`，可學 |
| 有梯度 | 沒有 | 有 |
| 用途 | 只當 flow loss 的像素權重 | **進 attention bias（改變前向計算）** + 免費 dynamic mask |
| 正規化 | per-pair min-max（相對閾值） | soft pooled label（絕對） |

**這是方法論上最大的分歧**：MonST3R 的遮罩是流程的副產品，Dyn-VGGT 的是一個被監督的模組，
而且會回饋進網路架構 —— 這也是 v3 的核心主張所在。

---

## 相關

- 實驗數字與探針結果：[table.md](table.md) 表 5
- v3 方法本體：[method.md](method.md)
