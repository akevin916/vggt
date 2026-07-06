# Dyn-VGGT v3：運動門控相機聚合 —— 在源頭解耦動態場景的相機 Pose（方法）

> Motion-Gated Camera Aggregation for Robust Pose in Dynamic Scenes
>
> 本文件是 [dyn_vggt_method_v2.md](dyn_vggt_method_v2.md) 診斷後的**架構重設計**。
> v2 確認 v1 的雙場 `X = X^can + m·Δ` 存在**雙線性不可辨識性**與**動態區偷懶捷徑**，且學習式 mask **跨域崩塌**（PO AUC 0.88 → Sintel 0.45）。
> v3 **放棄一切幾何重表示**，把火力集中在 VGGT 唯一被實測落後的指標——**動態場景的相機 pose**。

---

## 0. TL;DR

> VGGT 的深度在動態場景已領先，唯一輸給 MonST3R 的是**相機 pose**。根因（v1 §1.3）是 pose 與物體運動在 aggregator 的 global attention 裡**耦合**：camera token 把動態 patch 的運動「平均」進了 pose。
>
> v2 的現況是「motion 只當 **loss 端**的 gate」——這**改不了已經算出來的 pose**。v3 的核心：把動態屏蔽從 **loss 端搬到 attention 端**，讓 camera token 在前向聚合時就**結構性地只吸收靜態 patch**。門控信號用 **domain-invariant 的幾何殘差**監督與輸入，根治跨域崩塌。幾何頭（depth/point）原樣不動，深度品質完全保留。

---

## 1. 問題定位

### 1.1 pose 在哪裡誕生（已從 code 確認）

```
patch tokens [B,S,P,C]   (靜態 + 動態 混在一起)
      │
      │  ❶ aggregator global attention (24 blocks)：
      │     camera token 的 query attend 所有 S×P patch key
      ▼
camera token (idx 0)      ← ★ 動態運動在此被平均進 pose token ★
      │
      │  ❷ camera_head trunk (4 blocks)，只看 camera token
      ▼
   pose_enc [B,S,9]
```

- `camera_head.py:89` `pose_tokens = tokens[:, :, 0]`：pose **完全**來自 camera token。
- camera token 的內容在 `aggregator.py` 的 `_process_global_attention` 形成，與所有 patch（含動態）做 attention。
- camera head trunk 只後處理這個**已被污染**的 token，無法事後解混。

→ **§1.3 的 pose-motion coupling 有唯一架構位置：global attention 裡 camera-query 對動態 patch-key 的聚合。** 在此處切斷，pose 結構上就只是靜態幾何的函數。

### 1.2 v2 現況為何沒有 pose 增益

`vggt.py` 已移除雙線性組裝（正確），但 motion `m`「**只當 loss 端的 gate**」：
- `valid_frame` 靜態屏蔽只改變**哪些幀參與監督**，不改變**pose 怎麼算**。
- 推理時 camera token 仍由動態 patch 聚合而成 → 前向污染原封不動 → 無 pose 增益。

### 1.3 三條必須同時躲開的坑（v2 診斷）

| 坑 | v3 如何避免 |
|---|---|
| 雙線性不可辨識（`m·Δ` 乘積只監督和）| **沒有 `Δ`、沒有乘積**；門控是 attention bias，不是幾何組裝 |
| 動態區 `X^can` 無監督 → 偷懶捷徑 | **不新增任何幾何輸出**；depth/point 原樣全監督，無自由隱變量 |
| 學習式 mask 跨域崩塌（0.88→0.45）| 門控信號用 **domain-invariant 幾何殘差**作**目標**與**輸入**（§4），非外觀；並有 test-time 兜底（§7）|

---

## 2. 方法總覽

```
            輸入視頻 {I_t}  (S frames)
                  │
       ┌──────────▼───────────┐
       │  DINOv2 Patch Embed   │  (沿用)
       └──────────┬───────────┘
                  │
       ┌──────────▼─────────────────────────┐
       │  Aggregator (frame / [temporal] / global)            │
       │                                                       │
       │   block 0..k  ──► Gate Predictor g  (貢獻①, §3)       │
       │                        │ (detach → bias)              │
       │   block k+1..23 global attention 套 camera-query 門控 │  (貢獻② §2-3 在源頭清 pose)
       └──────────┬─────────────────────────┘
           ┌──────┴───────┬──────────┐
           ▼              ▼          ▼
       Camera Head    Depth Head   Point Head
       (清乾淨)        (原樣)        (原樣)
           │              │           │
           ▼              ▼           ▼
       pose_enc        depth       world_points        σ(g) = 動態 mask (免費副產物)
           │
           ▼ (可選) test-time pose polish (貢獻③ §7)
        解耦相機軌跡
```

| # | 貢獻 | 解決的根因 | 核心機制 |
|---|---|---|---|
| ① | **中段 Gate Predictor** | 雞生蛋（門控需 m、m 來自最終特徵）| 在 ~1/3 深度用輕量 MLP 出 patch 動態 logit `g` |
| ② | **運動門控相機聚合** | §1.1 pose-motion coupling | global attention 的 camera/register query 對動態 patch key 加 `−softplus(g)` bias |
| ③ | **幾何接地的門控監督** | mask 跨域崩塌 | `g` 的**目標**用 GT 剛體光流殘差、**輸入**含預測殘差通道 → domain-invariant |
| ④ | **pose-only test-time polish** | 前饋難例兜底 | 靜態區多視圖重投影、只優化 pose、20–50 迭代 |

**範圍刻意鎖定「純 pose」**：不輸出 scene flow、不改幾何表示。depth/point 頭原樣全監督，深度領先保留。

---

## 3. 貢獻①：中段 Gate Predictor —— `g` 怎麼產生

門控要在 aggregation **進行中**用到動態信號，但 motion 通常要讀**最終**特徵才得到 → 雞生蛋。解法：在 aggregator 中段插一個輕量預測器，**邊聚合邊出門控信號**。

```
            中段 aggregator 輸出 (約 block k=8 後)
            patch tokens  [B, S, P_patch, C]        # P_patch = (H/14)·(W/14)
                    │
        ┌───────────┴────────────┐
        │  (跨域關鍵) 幾何殘差通道  │  ‖f^gt − f^cam(預測 pose, 預測 depth)‖ → [B,S,P_patch,1]
        │  concat 進特徵           │
        └───────────┬────────────┘
                    ▼
            LayerNorm → Linear(C→C/4) → GELU → Linear(C/4→1)   # 末層 zero-init
                    ▼
              g  [B, S, P_patch]      # 每 patch 一個動態 logit
```

`g` 的三個去處：
1. **門控（detach）**：`bias = −softplus(g.detach())` → §4 的 attention bias。detach 確保 pose 梯度不反灌門控、避免「為 pose 好看而亂標動態」。
2. **監督（不 detach）**：`L_gate = BCE(σ(g), m*_patch)`，patch 解析度（pixel-level 硬 mask average-pool 到 patch 格 → `m*_patch`，見 §5.3）。
3. **輸出**：`σ(g)` 即 `motion_prob`（動態 mask 副產物）；需 pixel-level 圖時再接輕量 DPT upsample（同樣 `m*_patch` 監督）。

設計選擇：
- **解析度天然對齊**：`g` 在 patch 解析度，attention 的 key 本來就是 patch token → bias 直接套，零插值。
- **放中段 (~1/3)**：太早特徵未成熟；太晚則後面沒幾個 global block 可被清洗。1/3 是「特徵夠成熟」與「後段仍有 2/3 block 可門控」的平衡。
- **warm-start**：末層 zero-init → 訓練第 0 步 `g≡0` → `bias≡0` → 前向 byte-for-byte 等於 pretrained VGGT，不擾動既有幾何。`g` 隨訓練漸長，門控漸進通電。

---

## 4. 貢獻②：運動門控相機聚合（核心）

### 4.1 機制

attention：`A = softmax(QKᵀ/√d + bias)`，`out_i = Σ_j A[i,j]·V_j`。
**只**在 camera + register token 的 query 列，對 patch token 的 key 加負偏置：

```
bias[special_query, patch_key_j] = −softplus(g_j)      # g_j = patch j 的動態 logit (detach)
bias[其他所有位置]                = 0                    # patch↔patch 完全不動
```

| patch j | g_j | bias | softmax 後權重 |
|---|---|---|---|
| 靜態 | →−∞ | →0 | 維持原樣 |
| 動態 | →+∞ | →−∞ | →0（排除）|

- camera token **結構上只聚合靜態 patch** → pose 不再被運動污染（訓練+推理皆然）。
- **patch↔patch attention 一字不改** → depth/point 頭輸入特徵不變，**深度品質完全保留**。
- 後段每個 global block 都套此 bias，camera token 在後 2/3 聚合中**漸進地**只看靜態。

### 4.2 全動態幀的欠定問題（重要）

若某幀幾乎全動態，門控排除絕大多數 patch → camera token 無料可聚合 → pose **欠定**。這是「資訊不足」不是「監督不足」，對策：
1. **soft bias（`−softplus` 而非 `−∞`）**：全動態時仍保留微弱全局信號。
2. **top-k floor**：每幀至少保留 `g` 最小的 k 個 patch 參與聚合，保證最低限度輸入。
3. **temporal attention 借鄰幀**：透過時序塊從較靜態的相鄰幀取幾何上下文（保留 temporal aggregator 的價值之一）。
4. test-time polish 對此類幀自然降權。

### 4.3 實作關鍵：attention 拆兩條 path（顯存命脈）

把整個 `[N,N]` 浮點 mask 丟給 `F.scaled_dot_product_attention` 會讓 flash kernel fallback 到 memory 重的 math backend → 顯存暴增。正解：
- **patch↔patch**（佔 (S·P)² 大頭）：維持原 flash / memory-efficient attention，**不帶 bias**。
- **camera/register query → 所有 key**（只有 ~5 個 query）：單獨小 op，帶 bias；即使非 flash 也可忽略。

→ 架構②本身顯存增加 ≈ 0。

---

## 5. 貢獻③：幾何接地的門控監督（跨域關鍵）

**換監督目標不足以跨域**：網路仍可偷學 PO 外觀（PO AUC 高、Sintel 崩）。真正讓門控跨域的是**「目標」與「輸入」都接地到 domain-invariant 的幾何量**。

### 5.1 監督目標 `m*_geo`（理論公式，穩定、用 GT 算）

> **命名說明**：本文件用下標區分 `m*` 的不同變體 —— `m*_geo`（本節，理論通式）、`m*_raft`（§5.3a，RAFT 硬閾值）、`m*_inst`（§5.3b，instance×3D-scene-flow 硬閾值，**PO 訓練標籤實際採用**）、`m*_patch`（任一 pixel-level 硬 mask average-pool 到 patch 解析度後、真正餵進 `L_gate` 的軟機率，見 §6）。目前程式碼（`training/loss.py`）直接從 `m*_inst` pooling 出 `m*_patch`，**尚未實作**本節的 sigmoid 軟化式。

```
f^cam    = π( P_{t+1} P_t⁻¹ π⁻¹(u, D_t) ) − u          # 「假設靜態」的相機誘導光流
m*_geo   = σ( α_m·( ‖f^gt − f^cam‖ − β_m ) )            # 殘差大 = 動態；sigmoid 對「連續殘差」做 soft threshold
```
- **`P`、`D` 全用 GT**（訓練時皆有）→ `m*_geo` 是**固定穩定目標**，不依賴模型當下預測 → 無雞生蛋。
- `f^gt`：資料集內建光流或離線 RAFT。
- **注意：不可直接用 PointOdyssey 的原生 mask 當 `m*`**（它是外觀 mask，非運動 mask，見 §5.3）。`m*` 的實際產生方式見 §5.3。

```
L_gate = BCE( σ(g), m*_patch )      # patch 解析度；m*_patch 見 §5.3/§6
```

### 5.2 跨域的真正槓桿：幾何殘差**輸入通道**

gate predictor 的輸入除中段特徵外，concat 一條 **`‖f^gt − f^cam‖`**，其中 `f^cam` 用**模型自己預測的 pose+depth** 計算（**輸入用預測值；目標 §5.1 用 GT 值**）。
- 這個量在 PO 與 Sintel 是同一物理量（光流不一致程度），與外觀無關 → PO 學到「殘差大→動態」直接遷移到 Sintel。
- 輕微雞生蛋（要 pose 才能算殘差來清 pose）由 VGGT 既有**迭代 refine**化解：首輪粗估、後續用上一輪結果。

> **取捨點**：若覺得幾何殘差輸入通道工程量太重，可退為「只用 §5.1 GT 目標 + 大量混合資料」，跨域改善但不徹底，再靠 §7 test-time polish 兜底。

### 5.3 動態標籤 `m*` 的產生（實作 + 資料現實）

**問題：PointOdyssey 原生 mask 不是動態 mask。** `masks/` 是**逐 instance 分割**，dataset 用「非黑=動態」判定 → 室內場景連**牆、天花板、地板**都是 instance → 幾乎整片被標「動態」；且**靜止的前景（idle agent）也照標**。這是**外觀 mask**，會教門控學「室內/前景外觀→動態」→ 換到 Sintel 室內就全幅誤觸發（跨域崩塌的主因，非純 domain gap）。故必須改用**運動定義**的標籤。實作兩種：

**(a) RAFT 光流殘差 → `m*_raft`，存於 `dynmask_raft/`（domain-invariant、無需 GT track，可用於測試端/Sintel）**
`m*_raft = 1[‖f^gt(RAFT, t→t+Δ) − f^cam(GT depth,pose)‖ > τ_px]`。優點：純幾何、跨域一致（與 Sintel eval 同式）。**缺點（已實測，故不用於訓練標籤）**：單一固定 Δ 兩難——慢速室內需大 Δ 才累積到門檻，但**快相機**大 Δ 會讓 baseline×深度誤差把**靜態背景**殘差灌爆（整片誤標）；自適應 Δ 也不徹底。

**(b) instance × GT scene-flow → `m*_inst`，存於 `dynmask_inst/`（PO 訓練標籤實際採用）**
點「動」定義：`‖trajs_3d[t+IG] − trajs_3d[t]‖ > ITHR`（世界公尺）。將每個 instance **顏色切連通元件(CC)**，某 CC 內「動的點比例 > IFRAC」且點數 ≥ MINTRK → **填滿整塊 CC**，跳過面積 < MINCC 或 > MAXCC（背景/過度合併）者，最後 close(3×3)+補洞實心化。
- **為何最穩**：靜態世界點位移**恆等於 0**（任意 gap）→ 無 RAFT 的靜態背景灌爆、無 gap 快慢兩難；CC + MAXCC 解決「黑色/大片背景同色連成一塊被整片填」；不跳過黑色（室內人常被算圖成黑色）→ 用運動判斷救回。
- 定值：`IG=5, ITHR=0.01, IFRAC=0.1, MINTRK=8, MINCC=0.001, MAXCC=0.4`。腳本 `training/data/precompute_po_instance_dynmask.py`；dataset `dynamic_source="instance"` 載入。

**分工**：**有 GT track 的訓練集（PO）→ (b) `m*_inst`**（最可靠）；**無 GT track 的測試/Sintel → (a) `m*_raft`**（或用 instance 對 RAFT 做 CC 聚合的 raft-snap 變體）。兩者語義一致（動態=世界真的在動），只是估計來源不同。

**最後一步（兩者共用）**：不論 `m*_inst` 還是 `m*_raft`，兩者都是 **pixel 解析度的硬 0/1 mask**。要跟 patch 解析度的 `g` 算 BCE 前，必須先 average-pool 到 patch 格，得到本文件稱為 **`m*_patch`** 的 soft 機率（`training/loss.py` 裡的 `m_star_patch` 變數）：
```
m*_patch = adaptive_avg_pool2d( m*_inst (或 m*_raft), patch_grid )     # 硬 0/1 → 該 patch 內動態像素比例 ∈[0,1]
```

---

## 6. 損失函數（純 pose 版，比 v1 大幅精簡）

丟掉雙場、scene-flow、reproj 閉環、tsmooth，只剩：

```
L = L_cam + L_depth + L_point + λ_g · L_gate
```

- **`L_cam`（不變）**：逐分量 L1（T/q/FoV），跨 refine stage 以 `γ^{n−i−1}` 加權。
  > 注意：pose loss 直接拿 `pose_enc` 對 GT `pose_enc` 算，**不經像素** → 動態像素**不會污染 pose 的 loss**。pose 提升來自**架構②的前向門控**，非 loss 屏蔽。
- **`L_depth` / `L_point`（不變）**：confidence-weighted + 多尺度梯度，原樣全監督。
- **`L_gate`**：`BCE(σ(g), m*_patch)`，§5.1/§5.3 的 soft 標籤。`λ_g` 起始 1.0。

**`valid_frame` 建議維持原本寬鬆的「有效點數 > 100」，不要改成嚴格的「靜態點數 > 100」**：後者會在多動態場景造成幀數不足，而 pose 監督是直接的、得不到對等好處。

---

## 7. 貢獻④：pose-only test-time polish（可選兜底）

前饋門控若於難例仍不夠乾淨，加便宜的 test-time 精修：用前饋 `σ(g)` 排除動態像素，對**靜態區**做多視圖重投影最小化、**只優化 pose**、20–50 迭代。domain-invariant、不動深度。提供「純前饋（0 迭代）」與「精修」兩檔。

---

## 8. 訓練策略（課程 + 硬體適配）

VGGT 是強預訓練 baseline，**絕不從頭訓**。三階段漸進解凍。

### 8.1 三階段

| 階段 | 凍結 | 解凍訓練 | 開的 loss | 目的 |
|---|---|---|---|---|
| **S0** | aggregator 全部 + 原三頭 | 只 gate predictor | `L_gate` | 養門控信號；門控因 zero-init 仍 ≈no-op |
| **S1** | DINOv2 patch embed | gate predictor + 後段 global blocks + camera head | `L_gate` + `L_cam` | 門控通電，camera 學會「只看靜態」 |
| **S2** | 無（全網，lr 1e-5）| 全部 | 全開 + 混 ~30% 靜態防遺忘 | 端到端精修 |

### 8.2 硬體適配（32GB 單卡上限 4–6 幀；可用雙 24GB）

**(a) 純 pose scope 的最大紅利：關掉用不到的 dense 頭**

| Head | 訓練需要 | 理由 |
|---|---|---|
| camera | ✅ | 主角 |
| depth | ✅ | 算 `f^cam` 殘差（§5.2 輸入通道）需預測深度 |
| point / track / flow / motion(DPT) | ❌ 關掉 | pose 不需；`g` 已當 mask |

關掉 point+track 即釋出一大塊顯存，4–6 幀上限可往上推。

**(b) S0 快取凍結特徵**：S0 aggregator 全凍 → 不 backprop 穿過 aggregator → 不必存其 activation。把中段特徵**離線預算存盤**，S0 只訓 gate MLP → 顯存近零，可開長 sequence / 大 batch（24GB 也很舒服）。

**(c) 雙 24GB 配置**
- **DDP（首選）**：兩卡各一份模型、各自 sequence，吞吐 ×2。前提：先用 (a)(b) 把單卡 footprint 壓進 24GB（必要時 S→3–4 或解析度→364）。
- **FSDP**：S2 全網解凍時若 optimizer state 成瓶頸，分片參數+狀態降 per-GPU footprint。**切不了 activation**，故只適合 S2，不解 attention 峰值。

**(d) 機動槓桿（按性價比）**
1. 降解析度 518→364（P ~1369→~676，attention 記憶體約 ¼，幀數翻倍）；可低解析訓 gate、最後 518 微調。
2. gradient checkpointing（code 已開）、bf16 autocast（已開）維持。
3. 時序加大 stride 抽幀：更大 baseline、對 pose 更有利、幀數需求更低。
4. gradient accumulation 補足等效 batch。

---

## 9. 與 v1/v2 的關係與評測

- **保留**：temporal aggregator（貢獻①, v1）—— 正交、未被診斷實證有問題，且 §4.2 借鄰幀需要它。
- **丟棄**：雙場 `X^can + m·Δ`、scene-flow 頭、`L_reproj` / `L_tsmooth` / `L_flow`、4D 全局對齊（縮為 pose-only polish）。
- **新增**：中段 gate predictor、運動門控 attention、幾何接地監督。

**評測聚焦 pose**：Sintel / PointOdyssey 的 ATE / RPE（相機軌跡），對標 MonST3R 與原版 VGGT；輔以動態 mask 的 AUC（跨域 PO→Sintel 是關鍵指標）與 depth 不退化的對照。

---

## 一句話總結 novelty

> Dyn-VGGT v3 把「動態屏蔽」從 VGGT 的 **loss 端搬到 attention 端**：以中段輕量 gate 在 global attention 裡讓 camera token **結構性地只聚合靜態 patch**，在源頭解除 pose-motion 耦合；門控信號以 **domain-invariant 的幾何殘差**作目標與輸入，根治學習式 mask 的跨域崩塌——在不動幾何頭、保留深度領先的前提下，前饋地拿回動態場景的相機 pose。
