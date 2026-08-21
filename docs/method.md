# Dyn-VGGT v3：動態場景的前饋式相機 Pose（方法）

> Feed-Forward Camera Pose in Dynamic Scenes —— 從**前向**、**監督**、**時序**三個軸解耦 pose 與物體運動
>
> 本文件是 v2 診斷（[archive/checkpoints.md §2.2](archive/checkpoints.md)）後的**架構重設計**。
> v2 確認 v1 的雙場 `X = X^can + m·Δ` 存在**雙線性不可辨識性**與**動態區偷懶捷徑**，且學習式 mask **跨域崩塌**（PO AUC 0.88 → Sintel 0.45）。
> v3 **放棄一切幾何重表示**，把火力集中在 VGGT 唯一被實測落後的指標——**動態場景的相機 pose**。

---

## 0. TL;DR

> VGGT 的深度在動態場景已領先，唯一輸給 MonST3R 的是**相機 pose**（Sintel ATE 0.171 vs 0.108）。根因是 pose 與物體運動在 aggregator 的 global attention 裡**耦合**：camera token 把動態 patch 的運動「平均」進了 pose。
>
> v3 用**三個正交的貢獻**打同一個耦合，各自對應一條路徑：
>
> | | 貢獻 | 軸 | 一句話 |
> |---|---|---|---|
> | **①** | **Gate Mechanism** | **前向** | 聚合 camera token 時，用 attention bias 結構性地排除不符相機運動的動態 patch |
> | **②** | **Motion Loss** | **監督** | 用物理幾何約束（剛性 warp 只在靜態區成立）設計動態場景專屬的監督訊號 |
> | **③** | **Temporal Attention** | **時序** | 補上 VGGT 缺的純時間軸 attention，利用影片的時序相關性 |
>
> **範圍刻意鎖定「純 pose」且純前饋**：不輸出 scene flow、不改幾何表示、**不做 test-time 後處理**（對標 MonST3R 的 TTO 是刻意的取捨）。depth/point 頭原樣全監督，深度領先保留。

---

## 進度與下一步（最後更新 2026-07-27）

> 這是全文唯一的「現在信什麼」來源。下面各章節（§3.6 / §4.3 / §5.4 / §8）只放診斷細節與逐序列數字，
> 結論一律回來看這裡，不在該章重複下結論。完整數字見 [table.md](table.md)（含世代地圖、Δ% 基準、評測協定缺口）。

### 時間軸

| 日期 | 貢獻 | 動作 | 結果 | 下一步 |
|---|---|---|---|---|
| 2026-07-27 | ③ Temporal + Camera-Smooth | clean 世代重跑（`inst_gts`，同預算 warm from run1），**仍在訓練** | **目前 clean 家族 pose 最好**：`last.pt`（≈ep30+）ATE **0.1253**、ATE(12) **0.0470**、RPE-r **0.3867**，三項全表最低（除 MonST3R 外）；且單調下降未收斂（ep20 0.180 → ep30 0.134 → last 0.125）。⚠️ windowed `best.pt`（ep17）ATE 0.1651，比 `last` 差 24% —— **這條線只能用 `last.pt`，不可用 `best.pt`** | 等訓練收斂、用最終 ckpt 定論；**因子仍混淆**（temporal 解凍 + `camera_smooth` loss 疊在一起）——必須拆 temporal-only / smooth-only 才能歸因到哪一半在起作用 |
| 2026-07-24 | ② Motion (Static-Photo) | clean 世代重跑（`inst_g_photo`，同預算 warm from run1） | **負結果**：full-seq 最佳 ep10 = 0.1583，**比自己的 warm-start 起點（0.1533）還差**，且 ep10→ep20 單調惡化 | 此路線**暫緩**，不建議疊進全開組合；若要救回需重新檢討 loss 設計（§4.2 的 GT-depth 假設或權重），非本次重跑範圍 |
| 2026-07-10 | ①+②+③ 全開 | `inst_gtsp_buginit` | **lineage 不乾淨**（warm-start 自 buggy-init 而非 run1）→ ATE 0.1743，**比 base 還差**，已確認棄用 | 真正的「§7.3 run4」（乾淨 lineage、從 run1 warm-start 的全開組合）**尚未跑過** —— 待 ③ 定論後才值得跑，否則又要再棄一次 |
| 2026-07-09 | ① Gate | 修正 frame-interleaved attention bug（commit `e064087`），使 2026-07-06 的「連 oracle 都救不回」舊結論作廢 | oracle 在短序列有效（f12 −15.0%、f32 −11.0%），**完整序列歸零**（f50 +0.7%）；`predicted` 在所有長度都 ≈ `off`（−0.3~−1.5%）——gate 學到排序（AUC 0.883）但沒兌現；欠自信已確認（`signed_err +0.095`），但 C-1（hard label）證偽「BCE boundary floor 是病因」，砍 floor 只降 signed_err 6%、pose 無變化 | **主瓶頸**：`predicted` 不兌現的病因未知。下一假說是 class imbalance（`dyn_frac=0.202`，正例只有負例 1/4）→ 需試 `pos_weight≈4` re-weight，**未驗證** |

### 現在最值得做的三件事（依優先序）

1. **等 ③ smooth_temporal 收斂並拆解**——目前唯一的正結果還混著兩個因子（temporal 結構 vs camera_smooth loss），不拆就不知道賣點在哪。
2. **驗證 gate 欠自信的 class-imbalance 假說**（`pos_weight` re-weight）——這是①的唯一主瓶頸，且 BCE-floor 假說已被 C-1 排除，需要下一個候選解釋。
3. **確認 ③ 有效後才重跑乾淨 lineage 的全開組合**（真正的 run4）——現有的全開數字是 lineage 汙染的負結果，不能當「全開沒用」的證據。

> ② Motion 路線暫時擱置（clean 重跑是負結果），不在上述優先序中。

> ⚠️ 讀以上任何 Δ 之前，先看 §8 的評測協定警告：單次比較 2σ 雜訊門檻 **16.9%**，且沒有 held-out 資料——上面 ③ 的 −18%（相對 warm-start 起點）與 −27%（相對 base）尚未過雜訊帶檢驗，屬於「很有希望但還不能拍板」的狀態。

---

## 1. 問題定位

### 1.1 pose 在哪裡誕生（已從 code 確認）

**每一幀的 token 佈局**（`Aggregator.forward`，[aggregator.py:317-321](../vggt/models/aggregator.py#L317-L321)）：

```
每張圖 → DINOv2 patch embed → patch tokens [P, C]
                                    │
        tokens = cat([ camera_token(1) , register_token(4) , patch_tokens(P) ], dim=1)
                        └─ idx 0 ─┘   └── idx 1..4 ──┘   └── idx 5.. ──┘
                                    │
                        patch_start_idx = 1 + num_register_tokens = 5
```

- **camera token（1 個/幀）**：`self.camera_token = nn.Parameter(torch.randn(1, 2, 1, C))`
- **register token（4 個/幀）**：`self.register_token = nn.Parameter(torch.randn(1, 2, 4, C))`
- 兩者皆 `std=1e-6` 初始化。

> **`dim=1` 的那個 `2` 是「第一幀特殊」**（`slice_expand_and_flatten`，[aggregator.py:598](../vggt/models/aggregator.py#L598)）：index 0 的 token 只給 **frame 0**，index 1 的給**其餘 S−1 幀**。因為 VGGT 的 pose 是**相對於第一幀**的，frame 0 需要一個可區分的 query token。

**三種 token 的下場**：

| token | idx | 誰讀它 | 下場 |
|---|---|---|---|
| **camera** | 0 | `camera_head`：`pose_tokens = tokens[:, :, 0]`（[camera_head.py:89](../vggt/heads/camera_head.py#L89)）→ trunk 迭代 refine → `pose_enc` | **→ pose** |
| **register** | 1..4 | **沒有任何 head 讀** | **attention 跑完就丟** |
| **patch** | 5.. | depth/point/track head 用 `patch_start_idx` 切出來 | → 幾何 |

register token 是 DINOv2 那套的輔助設計（吸收全域資訊、穩定 attention），純粹是 attention 過程的 scratch space。

**耦合就發生在這裡。** 在 **global attention** 裡，所有幀的所有 token 攤平在一起做 attention —— 每一幀的 camera token（idx 0）會 attend 到**所有幀的所有 patch token**，其中**包含動態物體的 patch**：

```
out[camera_token] = Σ_j softmax(QKᵀ/√d)[camera, j] · V_j
                    └────────── j 跑遍所有 patch，靜態與動態一視同仁 ──────────┘
```

camera token 因此把動態物的運動「平均」進了自己的表示 → 送進 camera head → **pose 被污染**。

**這解釋了為什麼修正 §3.4 的 frame-interleaved bug 這麼關鍵**：global attention 的排列是 `[frame0(camera,reg×4,patch×P) | frame1(...) | …]` —— special token **散在每幀開頭**。舊版誤以為它們集中在最前面，導致只有 frame 0 的 camera token 被門控。

**也解釋了為什麼 patch↔patch attention 一字不改（§3.2）就能保住深度**：depth/point head 只讀 patch token，而 patch 之間的 attention 完全沒被動過。

### 1.2 為什麼 loss 端的 mask 改不了 pose

v2 的現況是「motion 只當 **loss 端**的 gate」——這**改不了已經算出來的 pose**。pose 在 aggregator 的 global attention 裡就已經被污染；等到算 loss 時再屏蔽動態區，只是不去罰它，並沒有把污染從前向拿掉。

**這正是貢獻① 的動機**：把動態屏蔽從 loss 端搬到 **attention 端**。

### 1.3 三個根因 → 三個貢獻

| 根因 | 症狀 | 對應貢獻 |
|---|---|---|
| **前向耦合**：camera query 無差別聚合所有 patch key | pose 被動態物的運動平均進去 | **① Gate**：在 attention 加負 bias |
| **監督不足**：`L_cam` 只比對 GT pose_enc，不經像素 | pose 沒有獨立於 GT 重參數化的訊號 | **② Motion Loss**：用剛性 warp 的幾何約束造獨立監督 |
| **時序結構未用**：VGGT 只有 frame / global attention | 逐幀獨立估計 → 軌跡跳動 | **③ Temporal**：補純時間軸 attention |

三者正交：① 改前向、② 加監督、③ 補結構。可獨立開關、獨立消融（§7.3）。

---

## 2. 方法總覽

```
            輸入視頻 {I_t}  (S frames)
                  │
       ┌──────────▼───────────┐
       │  DINOv2 Patch Embed   │  (沿用，永遠凍)
       └──────────┬───────────┘
                  │
       ┌──────────▼──────────────────────────────────────┐
       │  Aggregator                                      │
       │    aa_order = [frame, temporal(③), global]       │
       │                                                  │
       │   block 0..k  ──► Gate Predictor g   (① §3.1)    │
       │                        │ (detach → bias)         │
       │   block k+1..23 global attention 套              │
       │                 camera-query 門控     (① §3.2)   │
       └──────────┬──────────────────────────────────────┘
           ┌──────┴───────┬──────────┐
           ▼              ▼          ▼
       Camera Head    Depth Head   Point Head
       (清乾淨)        (原樣)       (關掉)
           │              │
           ▼              ▼
       pose_enc        depth              σ(g) = 動態 mask (免費副產物)
           │
           ├── L_cam            (VGGT 原生)
           ├── L_gate           (① §3.5)
           ├── L_static_photo   (② §4.2)
           └── L_camera_smooth  (③ §5.3)
```

根因對照見 §1.3；下表只列核心機制與開關：

| # | 貢獻 | 核心機制 | 開關 |
|---|---|---|---|
| **①** | **Gate Mechanism** | 中段 MLP 出 `g` → global attention 的 camera/register query 對動態 patch key 加 `−softplus(g)` bias；用運動定義的幾何標籤 `m*` 監督 | `enable_gate` + `loss.gate` |
| **②** | **Motion Loss** | 靜態區跨幀光度一致：GT 靜態點以**預測 pose** warp 到鄰幀，RGB 必須匹配 | `loss.static_photo` |
| **③** | **Temporal Attention** | 純時間軸 attention（每個空間位置 attend 自己的 S 幀）+ 軌跡平滑正則 | `enable_temporal` + `loss.camera_smooth` |

---

## 3. 貢獻① Gate Mechanism

> **在聚合 camera token 時，移除不符合相機運動的動態物體。**

### 3.1 中段 Gate Predictor —— `g` 怎麼產生

門控要在 aggregation **進行中**用到動態信號，但 motion 通常要讀**最終**特徵才得到 → 雞生蛋。解法：在 aggregator 中段插一個輕量預測器，**邊聚合邊出門控信號**。

```
            中段 aggregator 的 contextualized patch token
            (block_iter==gate_block_iter=7 完成後、已過 8 個 frame+global[+temporal] block)
            tokens[:, :, patch_start_idx:, :]  →  [B, S, P_patch, C]   # 切掉 camera(1)+register(4)
                    │                                                   # ← 目前輸入僅此，無 concat
                    ▼
            LayerNorm → Linear(C→C/4) → GELU → Linear(C/4→1)   # 末層 zero-init
                    ▼
              g  [B, S, P_patch]      # 每 patch 一個動態 logit
```

> **實作現況（重要，已對程式核實）**：gate predictor 的輸入是 **中段 aggregator 的 contextualized patch token**——不是原始 DINOv2 patch embedding，而是已跑過 8 個 aa-block（frame+global，`enable_temporal` 時每 3 個再加 temporal）的中段特徵，再 `[:, :, patch_start_idx:, :]` **切掉 camera+register special token**、只留 patch（[aggregator.py:380-383](../vggt/models/aggregator.py#L380-L383) 呼叫點 + [`GatePredictor.forward`](../vggt/models/aggregator.py#L54)：`x = norm(patch_tokens)`）。**沒有 concat 任何其他通道。** §3.7 描述的「幾何殘差**輸入通道**」`‖f^gt − f^cam‖` **尚未實作**，屬未來強化項，非現行架構。

`g` 的三個去處：
1. **門控（detach）**：`bias = min(0, softplus(0) − softplus(g.detach()))` → §3.2 的 attention bias。detach 確保 pose 梯度不反灌門控、避免「為 pose 好看而亂標動態」。
2. **監督（不 detach）**：`L_gate = BCE(σ(g), m*_patch)`，patch 解析度（見 §3.5）。
3. **輸出**：`σ(g)` 即 `motion_prob`（動態 mask 副產物）；需 pixel-level 圖時再接輕量 DPT upsample（同樣 `m*_patch` 監督）。

設計選擇：
- **解析度天然對齊**：`g` 在 patch 解析度，attention 的 key 本來就是 patch token → bias 直接套，零插值。
- **放中段 (~1/3)**：太早特徵未成熟；太晚則後面沒幾個 global block 可被清洗。1/3 是「特徵夠成熟」與「後段仍有 2/3 block 可門控」的平衡。
- **warm-start**：末層 zero-init → 訓練第 0 步 `g≡0` → `bias≡0` → 前向 byte-for-byte 等於 pretrained VGGT，不擾動既有幾何。`g` 隨訓練漸長，門控漸進通電。

### 3.2 運動門控相機聚合 —— bias 怎麼施加

attention：`A = softmax(QKᵀ/√d + bias)`，`out_i = Σ_j A[i,j]·V_j`。
**只**在 camera + register token 的 query 列，對 patch token 的 key 加負偏置：

```
bias[special_query, patch_key_j] = min(0, softplus(0) − softplus(g_j))   # g_j = patch j 的動態 logit (detach)
bias[其他所有位置]                = 0                                      # patch↔patch 完全不動
```

| patch j | g_j | bias | softmax 後權重 |
|---|---|---|---|
| 靜態 | →−∞ | →0 | 維持原樣 |
| init（zero-init）| 0 | **=0（精確）** | 逐位元 = pretrained VGGT |
| 動態 | →+∞ | →−∞ | →0（排除）|

**bias 函數為何是 `min(0, softplus(0) − softplus(g))`**（三個性質同時要）：
- `−softplus(g) = ln P_靜態`，softmax 後等於「**按靜態機率加權**」——單邊(≤0)、soft、只壓不升。
- `+softplus(0)` offset：讓 `g=0` 時 bias **精確為 0**，zero-init 門控在 init 逐位元等於 pretrained VGGT（warm-start 忠實；純 `−softplus(g)` 在 g=0 給 −ln2，會把每個 patch key 相對 special key 減半）。
- `min(0, ·)` 夾上界：拿回 **suppress-only**（靜態 patch 不被放大），避免 offset 引入正 bias。
- kink 在 g=0 不可微，但 **bias 走 detach、梯度不穿過它**（門控只靠 §3.5 的 BCE 學）→ 折角無害、前向仍連續。

效果：
- camera token **結構上只聚合靜態 patch** → pose 不再被運動污染（訓練+推理皆然）。
- **patch↔patch attention 一字不改** → depth/point 頭輸入特徵不變，**深度品質完全保留**。
- 後段每個 global block 都套此 bias，camera token 在後 2/3 聚合中**漸進地**只看靜態。

### 3.3 全動態幀的欠定問題（潛在，尚未驗證）

> **注意**：此問題最初在 §3.4 bug 修正前被誤判為「真正瓶頸」；修正後已知不成立（見頁首「進度與下一步」），此處僅保留作潛在風險記錄。

理論上若某幀幾乎全動態，門控排除絕大多數 patch → camera token 無料可聚合 → pose 可能**欠定**（資訊不足，非監督不足）。預留對策：
1. **soft bias（`−softplus` 而非 `−∞`）**：全動態時仍保留微弱全局信號（現行 bias 已是 soft）。
2. **top-k floor**：每幀至少保留 `g` 最小的 k 個 patch 參與聚合，保證最低限度輸入。
3. **temporal attention 借鄰幀**：透過時序塊從較靜態的相鄰幀取幾何上下文（**貢獻③ 的價值之一**，見 §5.1）。

### 3.4 實作關鍵：attention 拆兩條 path（顯存命脈）

把整個 `[N,N]` 浮點 mask 丟給 `F.scaled_dot_product_attention` 會讓 flash kernel fallback 到 memory 重的 math backend → 顯存暴增。正解：
- **patch↔patch**（佔 (S·P)² 大頭）：維持原 flash / memory-efficient attention，**不帶 bias**。
- **camera/register query → 所有 key**（只有 ~patch_start_idx·S 個 query）：單獨小 op，帶 bias；即使非 flash 也可忽略。

→ 貢獻① 本身顯存增加 ≈ 0。

> **⚠️ 實作正確性（2026-07-09 修正，commit `e064087`）**：global attention 的 token 是 **frame-interleaved** `[frame0(special+patch) | frame1(special+patch) | …]`，special token **散在每幀開頭**、不是集中在最前面。兩條 path 的切分必須逐幀 gather special query（`q.view(B,H,S,P,D)[:,:,:,:patch_start_idx]`），key bias 也照 interleaved 順序組（每幀 patch 欄位填 bias、special 欄位 0）。舊版用 `q[:,:,:n_special]`（`n_special=patch_start_idx·S`）誤把「最前 n_special 個」當 special → 因為 `n_special ≪ P`，那些索引全落在 frame 0 內 → **只 gate 到 frame 0 的 camera token，其餘幀完全沒作用**。這是 2026-07-06 舊結論作廢的原因。修正見 [`_gated_global_block_forward`](../vggt/models/aggregator.py)。

### 3.5 門控監督：`L_gate` + `m*` 標籤（跨域關鍵）

**跨域崩塌的真正原因是「PO 原生標籤有問題」，不是學習式 mask 本身的通病。** PO 的 `masks/` 是逐 instance 的**外觀分割**（非黑=前景）：室內連牆、地板、idle 前景都被標 → 這是**外觀 mask 不是運動 mask**。拿它監督，門控學到的是「PO 室內外觀→動態」，換到 Sintel 自然崩。

**解法：改用另外離線計算的「運動定義」幾何標籤來監督（已實作）。** 標籤在 PO 與 Sintel 是同一物理量（世界真的在動），跨域即不再崩。

> **`m*` 命名**：`m*_inst`（instance×3D-scene-flow 硬 mask，**PO 訓練實際採用**）、`m_geo`（幾何光流殘差 `‖f^gt − f^cam‖` 的**硬門檻** mask，Sintel/測試用；舊稱 `m*_raft`，但 Sintel 這條路徑吃的是 GT `.flo` 光流、非 RAFT，故正名 geo）、`m*_patch`（任一硬 mask average-pool 到 patch 格後、真正餵 `L_gate` 的軟機率）、`m_geo_soft`（同一殘差量的 **sigmoid 軟目標**，§3.7 未實作的理論式）。
>
> **`m*` 是貢獻① 與 ② 的共用基礎設施**：`L_gate` 用它當分類目標，`L_static_photo`（§4.2）用它挑靜態像素。

#### 3.5.1 監督目標與 loss（現行）

```
L_gate = BCE( σ(g), m*_patch )      # gate_logits 不 detach → 梯度回流 predictor
```
- **`m*_patch`**：pixel 級硬 0/1 動態 mask **average-pool 到 patch 格**的軟機率（`training/loss.py` 的 `m_star_patch`）。
- gate predictor 的**輸入只有 patch token**（見 §3.1），**尚未**接入 §3.7 的殘差輸入通道。
- **品質判斷用 AUC/F1，不看 BCE**：pool 後的軟標籤在 boundary patch 有不可約 floor，val BCE 會看似 overfit 卻與高 AUC 並存。
- **C-1 變體（hard label + ignore band）**：`m*_patch ≥ hard_hi(0.7) → 1`、`≤ hard_lo(0.3) → 0`、中間 band **不算 loss**，用意是砍掉 boundary floor。實作在 `compute_gate_loss`，config `inst_g_hard`。**結果見 §3.6 —— 是 null result。**

#### 3.5.2 動態標籤 `m*` 的產生（現行）

- **(a) instance × GT scene-flow `m*_inst`**（`dynmask_inst/`，**PO 訓練標籤實際採用**）：用 GT world track（`trajs_3d`）判斷每個 instance 連通塊是否在動，動則整塊填滿。**最穩**——靜態世界點位移恆為 0，無 RAFT 的背景灌爆與快慢兩難。細節見 `training/data/preprocess/po_instance_dynmask.py`；dataset 以 `dynamic_source="instance"` 載入。
- **(b) 幾何光流殘差 `m_geo`**：`‖f^gt − f^cam(GT depth+pose 重投影)‖ > bar`。純幾何、無需 GT track、跨域一致，**用於測試端/Sintel**。光流來源依資料集而異：**Sintel 用 GT `.flo` 光流**（`data/motion_mask.py` 實時算、落盤快取到 `<sintel_root>/mask/`）；其他測試集用 RAFT 估計光流（`dynmask_raft/`）——兩者同一物理量、皆是 geo 殘差。缺點是單一固定絕對門檻的快慢兩難，故 (1) 不用於訓練標籤；(2) Sintel 端改用**相對門檻** `bar = max(abs, α·‖f^gt‖)`（`derive_motion_mask(rel_threshold=α)`），解掉 cave_2 前半快速鏡頭整片灌爆。

**分工**：有 GT track（PO 訓練）用 (a)；無 GT track（Sintel/測試）用 (b)。兩者語義一致（動態=世界真的在動）——這就是「跨域不崩」的來源。

**跨域證據（S0，只訓 gate_predictor）**：

| 標籤 | PO AUC | Sintel AUC |
|---|---|---|
| RAFT 殘差 | 0.984 | **0.584** ← 崩 |
| **instance × scene-flow** | 0.926 | **0.767** |

instance 標籤讓 Sintel AUC **+0.183**，PO 只掉 0.058。**這是本貢獻最紮實的證據，但它證明的是 gate 品質，不是 pose。**

### 3.6 診斷現況與已知限制

> 完整表格見 [table.md](table.md)。以下為 clean 世代（post-`e064087`）的 run 1（`inst_g`）。

**(a) oracle 頭空只存在於短序列** —— gate 是結構性的短序列工具：

| frames | off | **oracle Δ** | **predicted Δ** |
|---|---|---|---|
| 4 | 0.0138 | **−10.8%** | +0.7% |
| 8 | 0.0268 | **−8.0%** | −1.0% |
| 12 | 0.0413 | **−15.0%** | −1.5% |
| 16 | 0.0438 | −3.8% | −0.2% |
| 32 | 0.1204 | **−11.0%** | −1.0% |
| **50（≈完整序列）** | 0.1537 | **+0.7%（歸零）** | −0.3% |

⚠️ f12 與 f16 之間 11 個百分點的跳動**目前無法解釋**（同 script、同 13 seq、同 `k=30`、同 `chunk_size=0`）。

**(b) `predicted` 在所有長度都 ≈ `off`（−0.3 ~ −1.5%）** —— 這是當前的**主瓶頸**。

**(c) 欠自信已確認，但病因未知**（`diag/gate_quality.py --cal_bins`，f16，14 seq）：

| | base | C-1 (hard) |
|---|---|---|
| AUC (macro) | 0.883 | 0.896 |
| F1 (macro) | 0.347 | 0.369 |
| p_dyn / p_stat | 0.346 / 0.065 | 0.368 / 0.065 |
| **校準 `signed_err`** | **+0.095** | **+0.089** |
| Sintel ATE (mean, 24 ep) | 0.1752 | 0.1785（Welch t=0.79，**無差異**）|

校準曲線顯示：**gate 給 0.35 的 patch，實際有 56% 在動**（`[0.3,0.4)` bin：pred 0.347 → empirical 0.561）。被低估的區間（0.2~0.6）**正好是 0.5 閾值所在**，含 12.9% 的 patch、實際動態率 47~72% —— 全被判成靜態。這完整解釋了 F1 0.35：**不是抓不到，是抓到了不敢跨門檻。**

**但 C-1 證偽了「BCE boundary floor 是病因」** —— 砍掉 floor 只讓 `signed_err` 降 6%、pose 完全沒動。**下一個假說：class imbalance**（`dyn_frac = 0.202`，靜態是動態的 4 倍 → BCE 系統性下拉），對應修法為正例 re-weight（`pos_weight ≈ 4`）。**未驗證。**

**(d) 逐序列：頭空與 gate 品質脫鉤**（f16）：

| seq | AUC | F1 | oracle Δ | 讀法 |
|---|---|---|---|---|
| ambush_5 | 0.968 | 0.323 | **−27.2%** | 排序近完美 + 最大頭空 + pred 只有 −0.7% → **欠自信最乾淨的證據** |
| cave_4 | **0.632** | 0.087 | **−26.7%** | 頭空第二大但排序最差 → 需 §3.7 幾何通道 |
| cave_2 | 0.863 | 0.073 | **+50.7%** | **唯一的 oracle 災難** —— 與 gate 品質脫鉤，指向標籤或機制本身 |

`corr(AUC, oracle Δ) = −0.006` —— 證實「該序列有多少頭空」與「gate 有多好」是兩個獨立的量。

### 3.7 未實作的跨域強化（未來項，記錄備考）

1. **sigmoid 軟化的監督目標 `m_geo_soft`**（取代 §3.5.1 的硬 mask pooling）：
   ```
   f^cam     = π( P_{t+1} P_t⁻¹ π⁻¹(u, D_t) ) − u        # 「假設靜態」的相機誘導光流（P,D 全用 GT）
   m_geo_soft = σ( α_m·( ‖f^gt − f^cam‖ − β_m ) )         # 對連續殘差做 soft threshold
   ```
2. **幾何殘差『輸入通道』**：把 `‖f^gt − f^cam(預測 pose+depth)‖` concat 進 gate predictor 輸入（§3.1）。此量在 PO 與 Sintel 是同一物理量，是「輸入端也接地」的跨域槓桿；但工程量較重（可微殘差、scale 對齊）。
   > **§3.6(d) 提升了這項的優先級**：`cave_4`/`ambush_4` 的 AUC 只有 0.63/0.62 且頭空巨大 —— 那是**特徵不足**，不是校準問題，只有輸入通道能救。

---

## 4. 貢獻② Motion Loss

> **根據物理幾何限制，設計對應的監督 loss，以符合動態資料集。**

### 4.1 動機：rigid-warp 只在靜態區成立

`L_cam` 直接拿 `pose_enc` 對 GT `pose_enc` 算 L1，**不經像素** —— 好處是動態像素不會污染 pose 的 loss，代價是 pose 只有一個「GT 的重參數化」訊號，沒有來自**真實影像內容**的獨立監督。

物理約束給了這個獨立訊號：**一個靜態世界點，用 GT depth 反投影、以相機的相對 pose warp 到下一幀，應該落在相同的 RGB 上。** 這個等式：
- 只依賴**剛體相機運動**，不需要 GT pose 出現在 loss 裡；
- **只在靜態點成立** —— 動態點即使 pose 完美也會違反 → 必須用 `m*`（§3.5.2）排除；
- 因此它天然是一個**動態場景專屬**的監督：靜態區給訊號、動態區被物理排除。

### 4.2 Static-Photo Loss

```
GT 靜態點 —(GT depth 反投影)→ 相機系 —(模型『預測』相對 pose)→ frame t+1 —(GT 內參投影)→ grid_sample 取 RGB
                                           ↓  取樣到的 RGB 必須匹配 source pixel（Huber loss）
```
- **只限 GT 靜態區**（`batch['motion_mask'] = m*_inst < dyn_thresh`）+ 有效深度。
- **梯度只進『預測 extrinsics』**（經 `F.grid_sample` 的取樣 grid）；GT depth/內參/影像皆常數，**depth head 完全不碰** → 維持 v3「不動幾何頭」的 scope。
- **遮擋防護**：把 t+1 的 `motion_mask` 也 warp 過去，source 點落在 t+1 動態區則丟棄。
- 用 `pose_enc_list[-1]`。實作：[`compute_static_photo_loss`](../training/loss.py)；config 開關 `static_photo`（檔名 suffix `_photo`）。

**變體（消融 run 3'）：改用預測深度反投影。** 把 GT depth 換成 `predictions["depth"]` → 梯度同時進 **depth_head 和 camera_head**，變成「pose+depth 聯合光度精修」。取捨：
- ⨯ 重新引入 **depth-pose 耦合/不可辨識**：重投影誤差 `= f(pose, depth)`，可靠「移動 depth」抄捷徑而非修 pose（v2 診斷過的偷懶捷徑），且可能改壞已領先的深度。
- ✓ 若要做，**必掛兩道保險**：(1) `L_depth`（對 GT depth）當錨；(2) `static_photo` 權重壓小、當微調。
- **列為獨立 ablation，不當預設**。

### 4.3 診斷現況與已知限制

> 結論見頁首「進度與下一步」（2026-07-24 一列）：**clean 世代重跑已完成，是負結果。**

**clean 世代**（`inst_g_photo`，warm from run1、同預算）：

| | ATE (14 seq) | ATE(12) | RPE-t | RPE-r |
|---|---|---|---|---|
| warm-start 起點（run1 ep15） | 0.1533 | 0.0517 | 0.0671 | 0.4923 |
| **photo，full-seq 最佳（ep10）** | **0.1583（比起點更差）** | 0.0496 | 0.0569 | 0.4562 |

ep10 已是全程最佳——ep10→ep20 單調惡化（見 [table.md](table.md) 表1）。static-photo loss 沒有把 pose 往好的方向拉，反而略微傷害。

**已知結構性問題**：
1. 本節目前**只有一個 loss**。`loss.py` 裡其餘的 `motion`/`flow`/`reproj`/`tsmooth` 都是 v1/v2 遺留、v3 未開。一個 loss 撐一個等價貢獻偏薄 —— **待補**。
2. `m*` 的幾何推導論述目前放在 §3.5.2（歸貢獻①），但它同時是本貢獻的立論基礎。歸屬待定。
3. **負結果的原因尚未排查**：可能是 loss 權重、GT-depth 反投影的 scale 對齊、或遮擋防護不足，尚未逐一消融。

**歷史記錄（僅存檔，已知不可信）**：bug 世代的 `inst_g_photo` 曾顯示 ATE 0.1620（vs base −5.5%），但那是離群值撐出來的——整個 gain 來自把 `temple_3` 從 0.6049 拉到 0.4346，12 個正常序列上其實比 base 差（ATE(12) +7.7%），且 `sqRel 6.210` 全場最差。**方向與 clean 世代的負結果一致**，只是當時被離群值蓋過去。

**兌現路徑**：§7.3 的 run 3 已完成，結果為負。**不建議繼續投入**，除非重新設計 loss（見上方結構性問題）。

---

## 5. 貢獻③ Temporal Attention

> **根據影片在時序上有關聯性的特性，新增跟時序相關的 attention。**

### 5.1 Temporal Aggregator 機制

VGGT 原生 aggregator 只有兩種 attention：**frame**（逐幀、單張影像內的空間注意力）與 **global**（把所有幀所有 token 攤平一起做）。缺一種「**純時間軸**」的注意力——讓同一空間位置沿它自己的 S 幀互看。

- **開關**：`enable_temporal=True` → `aa_order` 由 `[frame, global]` 變 `[frame, temporal, global]`。每 `temporal_every=3` 個 aa-block 插一個 temporal block → `n_temporal = depth // temporal_every = 24 // 3 = 8` 個。
- **時間軸 attention**：把 tokens reshape 成 `(B·P, S, C)`，attention 沿 **S(時間)** 跑——每個空間位置 attend 自己的 S 幀，捕捉軌跡/運動、並讓 camera token 從**較靜態的鄰幀**取幾何上下文。算完 reshape 回 `(B·S, P, C)`。
- **不改 head 介面**：temporal block **只更新 streaming tokens，不 emit intermediate** → head 輸入維持 `[B,S,P,2C]`（frame+global 串接），下游 head 維度不變。
- **獨立時間 RoPE**：用 1D `RotaryPositionEmbedding1D`，與空間 2D RoPE **分開** → 不動 pretrained 空間位置編碼。camera token(idx 0)與 patch token 給真實 frame index；register token 給 0（不轉）。

**與貢獻① 的關係**：temporal 是 §3.3「全動態幀借鄰幀靜態上下文」的載體 —— 兩者正交但互補。

### 5.2 Warm-start（與 gate 同哲學）

每個 temporal block 的 **LayerScale γ=0** → init 時整塊是**身分映射** → 加了 temporal 仍能 byte-for-byte 載 VGGT-1B（`enable_temporal=False` 時根本不建，也保持相容）。訓練中 γ 漸長、temporal 漸進通電。實作見 [`_process_temporal_attention`](../vggt/models/aggregator.py) 與 `Aggregator.__init__` 的 `temporal_blocks`。

### 5.3 Camera-Smooth Loss（軌跡時序正則）

硬/動態序列上觀察到明顯的**軌跡「跳動」**（native VGGT-1B 也有，非 v3 回歸）。加一個**只作用在網路自身輸出**的平滑先驗（類 MonST3R 的 trajectory-smoothness，但當訓練 loss 而非 test-time 項）。

- 對**預測 pose 序列**罰 **2 階（加速度）不連續**：`accel = v[t+1] − v[t]`，`v = ΔT/Δt`，對 T 與四元數各算。
- **Δt 正規化**：訓練 clip 抽幀間距不規則、可能重複（`batch["ids"]` 為真實時序 index）→ 用 `ΔT/Δt` 而非 raw ΔT；`Δt=0`（重複幀）排除。
- **四元數半球 sign-fix**：差分前對齊 `q/−q`，避免 sign flip 被誤當大旋轉跳動。
- **FoV 不平滑**；多 refine stage 以 `γ` 加權；需 `S≥3`（故 `img_nums` min ≥ 4）。**不用 GT pose**，純自參照正則。
- 實作：[`compute_camera_smooth_loss`](../training/loss.py)；config 開關 `camera_smooth`（檔名 suffix `_smooth`）。

> **為何與 temporal 同節**：兩者都在利用「影片有時序結構」這件事 —— temporal 是**機制**（讓網路能跨幀看），camera_smooth 是**目標**（要求輸出時序連貫）。但 camera_smooth **不需要 temporal 才能算**（它只罰預測 pose 的加速度）→ **若 run 2 有效，必須再拆一次才能分辨是結構還是 loss 在起作用**（見 §7.3）。

### 5.4 診斷現況與已知限制

> 結論見頁首「進度與下一步」（2026-07-27 一列）：**clean 世代重跑已在進行，目前是全 clean 家族 pose 最好的結果，但仍在訓練、且因子未拆解。**

**clean 世代**（`inst_gts`，warm from run1、同預算）：

| ckpt | ATE (14 seq) | ATE(12) | RPE-t | RPE-r |
|---|---|---|---|---|
| warm-start 起點（run1 ep15） | 0.1533 | 0.0517 | 0.0671 | 0.4923 |
| windowed `best.pt`（ep17，⚠️不可信） | 0.1651 | — | — | — |
| `epoch_30` | 0.1343 | 0.0500 | 0.0633 | 0.3943 |
| **`last.pt`（≈ep30+，仍在訓練）** | **0.1253** | **0.0470** | 0.0596 | **0.3867** |

- **單調下降未收斂**：ep20 0.180 → ep30 0.134 → `last` 0.125。還沒到可以定論的收斂點。
- **windowed `best.pt` 選點不可信**：`best.pt`（ep17）比 `last` 差 24%（0.1651 vs 0.1253）——這條線**一律取 `last.pt`**，不要用 `best.pt`（見 [table.md](table.md) 表1附註）。
- **因子仍混淆**：這個 run **同時**開了 temporal 解凍**和** `camera_smooth` loss，無法歸因是哪一半在起作用——§7.3 早已預告這個混淆，尚未拆解。
- **尚未過雜訊帶檢驗**：相對 warm-start 起點 −18%、相對 base −27%，皆大於單次比較但小於 §8.3 的 2σ=16.9% 門檻的 1~1.6 倍——方向樂觀但按協定還不能拍板，須等收斂 + 用非報告指標驗證。

**已棄用的舊數字**：bug 世代的同名 run（`best.pt`=epoch 2，config 預算 20 epoch，只跑了 10%）ATE 0.1568，且同時混了兩個因子；已被 clean 世代的 run1（0.1533，無 temporal、無 smooth）超越，說明那個數字反映的只是 bug fix 的缺席，無分析價值。

**兌現路徑**：§7.3 的 run 2 已在跑（見上）；等收斂後，**下一步是拆 temporal-only vs smooth-only** 才能歸因到哪一半在起作用（見頁首優先序第 1 項）。

---

## 6. 損失函數總表

```
L = L_cam + L_depth + λ_g · L_gate  [+ λ_p · L_static_photo]  [+ λ_s · L_camera_smooth]
      └ VGGT 原生      └ 貢獻①          └ 貢獻②                    └ 貢獻③
```

- **`L_cam`（不變）**：逐分量 L1（T/q/FoV），跨 refine stage 以 `γ^{n−i−1}` 加權。
  > pose loss 直接拿 `pose_enc` 對 GT `pose_enc` 算，**不經像素** → 動態像素**不會污染 pose 的 loss**。pose 提升來自**貢獻① 的前向門控**，非 loss 屏蔽。
- **`L_depth`（不變）**：confidence-weighted + 多尺度梯度。（point 頭在純 pose scope 關掉。）
- **`L_gate`（①）**：`BCE(σ(g), m*_patch)`，§3.5。`λ_g` 起始 1.0。
- **`L_static_photo`（②）**：§4.2。
- **`L_camera_smooth`（③）**：§5.3。
- **中括號兩項由 yaml 開關**：`MultitaskLoss.forward` 只在對應 config block 存在時才加該項。兩者梯度**都只進預測 pose**，不碰幾何頭。

**gate 品質判斷準則見 §3.5.1**（一律用 AUC/F1，不看 BCE）。

**`valid_frame` 建議維持原本寬鬆的「有效點數 > 100」**，不要改成嚴格的「靜態點數 > 100」：後者會在多動態場景造成幀數不足，而 pose 監督是直接的、得不到對等好處。

---

## 7. 訓練策略

VGGT 是強預訓練 baseline，**絕不從頭訓**。三階段漸進解凍。

### 7.1 三階段課程

| 階段 | 凍結 | 解凍訓練 | 開的 loss | 目的 |
|---|---|---|---|---|
| **S0** | aggregator 全部 + 原三頭 | 只 gate predictor | `L_gate` | 養門控信號；門控因 zero-init 仍 ≈no-op |
| **S1** | patch_embed + frame_blocks + 前段 global(0–7) + depth/point head | gate predictor + 後段 global(8–23) + camera head | `L_gate` + `L_cam` | 門控通電，camera 學會「只看靜態」 |
| **S2**（保幾何版）| patch_embed + frame_blocks + depth_head | 全部 global(0–23) + gate predictor + camera head | `L_cam` + `L_depth` + `L_gate` + 混 ~30% 靜態，lr 1e-5 | 端到端精修 |

**S1 一律在修正後前向（§3.4）下訓練。** gate_predictor 的品質是 `BCE(σ(g), m*)` 監督的，**與 §3.4 的 attention bug 無關** → 可從 **S0 checkpoint warm-start**，只重訓 camera 路徑。

**S2 刻意不照字面「全網解凍」**（v3 賣點是「不動幾何頭、保留深度領先」）：
- `patch_embed`（DINOv2）**永遠凍**——最強 pretrained 特徵、depth 領先的根。
- `frame_blocks`**凍住或極小 lr**——pose-motion 耦合在 **global attention**，不在 frame attention。
- `depth_head` 凍住但 **`L_depth` 保留**：depth 梯度只流回 shared trunk，把 trunk 釘住不漂 → 防遺忘錨。
- S2 真正解凍的是 **全部 global blocks + gate predictor + camera head**。

### 7.2 硬體適配（單張 RTX 5090 32GB）

**(a) 純 pose scope 的最大紅利：關掉用不到的 dense 頭**

| Head | 訓練需要 | 理由 |
|---|---|---|
| camera | ✅ | 主角 |
| depth | ✅ | 算 `f^cam` 殘差（§3.7 輸入通道）需預測深度 |
| point / track / flow / motion(DPT) | ❌ 關掉 | pose 不需；`g` 已當 mask |

**(b) S0 快取凍結特徵**：S0 aggregator 全凍 → 不 backprop 穿過 aggregator → 把中段特徵離線預算存盤，S0 只訓 gate MLP → 顯存近零。

**(c) 機動槓桿（按性價比）**
1. 降解析度 518→364（attention 記憶體約 ¼，幀數翻倍）。
2. gradient checkpointing（已開）、bf16 autocast（已開）。
3. 時序加大 stride 抽幀：更大 baseline、對 pose 更有利。
4. gradient accumulation 補足等效 batch。

### 7.3 因子消融 runs 1–4

以「gate→camera」為 base，貢獻② 與 ③ **各自單獨疊上 base** 再合併，才能分別歸因**邊際貢獻**。

**統一架構（四個 run 共用，關鍵）**：所有 run 都建成 `enable_gate=True + enable_temporal=True`。run 1 裡 temporal 仍 γ=0 身分、且凍住（等於沒有），但 checkpoint 帶著這些權重 → run 2/3/4 能直接載 run 1、不會 missing key。

| run | 貢獻 | base 上額外**解凍** | 額外 **loss** | warm-start | 測什麼 | 現況 |
|---|---|---|---|---|---|---|
| **1（base）** | ① | —（gate_predictor + global 8–23 + camera_head）| —（`L_cam + L_gate`）| `s0_inst` | gate 對 pose 的效果（`predicted`→`oracle`？）| ✅ 完成（clean run1，best ep15，ATE 0.1533） |
| **2** | ③ | temporal_blocks | `camera_smooth` | run 1 | temporal + 平滑的邊際貢獻 | 🔵 **進行中**（`smooth_temporal`；§5.4——目前全 clean 家族最佳，但未收斂、因子未拆） |
| **3** | ② | depth_head*（可凍）| `static_photo`（**GT depth**）| run 1 | 獨立光度信號的邊際貢獻（守 depth）| ✅ 完成，**負結果**（`photo`；§4.3） |
| **3'** | ② 變體 | **depth_head** | `static_photo`（**預測深度**）+ `L_depth` 錨 | run 1 | pose+depth 聯合光度精修（帶耦合風險）| ⬜ 未跑 |
| **4** | ①+②+③ | temporal + depth_head | `camera_smooth` + `static_photo` + `L_depth` | run 1 | 全開的組合效果 | ⚠️ 曾跑一版但 lineage 汙染（`photo_smooth_temporal`，見頁首 2026-07-10）；**乾淨版未跑**，待 run 2 收斂+拆解後才值得跑 |

\* run 3 用 GT depth 時 static_photo 梯度只進 pose，depth_head 可凍；run 3'（預測深度）**必須**解凍 depth_head 且開 `L_depth` 當錨（§4.2 變體）。

**凍結原則（四個 run 一致）**：`patch_embed`、`frame_blocks`、前段 global(0–7) 永遠凍；point/track head 關掉。

**⚠️ 四個 run 必須跑到同一個 epoch 預算** —— 見 §8。

**⚠️ run 2 混淆了兩個因子**（temporal 結構 + camera_smooth loss）。若 run 2 有效，**必須再拆一次**（temporal-only / smooth-only）才能歸因到貢獻③ 的哪一半。

**評測與歸因**：每個 run 完成後跑 `diag/gate_bias_ablation.py`（`predicted` vs `off`/`oracle`）+ `diag/gate_quality.py`（AUC/F1/校準）+ 軌跡可視化。2 vs 1、3 vs 1 給各自邊際量，4 給組合（是否正交相加或互相干擾）。

**收尾（可選）= §7.1 的 S2 保幾何版**：從 run 4 warm-start，混 ~30% 靜態、lr 1e-5、全 loss 同開。

---

## 8. 評測協定

> ⚠️ **本節記錄的問題會影響上面所有數字的讀法。完整分析見 [table.md](table.md)。**

### 8.1 目前的協定（含已知缺陷）

| 層 | 資料 | 訊號 | 決定性？ | 問題 |
|---|---|---|---|---|
| channel A（windowed val） | Sintel，隨機抽 4~16 幀 | loss + ATE | ❌ seed = `seed_value + epoch×100` **每 epoch 換資料** | 變動 = 模型 + 資料，分不開 |
| channel B（pose_eval） | Sintel 14 seq 全序列 | ATE | ✅ 完全決定性 | **= 報告指標** |
| 報告 | Sintel 14 seq 全序列 | ATE | — | 與 channel B 同一批資料 |

**核心缺陷：沒有任何 held-out 資料。** `SINTEL_EVAL_SEQUENCES` 被 val、pose_eval、報告、以及所有 `diag/` 工具共用同一份 14 seq 清單。

**但權重是乾淨的** —— 訓練資料是 PO/TartanAir/Waymo/Spring，模型從未在 Sintel 上收過梯度。**問題純粹是 checkpoint selection**，不是梯度洩漏。

**參考實作對照**：

| | 預算 | best 機制 | val 資料 | 選擇訊號 = 報告指標？ |
|---|---|---|---|---|
| 原版 VGGT | `max_epochs: 20` | **完全不存在**（`grep best` = 0） | Co3D `split='test'`（in-domain held-out） | 不適用（不選） |
| MonST3R | `--epochs=50` | 選 `loss_med` | PO-test + Sintel（`seed=777` **釘死**） | ❌ 不同 |
| **本專案** | `max_epochs: 50` | 選 **Sintel 全序列 ATE** | Sintel（seed 每 epoch 變） | **✅ 同一個** |

### 8.2 選擇偏差的實測量級

用 run 1 的 24 個 epoch 做 leave-one-out（用 13 seq 選 best epoch，測第 14 個）：

```
在參與選擇的 14 seq 上   −12.5%
在 held-out seq 上       −11.5%      ← 14 個 fold 有 13 個都選到 ep15
─────────────────────────────────
選擇偏差 ≈ 1%
```

**選擇強烈泛化 → `best.pt` 是真實更好的模型，不是抽樣運氣。** 偏差約 1 個百分點，可接受。**但這需要 LOO 分析才能辯護；改用非報告指標選擇則無需辯護。**

### 8.3 雜訊帶與最小可偵測效應

run 1 的 24 個 epoch（channel B，決定性）：`mean=0.1752, std=0.0149`，**無收斂趨勢**（ep24 0.1713 ≈ ep2 0.1715）。

- **單次比較的 2σ 門檻 = 16.9%** —— 小於此的「改善」不可信。
- **雜訊集中在單一序列**：`market_5` 逐 epoch 在 0.093 ~ 0.800 之間擺盪 **8.6 倍**，**獨佔 83.9% 的變異**。
- 對比 `cave_2`：mean 1.148 但 **CV 只有 4.7%** —— **難 ≠ 不穩**。market_5 的雙峰特徵是離散失敗模式，不是難度。

### 8.4 待修項（優先序）

1. **固定 epoch 預算** —— `best` 隨 N 單調變好（實測 N=5 的 0.1594 → N=15 的 0.1533，**4.0% 純來自跑更久**）。四個 ablation run 必須同 N。run 1 目前停在 **24/50**。
2. **查 market_5 的 8.6 倍擺盪** —— 一個序列 83.9% 的雜訊，最便宜的槓桿。
3. **加 in-domain held-out val（PO-test），seed 釘死** —— 回答「optimization 健康嗎」。⚠️ **不可用於跨域決策**：§3.5.2 的標籤實驗證明 in-domain 會誤導（用 PO AUC 選會選到 RAFT）。
4. **把 best 的選擇訊號換成非報告指標** —— 需先驗證該訊號與 Sintel ATE 相關。
5. **base (VGGT-1B) 只有 f50 一個長度** —— 無法分離「bug fix 收益」與「短序列較易」。

---

## 9. 與 v1/v2 的關係

- **保留**：temporal aggregator（v1 貢獻①）→ **升格為 v3 貢獻③**（§5）。正交、未被診斷實證有問題，且 §3.3 借鄰幀需要它。
- **丟棄**：雙場 `X^can + m·Δ`、scene-flow 頭、`L_reproj` / `L_tsmooth` / `L_flow`、4D 全局對齊；**test-time polish 也一併移除**（純前饋優先）。
- **新增**：中段 gate predictor、運動門控 attention、幾何接地監督（**貢獻①**）；static-photo loss（**貢獻②**）；camera-smooth loss（併入**貢獻③**）。

**評測聚焦 pose**：Sintel / PointOdyssey 的 ATE / RPE，對標 MonST3R 與原版 VGGT；輔以動態 mask 的 AUC（跨域 PO→Sintel 是關鍵指標）與 depth 不退化的對照。

⚠️ **MonST3R 是 test-time optimization（分鐘級/序列），本方法是單次前饋（秒級）。** raw ATE 直接對打是拿一次前向打一個最佳化迴圈。**成本軸（wall-clock / VRAM / 是否需 per-scene optimization）目前完全沒有資料，但它是本方法定位的核心宣稱 —— 待補。**

---

## 一句話總結 novelty

> Dyn-VGGT v3 從**三個正交的軸**解除 VGGT 在動態場景的 pose-motion 耦合：**前向**上，以中段輕量 gate 在 global attention 裡讓 camera token 結構性地只聚合靜態 patch（貢獻①），並用 **domain-invariant 的運動定義幾何標籤**（`m*_inst`/`m_geo`）監督它，根治 PO 原生外觀 mask 的跨域崩塌；**監督**上，用「剛性 warp 只在靜態區成立」這條物理約束，造出獨立於 GT pose 重參數化的光度訊號（貢獻②）；**時序**上，補足 VGGT 缺失的純時間軸 attention 與軌跡連貫性目標（貢獻③）。三者皆為 identity warm-start、皆不動幾何頭 —— 在保留深度領先的前提下，**純前饋**地拿回動態場景的相機 pose。
