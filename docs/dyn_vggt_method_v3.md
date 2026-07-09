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

## ⚠️ 狀態更新（2026-07-09：2026-07-06 舊結論已作廢）

> **2026-07-06 曾記錄「貢獻②門控對 pose 幾乎無增益、即使 oracle 也救不回、瓶頸是 §4.2 全動態幀 information-starvation」——這個結論是在一份有 bug 的門控前向上量的，現已作廢。**
>
> **根因是程式 bug，不是機制無效。** global attention 的 token 是 **frame-interleaved** 排列 `[frame0(special+patch) | frame1(...) | …]`，但 `_gated_global_block_forward` 舊版用 `q[:, :, :n_special]` 假設 special token 集中在最前面 → 實際上**只有 frame 0 的 camera token 被門控，frame 1..S-1 完全沒 gate**。pose 是全幀平均，`(S-1)/S` 的幀走的是與 `off` 完全相同的計算 → 這正是 off/predicted/oracle 三者幾乎無差的原因（連 oracle 也一樣，因為走同一條壞路徑）。
>
> **修正後（frame-interleaved 正確切分 + clamped warm-start bias，見 §4）實測**：高動態序列 oracle 明確有效。Sintel `cave_2`（動態比例 0.983）ATE `off 0.0453 → oracle 0.0353（−22%）`；5 序列平均 oracle 0.0298 vs off 0.0323。**§4.2 的「information-starvation 才是瓶頸」結論撤回**——masking-accuracy 確實有用。
>
> **仍待辦**：`predicted`（模型自己的門控）目前仍 ≈ `off`，因為現有 checkpoint 的 gate_predictor 與 camera head 是在**壞掉的前向**下訓練的、從沒學過利用門控。需**用修正前向重訓 S1**（見 §8），才能公平評估「學出來的 gate 對 pose 有沒有用」。

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
           ▼
        解耦相機軌跡
```

| # | 貢獻 | 解決的根因 | 核心機制 |
|---|---|---|---|
| ① | **中段 Gate Predictor** | 雞生蛋（門控需 m、m 來自最終特徵）| 在 ~1/3 深度用輕量 MLP 出 patch 動態 logit `g` |
| ② | **運動門控相機聚合** | §1.1 pose-motion coupling | global attention 的 camera/register query 對動態 patch key 加 `−softplus(g)` bias |
| ③ | **幾何接地的門控監督** | mask 跨域崩塌 | `g` 的**目標**用運動定義的幾何標籤（m\*_inst / m\*_raft）→ domain-invariant |

**範圍刻意鎖定「純 pose」且純前饋（feed-forward）**：不輸出 scene flow、不改幾何表示、**不做 test-time 後處理**（避免推論落到後處理；難例兜底留待未來）。depth/point 頭原樣全監督，深度領先保留。

---

## 3. 貢獻①：中段 Gate Predictor —— `g` 怎麼產生

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

> **實作現況（重要，已對程式核實）**：gate predictor 的輸入是 **中段 aggregator 的 contextualized patch token**——不是原始 DINOv2 patch embedding，而是已跑過 8 個 aa-block（frame+global，`enable_temporal` 時每 3 個再加 temporal）的中段特徵，再 `[:, :, patch_start_idx:, :]` **切掉 camera+register special token**、只留 patch（[aggregator.py:380-383](../vggt/models/aggregator.py#L380-L383) 呼叫點 + [`GatePredictor.forward`](../vggt/models/aggregator.py#L54)：`x = norm(patch_tokens)`）。**沒有 concat 任何其他通道。** §5.3 描述的「幾何殘差**輸入通道**」`‖f^gt − f^cam‖` **尚未實作**，屬未來強化項，非現行架構。

`g` 的三個去處：
1. **門控（detach）**：`bias = min(0, softplus(0) − softplus(g.detach()))` → §4 的 attention bias。detach 確保 pose 梯度不反灌門控、避免「為 pose 好看而亂標動態」。
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
- kink 在 g=0 不可微，但 **bias 走 detach、梯度不穿過它**（門控只靠 §5 的 BCE 學）→ 折角無害、前向仍連續。

- camera token **結構上只聚合靜態 patch** → pose 不再被運動污染（訓練+推理皆然）。
- **patch↔patch attention 一字不改** → depth/point 頭輸入特徵不變，**深度品質完全保留**。
- 後段每個 global block 都套此 bias，camera token 在後 2/3 聚合中**漸進地**只看靜態。

### 4.2 全動態幀的欠定問題（潛在，尚未驗證）

> **注意**：這個問題是在 §4 門控 bug **修正前**提出的，當時被誤判為「即使 oracle 也救不回 → 真正瓶頸」。bug 修正後 oracle 已明確有效（見頁首 banner），所以此處**只作為一個尚待驗證的潛在風險記錄**，不再是已成立的結論。是否真的成為瓶頸，要等 §8.3 run 1 的結果（若 gate 生效後 `predicted` 仍 ≈ `off`，才回頭檢視這條）。

理論上若某幀幾乎全動態，門控排除絕大多數 patch → camera token 無料可聚合 → pose 可能**欠定**（資訊不足，非監督不足）。預留對策：
1. **soft bias（`−softplus` 而非 `−∞`）**：全動態時仍保留微弱全局信號（現行 bias 已是 soft）。
2. **top-k floor**：每幀至少保留 `g` 最小的 k 個 patch 參與聚合，保證最低限度輸入。
3. **temporal attention 借鄰幀**：透過時序塊從較靜態的相鄰幀取幾何上下文（§6 的 temporal aggregator 價值之一）。

### 4.3 實作關鍵：attention 拆兩條 path（顯存命脈）

把整個 `[N,N]` 浮點 mask 丟給 `F.scaled_dot_product_attention` 會讓 flash kernel fallback 到 memory 重的 math backend → 顯存暴增。正解：
- **patch↔patch**（佔 (S·P)² 大頭）：維持原 flash / memory-efficient attention，**不帶 bias**。
- **camera/register query → 所有 key**（只有 ~patch_start_idx·S 個 query）：單獨小 op，帶 bias；即使非 flash 也可忽略。

→ 架構②本身顯存增加 ≈ 0。

> **⚠️ 實作正確性（2026-07-09 修正）**：global attention 的 token 是 **frame-interleaved** `[frame0(special+patch) | frame1(special+patch) | …]`，special token **散在每幀開頭**、不是集中在最前面。兩條 path 的切分必須逐幀 gather special query（`q.view(B,H,S,P,D)[:,:,:,:patch_start_idx]`），key bias 也照 interleaved 順序組（每幀 patch 欄位填 bias、special 欄位 0）。舊版用 `q[:,:,:n_special]`（`n_special=patch_start_idx·S`）誤把「最前 n_special 個」當 special → 因為 `n_special ≪ P`，那些索引全落在 frame 0 內 → **只 gate 到 frame 0 的 camera token，其餘幀完全沒作用**（見頁首狀態 banner）。修正見 [`_gated_global_block_forward`](../vggt/models/aggregator.py)。

---

## 5. 貢獻③：幾何接地的門控監督（跨域關鍵）

**跨域崩塌的真正原因是「PO 原生標籤有問題」，不是學習式 mask 本身的通病。** PO 的 `masks/` 是逐 instance 的**外觀分割**（非黑=前景）：室內連牆、地板、idle 前景都被標 → 這是**外觀 mask 不是運動 mask**。拿它監督，門控學到的是「PO 室內外觀→動態」，換到 Sintel 自然崩。

**解法：改用另外離線計算的「運動定義」幾何標籤來監督（已實作）。** 標籤在 PO 與 Sintel 是同一物理量（世界真的在動），跨域即不再崩——這是本貢獻的核心，見 §5.2 的 `m*_inst` / `m*_raft`。§5.3 的「sigmoid 軟目標」與「幾何殘差**輸入通道**」是可選的**進一步**強化，非跨域必要條件（目前程式皆未接）。

> **`m*` 命名**：`m*_inst`（instance×3D-scene-flow 硬 mask，**PO 訓練實際採用**）、`m*_raft`（RAFT 殘差硬 mask，Sintel/測試用）、`m*_patch`（任一硬 mask average-pool 到 patch 格後、真正餵 `L_gate` 的軟機率）、`m*_geo`（§5.3 未實作的 sigmoid 軟目標理論式）。

### 5.1 實作：監督目標與 loss（現行）

```
L_gate = BCE( σ(g), m*_patch )      # gate_logits 不 detach → 梯度回流 predictor
```
- **`m*_patch`**：pixel 級硬 0/1 動態 mask **average-pool 到 patch 格**的軟機率（`training/loss.py` 的 `m_star_patch`）。硬 mask 來源見 §5.2。
- gate predictor 的**輸入只有 patch token**（見 §3 實作現況），**尚未**接入 §5.3 的殘差輸入通道。
- **品質判斷用 AUC/F1，不看 BCE**：pool 後的軟標籤在 boundary patch 有不可約 floor，val BCE 會看似 overfit 卻與高 AUC 並存。

### 5.2 動態標籤 `m*` 的產生（現行）

**PointOdyssey 原生 mask 不能直接用**：`masks/` 是逐 instance 分割（「非黑=動態」），室內連牆/地板都被標、idle 前景也照標 → 是**外觀 mask** 不是運動 mask，會教門控學「室內外觀→動態」而跨域崩塌。故改用**運動定義**的標籤，兩種：

- **(a) instance × GT scene-flow `m*_inst`**（`dynmask_inst/`，**PO 訓練標籤實際採用**）：用 GT world track（`trajs_3d`）判斷每個 instance 連通塊是否在動，動則整塊填滿。**最穩**——靜態世界點位移恆為 0，無 RAFT 的背景灌爆與快慢兩難。細節見 `training/data/preprocess/po_instance_dynmask.py`；dataset 以 `dynamic_source="instance"` 載入。
- **(b) RAFT 光流殘差 `m*_raft`**（`dynmask_raft/`）：`‖f^gt(RAFT) − f^cam(GT)‖ > τ`。純幾何、無需 GT track、跨域一致，**用於測試端/Sintel**。缺點是單一固定 Δ 兩難（慢速需大 Δ、快相機大 Δ 會把靜態背景灌爆），故不用於訓練標籤。

**分工**：有 GT track（PO 訓練）用 (a)；無 GT track（Sintel/測試）用 (b)。兩者語義一致（動態=世界真的在動）——這就是「跨域不崩」的來源。兩者都是 pixel 級硬 mask，餵 `L_gate` 前 average-pool 成 `m*_patch`。

### 5.3 未實作的跨域強化（未來項，記錄備考）

以下兩者是原始設計裡「更徹底 domain-invariant」的手段，**目前程式皆未接**：

1. **sigmoid 軟化的監督目標 `m*_geo`**（取代 §5.1 的硬 mask pooling）：
   ```
   f^cam  = π( P_{t+1} P_t⁻¹ π⁻¹(u, D_t) ) − u          # 「假設靜態」的相機誘導光流（P,D 全用 GT）
   m*_geo = σ( α_m·( ‖f^gt − f^cam‖ − β_m ) )            # 對連續殘差做 soft threshold
   ```
   固定穩定目標、不依賴模型當下預測；現行改用 §5.2 硬 mask + average-pool，尚未實作此式。

2. **幾何殘差『輸入通道』**：把 `‖f^gt − f^cam(預測 pose+depth)‖` concat 進 gate predictor 輸入（見 §3）。此量在 PO 與 Sintel 是同一物理量，是「輸入端也接地」的跨域槓桿；但工程量較重（可微殘差、scale 對齊），目前僅用中段 patch token 當輸入。

---

## 6. 保留架構：Temporal Aggregator（跨幀注意力，保留自 v1 貢獻①）

VGGT 原生 aggregator 只有兩種 attention：**frame**（逐幀、單張影像內的空間注意力）與 **global**（把所有幀所有 token 攤平一起做）。缺一種「**純時間軸**」的注意力——讓同一空間位置沿它自己的 S 幀互看。v1 貢獻① 補上這條，v3 保留(與門控正交，且 §4.2 全動態幀借鄰幀靜態上下文需要它)。

### 機制

- **開關**：`enable_temporal=True` → `aa_order` 由 `[frame, global]` 變 `[frame, temporal, global]`。每 `temporal_every=3` 個 aa-block 插一個 temporal block → `n_temporal = depth // temporal_every = 24 // 3 = 8` 個。
- **時間軸 attention**：把 tokens reshape 成 `(B·P, S, C)`，attention 沿 **S(時間)** 跑——每個空間位置 attend 自己的 S 幀,捕捉軌跡/運動、並讓 camera token 從**較靜態的鄰幀**取幾何上下文。算完 reshape 回 `(B·S, P, C)`。
- **不改 head 介面**：temporal block **只更新 streaming tokens，不 emit intermediate** → head 輸入維持 `[B,S,P,2C]`（frame+global 串接），下游 head 維度不變。
- **獨立時間 RoPE**：用 1D `RotaryPositionEmbedding1D`，與空間 2D RoPE **分開** → 不動 pretrained 空間位置編碼。camera token(idx 0)與 patch token 給真實 frame index；register token 給 0（不轉）。

### Warm-start（與 gate 同哲學）

每個 temporal block 的 **LayerScale γ=0** → init 時整塊是**身分映射** → 加了 temporal 仍能 byte-for-byte 載 VGGT-1B（`enable_temporal=False` 時根本不建，也保持相容）。訓練中 γ 漸長、temporal 漸進通電。實作見 [`_process_temporal_attention`](../vggt/models/aggregator.py) 與 `Aggregator.__init__` 的 `temporal_blocks`。

> **與訓練流程的關係**：temporal 是 **camera-smooth（§7.2）的跨幀機制載體**，也是 §4.2 借鄰幀的對策。整合流程中由 run 2 / run 4 解凍並訓練（見 §8.3）。注意 camera-smooth loss 本身不強制 temporal 才能算（它只罰預測 pose 的加速度）——兩者是「機制 + 目標」的搭配，若 run 2 有效仍要分辨是 temporal 結構還是 smooth loss 在起作用。

---

## 7. 損失函數（純 pose 版，比 v1 大幅精簡）

丟掉雙場、scene-flow、reproj 閉環、tsmooth，核心只剩三項，外加兩個**可選**的 pose-only 擴充：

```
L = L_cam + L_depth + λ_g · L_gate  [+ λ_p · L_static_photo]  [+ λ_s · L_camera_smooth]
```

- **`L_cam`（不變）**：逐分量 L1（T/q/FoV），跨 refine stage 以 `γ^{n−i−1}` 加權。
  > 注意：pose loss 直接拿 `pose_enc` 對 GT `pose_enc` 算，**不經像素** → 動態像素**不會污染 pose 的 loss**。pose 提升來自**架構②的前向門控**，非 loss 屏蔽。
- **`L_depth`（不變）**：confidence-weighted + 多尺度梯度。（point 頭在純 pose scope 關掉。）
- **`L_gate`**：`BCE(σ(g), m*_patch)`，§5.1/§5.2 的軟標籤。`λ_g` 起始 1.0。
- **中括號兩項為可選擴充**（§7.1/§7.2），由 yaml 開關：`MultitaskLoss.forward` 只在對應 config block 存在時才加該項。兩者梯度**都只進預測 pose**，不碰幾何頭 → 維持 v3「不動 depth」的 scope。

**gate 品質一律用 AUC/F1 判斷，不看 BCE**：`m*_patch` 是 average-pool 的 soft 標籤，boundary patch 有不可約 floor，val BCE 會看似 overfit 卻與高 AUC 並存。

**`valid_frame` 建議維持原本寬鬆的「有效點數 > 100」，不要改成嚴格的「靜態點數 > 100」**：後者會在多動態場景造成幀數不足，而 pose 監督是直接的、得不到對等好處。

### 7.1 額外設計：Static-Photo Loss（靜態區跨幀光度一致，"route B"）

給 pose 一個**獨立於 `L_cam` 的監督**——不是 GT pose 的重參數化，而是**真實影像內容**。

```
GT 靜態點 —(GT depth 反投影)→ 相機系 —(模型『預測』相對 pose)→ frame t+1 —(GT 內參投影)→ grid_sample 取 RGB
                                           ↓  取樣到的 RGB 必須匹配 source pixel（Huber loss）
```
- **只限 GT 靜態區**（`m*_inst < dyn_thresh`）+ 有效深度：動態點即使 pose 完美也違反剛性 warp，排除才 well-posed。
- **梯度只進『預測 extrinsics』**（經 `F.grid_sample` 的取樣 grid）；GT depth/內參/影像皆常數，**depth head 完全不碰**。
- **遮擋防護**：把 t+1 的 `motion_mask` 也 warp 過去，source 點落在 t+1 動態區則丟棄。
- 用 `pose_enc_list[-1]`。實作：[`compute_static_photo_loss`](../training/loss.py)；config 開關 `static_photo`（檔名 suffix `_photo`）。

**變體（消融 run 3'）：改用預測深度反投影。** 把 GT depth 換成 `predictions["depth"]` → 梯度同時進 **depth_head 和 camera_head**，變成「pose+depth 聯合光度精修」。取捨：
- ⨯ 重新引入 **depth-pose 耦合/不可辨識**：重投影誤差 `= f(pose, depth)`，可靠「移動 depth」抄捷徑而非修 pose（v2 診斷過的偷懶捷徑），且可能改壞已領先的深度——**與 v3「不動幾何頭」主張相衝**。
- ✓ 若要做，**必掛兩道保險**：(1) `L_depth`（對 GT depth）當錨，讓深度不能自由漂；(2) `static_photo` 權重壓小、當微調。
- 需改 loss code（換輸入來源、有效性 mask、預測深度 scale 對齊）。**列為獨立 ablation，不當預設**。

### 7.2 額外設計：Camera-Smooth Loss（相機軌跡平滑正則）

硬/動態序列上觀察到明顯的**軌跡「跳動」**（native VGGT-1B 也有，非 v3 回歸）。加一個**只作用在網路自身輸出**的平滑先驗（類 MonST3R 的 trajectory-smoothness，但當訓練 loss 而非 test-time 項）。

- 對**預測 pose 序列**罰 **2 階（加速度）不連續**：`accel = v[t+1] − v[t]`，`v = ΔT/Δt`，對 T 與四元數各算。
- **Δt 正規化**：訓練 clip 抽幀間距不規則、可能重複（`batch["ids"]` 為真實時序 index）→ 用 `ΔT/Δt` 而非 raw ΔT；`Δt=0`（重複幀）排除。
- **四元數半球 sign-fix**：差分前對齊 `q/−q`，避免 sign flip 被誤當大旋轉跳動。
- **FoV 不平滑**；多 refine stage 以 `γ` 加權；需 `S≥3`（故 `img_nums` min ≥ 4）。**不用 GT pose**，純自參照正則。
- 實作：[`compute_camera_smooth_loss`](../training/loss.py)；config 開關 `camera_smooth`（檔名 suffix `_smooth`）。

---

## 8. 訓練策略（課程 + 硬體適配）

VGGT 是強預訓練 baseline，**絕不從頭訓**。三階段漸進解凍。

### 8.1 三階段

| 階段 | 凍結 | 解凍訓練 | 開的 loss | 目的 |
|---|---|---|---|---|
| **S0** | aggregator 全部 + 原三頭 | 只 gate predictor | `L_gate` | 養門控信號；門控因 zero-init 仍 ≈no-op |
| **S1** | patch_embed + frame_blocks + 前段 global(0–7) + depth/point head | gate predictor + 後段 global(8–23) + camera head | `L_gate` + `L_cam` | 門控通電，camera 學會「只看靜態」 |
| **S2**（保幾何版）| patch_embed + frame_blocks + depth_head | 全部 global(0–23) + gate predictor + camera head | `L_cam` + `L_depth` + `L_gate` + 混 ~30% 靜態，lr 1e-5 | 端到端精修 |

**S1 一律在修正後前向（§4）下訓練。** gate_predictor 的品質是 `BCE(σ(g), m*)` 監督的，**與 §4 的 attention bug 無關** → 可從 **S0 checkpoint warm-start（沿用已訓練的 gate）**，只重訓 camera 路徑（後段 global + camera head）去學會利用「現在真的生效」的門控。舊 S1 checkpoint 的 camera head 是在 no-op 門控下適應的，重訓即可。

**S2 刻意不照字面「全網解凍」。** v3 賣點是「不動幾何頭、保留深度領先」，故：
- `patch_embed`（DINOv2）**永遠凍**——最強 pretrained 特徵、depth 領先的根，PO 小資料上以 1e-5 解凍 forgetting 風險最高、對 pose 無直接幫助。
- `frame_blocks`（frame attention）**凍住或極小 lr**——pose-motion 耦合在 **global attention**（camera-query 對 patch-key），**不在 frame attention**（逐幀、不跨幀）；動它=動 depth 的輸入特徵，風險大於收益。
- `depth_head` 凍住但 **`L_depth` 保留**：depth 梯度只流回 shared trunk（global blocks），把 trunk 釘住不漂 → 這是 S2 敢解凍 trunk 的防遺忘錨，也保證 depth 零退化。
- S2 真正解凍的是 **全部 global blocks + gate predictor + camera head**。

**是否做 S2、解凍多廣，等 S1（修正版）結果再定**：若 S1 的 `predicted` 已明顯往 `oracle` 靠 → 做上面精修；若 gate 生效後 predicted 仍 ≈ off → 瓶頸轉為 §4.2 全動態幀資訊不足，重心應移到 temporal 借鄰幀，而非加大解凍面。

### 8.2 硬體適配（單張 RTX 5090 32GB）

實機為**單張 RTX 5090（32GB）**。config 已實測 `img_nums` 可到 8。

**(a) 純 pose scope 的最大紅利：關掉用不到的 dense 頭**

| Head | 訓練需要 | 理由 |
|---|---|---|
| camera | ✅ | 主角 |
| depth | ✅ | 算 `f^cam` 殘差（§5.2 輸入通道）需預測深度 |
| point / track / flow / motion(DPT) | ❌ 關掉 | pose 不需；`g` 已當 mask |

關掉 point+track 即釋出一大塊顯存，幀數上限可往上推。

**(b) S0 快取凍結特徵**：S0 aggregator 全凍 → 不 backprop 穿過 aggregator → 不必存其 activation。把中段特徵**離線預算存盤**，S0 只訓 gate MLP → 顯存近零，可開長 sequence / 大 batch。

**(c) 機動槓桿（按性價比）**
1. 降解析度 518→364（P ~1369→~676，attention 記憶體約 ¼，幀數翻倍）；可低解析訓 gate、最後 518 微調。
2. gradient checkpointing（code 已開）、bf16 autocast（已開）維持。
3. 時序加大 stride 抽幀：更大 baseline、對 pose 更有利、幀數需求更低。
4. gradient accumulation 補足等效 batch。

### 8.3 整合訓練流程：gate-base 因子消融（runs 1–4）

在 §8.1 的 S0→S1 之上，把兩個擴充（camera-smooth+temporal、static-photo）設計成**因子消融**：以「gate→camera」為 base，各擴充**單獨疊上 base** 再合併，才能分別歸因每項對 pose 的**邊際貢獻**（而非混在一起無法拆解）。

**統一架構（四個 run 共用，關鍵）**：所有 run 都建成 `enable_gate=True + enable_temporal=True`。run 1 裡 temporal 仍 γ=0 身分、且凍住（等於沒有），但 checkpoint 帶著這些權重 → run 2/3/4 能直接載 run 1、不會 missing key。gate 與 temporal 都是身分 warm-start，故從 VGGT-1B / `s0_inst` 起步皆乾淨。

| run | base 上額外**解凍** | 額外 **loss** | warm-start | 測什麼 |
|---|---|---|---|---|
| **1（base）** | —（gate_predictor + global 8–23 + camera_head）| —（`L_cam + L_gate`）| `s0_inst` | 修正前向下，gate 對 pose 的效果（`predicted`→`oracle`？）|
| **2** | temporal_blocks | `camera_smooth` | run 1 | temporal + 平滑 對軌跡跳動/ATE 的邊際貢獻 |
| **3** | depth_head*（可凍）| `static_photo`（**GT depth**，純 pose）| run 1 | 獨立光度信號對 pose 的邊際貢獻（守 depth）|
| **3'** | **depth_head** | `static_photo`（**預測深度**）+ `L_depth` 錨 | run 1 | pose+depth 聯合光度精修（§7.1 變體，帶耦合風險）|
| **4** | temporal + depth_head | `camera_smooth` + `static_photo` + `L_depth` | run 1 | 全開的組合效果 |

\* run 3 用 GT depth 時 static_photo 梯度只進 pose，depth_head 可凍；run 3'（預測深度）**必須**解凍 depth_head 且開 `L_depth` 當錨（見 §7.1 變體）。

**凍結原則（四個 run 一致）**：`patch_embed`、`frame_blocks`、前段 global(0–7) 永遠凍；point/track head 關掉。每個 run 只在 base 解凍集上「加解凍」上表對應模組。

**評測與歸因**：每個 run 完成後跑 Sintel gate ablation（`eval/gate_bias_ablation.py`，看 `predicted` vs `off`/`oracle`）+ 軌跡可視化。run 1 是關鍵閘門——**gate 若在 base 就無效，先別急著疊 2/3/4**，回頭查 §4.2 資訊不足 / temporal 借鄰幀。2 vs 1、3 vs 1 給各自邊際量，4 給組合（是否正交相加或互相干擾）。

**收尾（可選）= §8.1 的 S2 保幾何版**：從 run 4 warm-start，解凍全部 global + gate + camera + temporal（patch_embed/frame_blocks 仍凍、depth_head 凍但 `L_depth` 開），混 ~30% 靜態、lr 1e-5、全 loss 同開。

---

## 9. 與 v1/v2 的關係與評測

- **保留**：temporal aggregator（貢獻①, v1，見 §6）—— 正交、未被診斷實證有問題，且 §4.2 借鄰幀需要它；整合流程由 run 2/4 訓練（§8.3）。
- **丟棄**：雙場 `X^can + m·Δ`、scene-flow 頭、`L_reproj` / `L_tsmooth` / `L_flow`、4D 全局對齊；**test-time polish 也一併移除**（純前饋優先，見 §2）。
- **新增**：中段 gate predictor、運動門控 attention、幾何接地監督；外加兩個可選 pose-only 擴充——static-photo loss（§7.1）、camera-smooth loss（§7.2）。

**評測聚焦 pose**：Sintel / PointOdyssey 的 ATE / RPE（相機軌跡），對標 MonST3R 與原版 VGGT；輔以動態 mask 的 AUC（跨域 PO→Sintel 是關鍵指標）與 depth 不退化的對照。

---

## 一句話總結 novelty

> Dyn-VGGT v3 把「動態屏蔽」從 VGGT 的 **loss 端搬到 attention 端**：以中段輕量 gate 在 global attention 裡讓 camera token **結構性地只聚合靜態 patch**，在源頭解除 pose-motion 耦合；門控信號以 **domain-invariant 的運動定義幾何標籤**作監督目標（`m*_inst`/`m*_raft`，殘差輸入通道列為未來強化），根治 PO 原生外觀 mask 的跨域崩塌——在不動幾何頭、保留深度領先的前提下，**純前饋**地拿回動態場景的相機 pose。
