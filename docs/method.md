# Dyn-VGGT 方法

> **動態與非理想場景下的前饋式相機 Pose** —— 從**前向**、**時序**、**穩健性**三個軸
> 解耦 pose 與「不該影響 pose 的東西」（物體運動、照明事件）。
>
> **本文件只寫機制，不寫數字也不下結論。**
> 「現在信什麼」看 [status.md](status.md)；每個 run 的身分與下場看 [experiments.md](experiments.md)；
> 數字看 [results/natural.md](results/natural.md)（自然場景）、[results/medical.md](results/medical.md)（醫學）、
> [results/sota.md](results/sota.md)（對外對照）。

## 命名

**Gate（motion-gated camera aggregation）這一版是方法的第一個正式版本。**
本文件之前的內部編號 v1 / v2（雙場重表示、學習式 mask）是**被診斷否決的前身**，
已封存在 [archive/checkpoints.md](checkpoints.md) §2，本文只在 §6 用一段交代。

> 📌 **對照舊文件**：本方法在 2026-08 之前的文件裡稱作 **v3**。
> code / config / checkpoint 的名字裡沒有版本號（2026-08-20 已統一成 `inst_<成分>`），
> 只有舊文件會出現「v3」，遇到時視為本文件描述的同一個東西。

---

## 0. 範圍

VGGT 的 depth 在動態場景已經領先，**唯一輸給 MonST3R 的是相機 pose**。
根因是 pose 與物體運動在 aggregator 的 global attention 裡**耦合**：camera token
把動態 patch 的運動「平均」進了 pose。

| | 機制 | 軸 | 一句話 |
|---|---|---|---|
| **①** | **Gate**（§3） | 前向 | 聚合 camera token 時，用 attention bias 結構性地排除不符相機運動的動態 patch |
| **②** | **Temporal + Camera-Smooth**（§4） | 時序 | 補上 VGGT 缺的純時間軸 attention，並要求輸出軌跡時序連貫 |
| **③** | **照明穩健性 L_inf**（§5） | 穩健性 | 約束「一幀的**外觀**能影響其他幀的**幾何**多少」 |

**範圍刻意鎖定純 pose 且純前饋**：不輸出 scene flow、不改幾何表示、
**不做 test-time 後處理**（對標 MonST3R 的 TTO 是刻意的取捨）。
depth / point 頭原樣、原則上不動，深度領先保留。

**成本軸是這個定位的核心宣稱，而且已經有資料**：同一台機器、64 幀一條序列，
Ours 7.2 s vs MonST3R 450.7 s（63 倍），gate 對推論時間量不到成本。
量測細節見 [final_video_pipeline.md](topics/video_pipeline.md)。

---

## 1. 問題定位

### 1.1 pose 在哪裡誕生

**每一幀的 token 佈局**（`Aggregator.forward`，[aggregator.py:317-321](../vggt/models/aggregator.py#L317-L321)）：

```
每張圖 → DINOv2 patch embed → patch tokens [P, C]
                                    │
        tokens = cat([ camera_token(1) , register_token(4) , patch_tokens(P) ], dim=1)
                        └─ idx 0 ─┘   └── idx 1..4 ──┘   └── idx 5.. ──┘
                                    │
                        patch_start_idx = 1 + num_register_tokens = 5
```

- `camera_token` / `register_token` 皆 `std=1e-6` 初始化。
- `dim=1` 的那個 `2` 是**第一幀特殊**（`slice_expand_and_flatten`，
  [aggregator.py:598](../vggt/models/aggregator.py#L598)）：index 0 只給 frame 0，index 1 給其餘 S−1 幀。
  因為 VGGT 的 pose 是**相對於第一幀**的，frame 0 需要一個可區分的 query token。

| token | idx | 誰讀它 | 下場 |
|---|---|---|---|
| **camera** | 0 | `camera_head`：`pose_tokens = tokens[:, :, 0]` → trunk 迭代 refine → `pose_enc` | **→ pose** |
| **register** | 1..4 | 沒有任何 head 讀 | attention 跑完就丟 |
| **patch** | 5.. | depth / point / track head 用 `patch_start_idx` 切出來 | → 幾何 |

### 1.2 為什麼 loss 端的 mask 改不了 pose

把 motion 只當 **loss 端**的 gate **改不了已經算出來的 pose**。
pose 在 global attention 裡就已經被污染；等到算 loss 時再屏蔽動態區，
只是不去罰它，並沒有把污染從前向拿掉。

**這正是 §3 的動機**：把動態屏蔽從 loss 端搬到 **attention 端**。

### 1.3 根因對照

| 根因 | 症狀 | 對應機制 |
|---|---|---|
| **前向耦合**：camera query 無差別聚合所有 patch key | pose 被動態物的運動平均進去 | ① Gate（§3） |
| **時序結構未用**：只有 frame / global attention | 逐幀獨立估計 → 軌跡跳動 | ② Temporal（§4） |
| **信任度不可表達**：global attention 沒有「這個 view 不可信」的說法 | 單一幀壞掉污染整段 | ③ L_inf（§5） |

三者正交：① 改前向、② 補結構、③ 加約束。可獨立開關、獨立消融。

---

## 2. 總覽

```
            輸入視頻 {I_t}  (S frames)
                  │
       ┌──────────▼───────────┐
       │  DINOv2 Patch Embed   │  (沿用，永遠凍)
       └──────────┬───────────┘
                  │
       ┌──────────▼──────────────────────────────────────┐
       │  Aggregator                                      │
       │    aa_order = [frame, temporal(②), global]       │
       │                                                  │
       │   block 0..k  ──► Gate Predictor g   (① §3.1)    │
       │                        │ (detach → bias)         │
       │   block k+1..23 global attention 套              │
       │                 camera-query 門控     (① §3.2)   │
       └──────────┬──────────────────────────────────────┘
           ┌──────┴───────┬──────────┐
           ▼              ▼          ▼
       Camera Head    Depth Head   Point Head
           │              │              │
           ▼              ▼              ▼
       pose_enc        depth        world_points     σ(g) = 動態 mask (免費副產物)
           │
           ├── L_cam            (VGGT 原生)
           ├── L_gate           (① §3.4)
           ├── L_camera_smooth  (② §4.3)
           └── L_inf            (③ §5.3)
```

| # | 機制 | 開關 |
|---|---|---|
| ① Gate | `model.enable_gate` + `loss.gate` |
| ② Temporal | `model.enable_temporal` + `loss.camera_smooth` |
| ③ L_inf | `corruption.enabled` + `loss.influence` |

`MultitaskLoss.forward` **只在對應 config block 存在時才加該項** → loss 純靠 yaml 開關。

---

## 3. 機制① Gate

> **在聚合 camera token 時，移除不符合相機運動的動態物體。**

### 3.1 中段 Gate Predictor —— `g` 怎麼產生

門控要在 aggregation **進行中**用到動態信號，但 motion 通常要讀**最終**特徵才得到 → 雞生蛋。
解法：在 aggregator 中段插一個輕量預測器，**邊聚合邊出門控信號**。

```
            中段 aggregator 的 contextualized patch token
            (block_iter == gate_block_iter = 7 完成後)
            tokens[:, :, patch_start_idx:, :]  →  [B, S, P_patch, C]   # 切掉 camera(1)+register(4)
                    │
                    ▼
            LayerNorm → Linear(C→C/4) → GELU → Linear(C/4→1)   # 末層 zero-init
                    ▼
              g  [B, S, P_patch]      # 每 patch 一個動態 logit
```

**輸入只有 patch token，沒有 concat 任何其他通道**（[aggregator.py:380-383](../vggt/models/aggregator.py#L380-L383)
呼叫點 + [`GatePredictor.forward`](../vggt/models/aggregator.py#L54)：`x = norm(patch_tokens)`）。
§3.6 描述的「幾何殘差輸入通道」**尚未實作**。

`g` 的三個去處：

1. **門控（detach）** → §3.2 的 attention bias。detach 確保 pose 梯度不反灌門控、
   避免「為 pose 好看而亂標動態」。
2. **監督（不 detach）** → §3.4 的 `L_gate`。
3. **輸出**：`σ(g)` 即動態 mask 副產物。

設計選擇：

- **解析度天然對齊**：`g` 在 patch 解析度，attention 的 key 本來就是 patch token → bias 直接套，零插值。
- **放中段（~1/3）**：太早特徵未成熟；太晚則後面沒幾個 global block 可被清洗。
- **warm-start**：末層 zero-init → 第 0 步 `g≡0` → `bias≡0` → 前向 byte-for-byte 等於 pretrained VGGT。

### 3.2 運動門控相機聚合 —— bias 怎麼施加

attention：`A = softmax(QKᵀ/√d + bias)`。**只**在 camera + register token 的 query 列，
對 patch token 的 key 加負偏置：

```
bias[special_query, patch_key_j] = min(0, softplus(0) − softplus(g_j))    # g_j detach
bias[其他所有位置]                = 0                                       # patch↔patch 完全不動
```

| patch j | g_j | bias | softmax 後權重 |
|---|---|---|---|
| 靜態 | →−∞ | →0 | 維持原樣 |
| init（zero-init） | 0 | **=0（精確）** | 逐位元 = pretrained VGGT |
| 動態 | →+∞ | →−∞ | →0（排除） |

**為什麼是這個函數**（三個性質同時要）：

- `−softplus(g) = ln P_靜態`，softmax 後等於「**按靜態機率加權**」——單邊（≤0）、soft、只壓不升。
- `+softplus(0)` offset：讓 `g=0` 時 bias **精確為 0**，zero-init 門控在 init 逐位元等於
  pretrained VGGT。純 `−softplus(g)` 在 g=0 給 −ln2，會把每個 patch key 相對 special key 減半。
- `min(0, ·)` 夾上界：拿回 **suppress-only**，避免 offset 引入正 bias。
- kink 在 g=0 不可微，但 **bias 走 detach、梯度不穿過它** → 折角無害、前向仍連續。

效果：camera token **結構上只聚合靜態 patch**；**patch↔patch attention 一字不改** →
depth / point 頭輸入特徵不變。

#### 3.2.1 bias 參考點的兩個變體（flag）

`softplus(0)` 這個參考點把 clamp 的 kink 放在 **σ(g)=0.5**，也就是
**gate 只對「相信它動的機率過半」的 patch 動手**。兩個 flag 可以改掉這件事：

| flag | bias | 意義 | 代價 |
|---|---|---|---|
| （預設） | `min(0, softplus(0) − softplus(g))` | kink 在 σ(g)=0.5 | 信心不到 0.5 的 patch 完全不被壓 |
| `gate_leaky=λ>0` | clamp 的平段改成斜率 λ | 只是輕微可疑的 patch 不再與確定靜態的 patch 無法區分；`λ=1.0` 等於完全拿掉 clamp | 無 warm-start 損失（`<=` 而非 `<`，g=0 處梯度不被砍） |
| `gate_bias_zero_ref=True` | `−softplus(g)` | 參考點改成 `softplus(−∞)=0`，bias 處處 ≤0，clamp 變 no-op，隨證據平滑變化 | **破壞 warm-start**：g=0 時 bias 是 −log(2) 而非 0，起手就不是 pretrained VGGT。eval-only 使用不受影響 |

`gate_leaky=1.0` 與 `gate_bias_zero_ref` 只差一個常數 `log(2)`——patch **之間**的排序完全相同，
差別只在相對 camera / register key（那格 bias 恆為 0）。

> 這兩個 flag 是為了處理「gate 欠自信 → 落在 kink 死區 → 結構性失效」而加的，
> 實測效果見 [status.md](status.md) A.2。

### 3.3 實作關鍵：attention 拆兩條 path

把整個 `[N,N]` 浮點 mask 丟給 `F.scaled_dot_product_attention` 會讓 flash kernel
fallback 到 memory 重的 math backend → 顯存暴增。正解：

- **patch↔patch**（佔 (S·P)² 大頭）：維持原 flash / memory-efficient attention，**不帶 bias**。
- **camera/register query → 所有 key**（只有 ~`patch_start_idx·S` 個 query）：單獨小 op，帶 bias。

→ gate 本身顯存增加 ≈ 0，推論時間也量不到。

> ⚠️ **正確性陷阱（2026-07-09 已修，commit `e064087`）**：global attention 的 token 是
> **frame-interleaved** `[frame0(special+patch) | frame1(special+patch) | …]`，
> special token **散在每幀開頭**、不是集中在最前面。兩條 path 的切分必須逐幀 gather
> special query（`q.view(B,H,S,P,D)[:,:,:,:patch_start_idx]`），key bias 也照 interleaved 順序組。
> 舊版用 `q[:,:,:n_special]` 誤把「最前 n_special 個」當 special → 那些索引全落在 frame 0 內
> → **只 gate 到 frame 0，其餘幀完全沒作用**。這是 2026-07-06 之前所有 gate 結論作廢的原因。

### 3.4 門控監督：`L_gate` 與 `m*` 標籤

**跨域崩塌的真正原因是「PO 原生標籤有問題」，不是學習式 mask 本身的通病。**
PO 的 `masks/` 是逐 instance 的**外觀分割**（非黑=前景）：室內連牆、地板、idle 前景都被標
→ 這是**外觀 mask 不是運動 mask**。拿它監督，門控學到的是「PO 室內外觀→動態」，換到 Sintel 自然崩。

**解法：改用離線計算的「運動定義」幾何標籤。** 標籤在 PO 與 Sintel 是同一物理量
（世界真的在動），跨域即不再崩。

```
L_gate = BCE( σ(g), m*_patch )      # gate_logits 不 detach → 梯度回流 predictor
```

**`m*` 命名**：

| 名稱 | 是什麼 | 用在哪 |
|---|---|---|
| `m*_inst` | instance × 3D scene-flow 硬 mask | **PO 訓練標籤，實際採用** |
| `m_geo` | 幾何光流殘差 `‖f^gt − f^cam‖` 的硬門檻 mask（舊稱 `m*_raft`） | Sintel / 測試端 |
| `m*_patch` | 任一硬 mask average-pool 到 patch 格後的軟機率 | 真正餵 `L_gate` 的目標 |
| `m_geo_soft` | 同一殘差量的 sigmoid 軟目標 | §3.6，未實作 |

**產生方式**：

- **`m*_inst`**（`dynmask_inst/`）：用 GT world track（`trajs_3d`）判斷每個 instance
  連通塊是否在動，動則整塊填滿。**最穩**——靜態世界點位移恆為 0，無 RAFT 的背景灌爆與快慢兩難。
  `training/data/preprocess/po_instance_dynmask.py`；dataset 以 `dynamic_source="instance"` 載入。
- **`m_geo`**：`‖f^gt − f^cam(GT depth+pose 重投影)‖ > bar`。純幾何、無需 GT track。
  Sintel 用 GT `.flo` 光流（`data/motion_mask.py`，落盤快取到 `<sintel_root>/mask/`）；
  其他測試集用 RAFT 估計光流。Sintel 端用**相對門檻** `bar = max(abs, α·‖f^gt‖)`
  解掉快速鏡頭整片灌爆。

**分工**：有 GT track（PO 訓練）用 `m*_inst`；無 GT track（Sintel / 測試）用 `m_geo`。
兩者語義一致——**這就是跨域不崩的來源**。

> 📌 **不要用 (a)/(b) 這種字母指涉這兩個標籤。** 舊文件與程式註解的字母歷來相反
> （程式的 (a) 是 `m_geo`，文件的 (a) 是 `m*_inst`），2026-08-31 已把程式端全部改成寫名字。

**兩個操作準則**：

- **品質判斷一律用 AUC / F1，不看 BCE**。pool 後的軟標籤在 boundary patch 有不可約 floor，
  val BCE 會看似 overfit 卻與高 AUC 並存。
- **hard label + ignore band 變體**（`m*_patch ≥ 0.7 → 1`、`≤ 0.3 → 0`、中間不算 loss）
  已實作於 `compute_gate_loss`，用意是砍掉 boundary floor。**這是 null result**，
  見 [experiments.md](experiments.md) §6.1。

### 3.5 全動態幀的欠定問題（潛在風險，未實證）

理論上若某幀幾乎全動態，門控排除絕大多數 patch → camera token 無料可聚合 → pose 可能**欠定**
（資訊不足，非監督不足）。預留對策：

1. **soft bias**（現行 bias 已是 soft，非 `−∞`）：全動態時仍保留微弱全局信號。
2. **top-k floor**：每幀至少保留 `g` 最小的 k 個 patch。
3. **temporal attention 借鄰幀**：從較靜態的相鄰幀取幾何上下文（§4 的價值之一）。

> 這個問題在 §3.3 的 bug 修正前曾被誤判為真正瓶頸；修正後已知不成立。此處僅保留作風險記錄。

### 3.6 未實作的強化（記錄備考）

給 gate predictor 加一路**幾何殘差輸入通道** `‖f^gt − f^cam‖`——用預測 depth + 預測 pose
重投影得到的 ego-flow 與觀測光流的差。動機是讓 predictor 直接看到「運動」這個物理量
而非只看外觀特徵。**未實作**，目前 predictor 的輸入只有 patch token。

---

## 4. 機制② Temporal Attention 與軌跡平滑

### 4.1 Temporal Aggregator

VGGT 原生 aggregator 只有兩種 attention：**frame**（單張影像內的空間注意力）與
**global**（所有幀所有 token 攤平）。缺一種「**純時間軸**」——讓同一空間位置沿它自己的 S 幀互看。

- **開關**：`enable_temporal=True` → `aa_order` 由 `[frame, global]` 變 `[frame, temporal, global]`。
  每 `temporal_every=3` 個 aa-block 插一個 → `n_temporal = 24 // 3 = 8` 個。
- **時間軸 attention**：tokens reshape 成 `(B·P, S, C)`，attention 沿 **S** 跑，
  算完 reshape 回 `(B·S, P, C)`。
- **不改 head 介面**：temporal block **只更新 streaming tokens，不 emit intermediate**
  → head 輸入維持 `[B,S,P,2C]`（frame+global 串接）。
- **獨立時間 RoPE**：1D `RotaryPositionEmbedding1D`，與空間 2D RoPE 分開 → 不動 pretrained
  空間位置編碼。camera token 與 patch token 給真實 frame index；register token 給 0。

### 4.2 Warm-start

每個 temporal block 的 **LayerScale γ=0** → init 時整塊是**身分映射** → 加了 temporal
仍能 byte-for-byte 載 VGGT-1B。訓練中 γ 漸長、temporal 漸進通電。

> 這帶來一個有用的性質：config 可以「建了 temporal 但凍住」，效果等於沒有，
> 但 checkpoint 帶著這些權重 → 後續 run 能直接載、不會 missing key。
> `scared_cam_b16` 就是靠這個當「架構等同原版」的對照組。

### 4.3 Camera-Smooth Loss

硬 / 動態序列上觀察到明顯的**軌跡跳動**（native VGGT-1B 也有，非本方法的回歸）。
加一個**只作用在網路自身輸出**的平滑先驗（類 MonST3R 的 trajectory-smoothness，
但當訓練 loss 而非 test-time 項）。

- 對**預測 pose 序列**罰 **2 階（加速度）不連續**：`accel = v[t+1] − v[t]`，`v = ΔT/Δt`，
  對 T 與四元數各算。
- **Δt 正規化**：訓練 clip 抽幀間距不規則、可能重複（`batch["ids"]` 為真實時序 index）
  → 用 `ΔT/Δt` 而非 raw ΔT；`Δt=0`（重複幀）排除。
- **四元數半球 sign-fix**：差分前對齊 `q/−q`，避免 sign flip 被誤當大旋轉跳動。
- **FoV 不平滑**；多 refine stage 以 `γ` 加權；需 `S≥3`（故 `img_nums` min ≥ 4）。
- **不用 GT pose**，純自參照正則。

> **為何與 temporal 同節**：兩者都在利用「影片有時序結構」——temporal 是**機制**
> （讓網路能跨幀看），camera_smooth 是**目標**（要求輸出時序連貫）。
> 但 camera_smooth **不需要 temporal 才能算**
> → **兩者疊在一起跑出來的增益必須再拆一次才能歸因**。

---

## 5. 機制③ 照明穩健性：Photometric Corruption + L_inf

> ⚪ **本節只描述機制。這條線目前沒有結論**，實驗狀態見 [status.md](status.md) C。

### 5.1 動機

內視鏡影片有一種 VGGT 沒有機制可以吸收的失效：**單一幀**壞掉
（AWB 重鎖整片變綠、gain 跳階、specular blob 飽和），**整段**重建就跑掉。
這不是漸進的精度問題，是**路由問題**：global attention 讓每一幀都能 attend 到那一幀的
patch token，而架構裡沒有任何地方能表達「這個 view 不可信」——
**frame 0 最嚴重，因為它定義了參考座標系。**

### 5.2 Photometric Corruption

`training/data/photometric_corruption.py`。**唯一的硬規則：幾何永遠不動。**
沒有 warp、crop、resize、flip、shift——只有 per-pixel 的值映射。
因為 §5.3 要逐像素比較兩次前向，兩者必須保持 pixel-aligned；
這也正是讓 L_inf 的有限差分**只張成外觀方向**的原因。

**兩類，不可互換**：

| 類 | 內容 | 性質 | 用途 |
|---|---|---|---|
| **C1** | color_cast、exposure、gamma、light_field、contrast | **保訊息**：內容在映射後仍在，幾何原則上仍可還原 | **L_inf 只用這類** |
| **C2** | specular（飽和裁掉）、haze、blur、noise | **毀訊息**：那一幀真的不再攜帶幾何 | 保留給 reliability-gate 與 held-out 穩健性評估 |

**兩個物理細節**：

- **乘性光照在線性空間作用**。內視鏡影格是 gamma-encoded；光源靠近是放大 radiance 而非
  sRGB code value。所以 gain 與 light field 要 round-trip 過線性光，
  而 tone-curve 類（gamma、contrast）直接作用在編碼值上——那才是它們物理上發生的地方。
- **severity 是雙峰而非均勻**。真實失效是**罕見但嚴重**。均勻的輕微抖動是在訓練一個不會發生的
  分佈，而把真正會弄壞重建的尾部留著沒練過。

**train C1 / test C2 就是這條線的泛化測試**：一個對沒訓練過的 corruption 也能優雅退化的
gate，學到的是「不可信」而不是「綠色」。

### 5.3 L_inf —— 約束的是 Jacobian，不是輸出值

每個 train step 前向兩次：

- **teacher**：乾淨序列，`no_grad`。`no_grad` 是成本能壓在 ~1.3× 而非 2× 的原因——
  不存 activation。
- **student**：同一序列，1–2 幀被 photometric corruption 弄壞。**`L_sup` 算在 student 上**，
  所以 corruption 同時就是 augmentation。

```
L_inf = mean_{s 未被 corrupt} ‖ Y_s(I 的第 k 幀壞掉) − sg[ Y_s(I) ] ‖₁
```

**為什麼這不是 L_sup 的重述。** `L_sup` 只能說「每一幀的輸出要接近 GT」，
它沒有任何辦法說「第 s 幀的輸出**不可以依賴**第 k 幀的外觀」。
後者是關於 **Jacobian** 的陳述，而上面那個差就正好是
`dY_s / d(I_k 的外觀)` 的有限差分。因為 corruption 只改像素值、不含幾何成分，
**探測方向只張成 nuisance 子空間**——它無法要求模型丟掉第 k 幀的**幾何**貢獻，
只能丟掉外觀引起的那一份。multi-view 資訊因此完整保留。

**被 corrupt 的幀刻意排除在總和外。** 一幀剛被塗綠，它**本來就該**預測得比較差——
資訊真的被拿掉了。不該發生的是它把鄰居一起拖下水，那才是這一項要擋的失效。

**退化解與擋它的東西。** 單獨看，L_inf 的最小值是一個完全忽略輸入、輸出常數的模型。
兩件事擋在前面：

1. **point head 必須同時受 `L_sup` 監督**（`loss.point` 不可為 null——開了 head 卻不監督，
   等於直接把捷徑交給 L_inf）。
2. 每次只有 S 幀裡的 1–2 幀被 corrupt，塌成單視角重建會在每一條序列上付出精度代價、永遠不划算。

**兩個 channel**：`use_point`（`world_points`，在 frame-1 座標系，是唯一同時攜帶
depth 與 pose drift 的輸出）與 `use_pose`（`pose_enc_list[-1]`）。
兩者尺度差一個數量級，`w_pose` 要另外調。

**數值處理**（這不是防禦性編碼，是修一個實測到的失效，見 [experiments.md](experiments.md) §6.5）：

- point channel 套用與 `compute_point_loss` **相同的** quantile filter（`valid_range=0.98`）。
  兩個 loss 對 outlier 的定義必須一致，否則沒被監督的像素會在 filter 背後無人看管地長大，
  而 L_inf 照單全收。
- 除數用 teacher 點圖的 **median 而非 mean**。teacher 跑在 `no_grad` 下，
  這個除數不帶梯度、無法靠灌大點圖來作弊。

### 5.4 訓練端的接線

`trainer._step`。三個必須知道的實作點：

- **只在 train phase**。validation 必須維持單次前向，否則它的 loss 就不再能跟過去所有 run 比。
- **teacher 的 autocast 必須 `cache_enabled=False`**。autocast 會快取權重的 bf16 轉型；
  teacher pass 會把快取填滿，student pass 就會吃快取而不是自己產生，
  於是 gradient checkpointing 在 backward 重算時存到的 tensor 與 forward 不同 →
  `CheckpointError: Recomputed values ... have different metadata`。
  只對 teacher 關快取，student pass 與單次前向的情況 byte-identical。
- **必須 log `corrupt_frac`**。否則 L_inf 下降時讀不出來是「模型變穩健了」還是
  「這幾步剛好比較少幀被弄壞」。

**`loss.influence.weight: 0` 就是 augmentation-only 的消融組**，其他什麼都不變。
**在跑它之前不能對 L_inf 下任何結論。**

---

## 6. 已收掉的機制（保留記錄）

| 機制 | 是什麼 | 為什麼收 |
|---|---|---|
| **v1 雙場** | `X = X^can + m·Δ`，canonical 場 + 動態位移場 | 雙線性不可辨識性 + 動態區偷懶捷徑 |
| **v2 學習式 mask** | 從外觀學動態 mask | 跨域崩塌（PO AUC 0.88 → Sintel 0.45）。**§3.4 的幾何標籤就是這個問題的解** |
| **Motion / static-photo loss** | 靜態區跨幀光度一致：GT 靜態點以**預測 pose** warp 到鄰幀，RGB 必須匹配 | clean 重跑是負結果，見 [experiments.md](experiments.md) §6.2 |
| **`L_ego_flow`** | 像素空間 ego-flow 重投影一致性 | 陰性收束。低視差序列任何 flow loss 都無效，見 [ego_flow.md](topics/ego_flow.md) |
| **test-time polish / 4D 全局對齊** | v1 的後處理 | 與「純前饋」的定位衝突，刻意移除 |

> static-photo 的完整機制描述（含遮擋防護與「改用預測深度」的變體及其耦合風險）
> 保留在 git history 的 `docs/method.md`（commit `092445e` 之前的版本）。

---

## 7. 損失函數總表

```
L = L_cam + L_depth [+ λ_g·L_gate] [+ λ_s·L_camera_smooth] [+ λ_i·L_inf] [+ L_point]
      └ VGGT 原生        └ ①              └ ②                   └ ③
```

| 項 | 內容 | 梯度去哪 |
|---|---|---|
| `L_cam` | 逐分量 L1（T / q / FoV），跨 refine stage 以 `γ^{n−i−1}` 加權 | camera 路徑 |
| `L_depth` | confidence-weighted + 多尺度梯度 | depth_head（凍結時只當 trunk 錨） |
| `L_gate` | `BCE(σ(g), m*_patch)`，§3.4 | gate_predictor（**不** detach） |
| `L_camera_smooth` | 2 階軌跡平滑，§4.3 | **只**進預測 pose |
| `L_inf` | §5.3 | student 前向全體（teacher 是 `sg[·]`） |

**三個必須記住的陷阱**：

- **`L_cam` 不經像素** → 動態像素不會污染 pose 的 loss。pose 的提升來自
  **§3 的前向門控**，不是 loss 屏蔽。這是 §1.2 的另一面。
- **`L_depth` 的 confidence 項有一個陷阱**：`L_conf = γ·l_reg·c − α·log(c)`，
  最佳解 `c* = α/(γ·l_reg)` 會在 `l_reg → 0` 時把 `L_conf` 推向 −∞。
  對從頭訓的 head 無害，但對**訓練得很好的 pretrained head** 會從第 0 步就主宰目標，
  讓最便宜的下降方向變成「灌大信心」。實際踩過，見 [experiments.md](experiments.md) §6.4。
- **`valid_frame` 維持寬鬆的「有效點數 > 100」**，不要改成「靜態點數 > 100」：
  後者會在多動態場景造成幀數不足，而 pose 監督是直接的、得不到對等好處。

---

## 8. 訓練策略

VGGT 是強預訓練 baseline，**絕不從頭訓**。

### 8.1 漸進解凍

| 階段 | 凍結 | 解凍 | 開的 loss | 目的 |
|---|---|---|---|---|
| **S0** | aggregator 全部 + 三頭 | 只 gate predictor | `L_gate` | 養門控信號；門控因 zero-init 仍 ≈ no-op |
| **S1** | patch_embed + frame_blocks + global(0–7) + depth/point head | gate predictor + global(8–23) + camera head | `L_gate` + `L_cam` | 門控通電，camera 學會只看靜態 |
| **S2** | patch_embed + frame_blocks + depth_head | 全部 global + gate predictor + camera head | 全開 + 混 ~30% 靜態，lr 1e-5 | 端到端精修 |

gate_predictor 的品質由 `L_gate` 監督，**與 §3.3 的 attention bug 無關**
→ 可從 S0 checkpoint warm-start，只重訓 camera 路徑。

**永久凍結原則**：

- `patch_embed`（DINOv2）**永遠凍**——最強 pretrained 特徵、depth 領先的根。
- `frame_blocks` 凍住或極小 lr——pose-motion 耦合在 **global attention**，不在 frame attention。
- `depth_head` 預設凍住但 `L_depth` 保留：depth 梯度只流回 shared trunk，把 trunk 釘住不漂
  → 防遺忘錨。**解凍它是一個獨立的實驗因子，而且踩過雷**（§7 的第二個陷阱）。

### 8.2 硬體適配（單張 RTX 5090 32 GB）

- **關掉用不到的 dense 頭**：純 pose scope 下 point / track / flow / motion 頭關掉。
  （L_inf 那條線是例外——它**需要** point head，見 §5.3。）
- **真正的 VRAM 旋鈕是 `img_nums` 的上界，不是 `max_img_per_gpu`。**
  `max_img_per_gpu` 只在 `img_nums` 上界低於它時才綁得住；最壞情況永遠是 `max(img_nums)`。
- gradient checkpointing（已開）、bf16 autocast（已開）。
- 時序加大 stride 抽幀：更大 baseline、對 pose 更有利。
  SCARED 上這是一個可調參數 `nearby_expand_range`——實測 GT 相機位移在 gap 240 之後
  **飽和**（內視鏡會逗留與折返），所以更寬的窗買到 baseline 的同時會賠掉視角重疊。

### 8.3 config 繼承

config 之間用 Hydra `defaults` 串成繼承鏈，**只寫 delta，不整份複製**。
⚠️ **Hydra 對 list 是「取代」不是「合併」**——覆寫 `frozen_module_names` 或
`gradient_clip.configs` 時要把整份清單重寫（兩者語意都與順序無關，可安心重排）。

---

## 9. 評測協定

> ⚠️ **本節的問題會影響所有數字的讀法。**

### 9.1 兩個 channel

| 層 | 資料 | 訊號 | 決定性？ |
|---|---|---|---|
| channel A（windowed val） | 隨機抽 4~16 幀 | loss + ATE | ❌ seed 每 epoch 換 → 變動 = 模型 + 資料，分不開 |
| channel B（pose_eval） | 全序列 | ATE | ✅ 完全決定性，**= 報告指標** |

**核心缺陷：自然場景線沒有任何 held-out 資料。** Sintel 的 14 seq 被 val、pose_eval、
報告、以及所有 `diag/` 工具共用。**但權重是乾淨的**——訓練資料是 PO / TartanAir / Waymo / Spring，
模型從未在 Sintel 上收過梯度。**問題純粹是 checkpoint selection，不是梯度洩漏。**

選擇偏差已用 leave-one-out 量過：參與選擇的 14 seq 上 −12.5%、held-out seq 上 −11.5%
→ **偏差約 1 個百分點**，且選擇強烈泛化（14 個 fold 有 13 個選到同一個 epoch）。

### 9.2 雜訊帶

run 1 的 24 個 epoch（channel B）：`mean=0.1752, std=0.0149`。

- **單次比較的 2σ 門檻 = 16.9%** —— 小於此的「改善」不可信。
- **雜訊集中在單一序列**：`market_5` 逐 epoch 在 0.093~0.800 之間擺盪 **8.6 倍**，
  **獨佔 83.9% 的變異**。對比 `cave_2` mean 1.148 但 CV 只有 4.7% —— **難 ≠ 不穩**。

### 9.3 chunk_size 陷阱

**任何跨幀指標都必須整段一次 forward（`chunk_size=0`）。**
獨立 chunk 拼接會讓接縫主宰 ATE 且數字不再隨模型變化。
權威說明在 `eval_utils/vggt_infer.infer_sequence_chunked` 的 docstring。

**推論：SCARED 的全序列 evo ATE 我們不報。** VGGT 單次上限 80 幀，Sim3 拼接讓 ATE
擺動 −1%~+32% 且與 seam 數無關、不可預測。**是不估，不是估錯**——代價是
EndoSfM3D 那一系的 pose 欄填不了，我們只對得上 AF-SfMLearner 的 snippet 協定。

### 9.4 與 MonST3R 比較的前提

**MonST3R 是 test-time optimization（分鐘級 / 序列），本方法是單次前饋（秒級）。**
raw ATE 直接對打是拿一次前向打一個最佳化迴圈。成本軸的實測數字見 §0 與
[final_video_pipeline.md](topics/video_pipeline.md)。

---

## 10. 文件 ↔ 程式對照表

改了哪一邊，另一邊要跟著看。

| 機制 | flag / config key | 程式位置 |
|---|---|---|
| GatePredictor | `model.enable_gate`、`model.gate_block_iter`（預設 7） | `vggt/models/aggregator.py`：`GatePredictor`、`Aggregator.forward` |
| gate attention bias | — | `aggregator.py`：`_process_global_attention` / `_gated_global_block_forward` |
| bias 參考點變體 | `model.gate_leaky`、`model.gate_bias_zero_ref` | `aggregator.py`：`Aggregator.__init__` + bias 計算處 |
| gate 讓 pose loss 訓練 | `model.gate_pose_grad` | `aggregator.py` |
| gate 輸出與 override | — | `vggt/models/vggt.py`：`predictions["gate_logits"]`、`gate_logits_override` |
| `L_gate` | `loss.gate` | `training/loss.py`：`compute_gate_loss` |
| oracle mask → override | — | `training/loss.py`：`oracle_gate_logits_from_mask` |
| Temporal attention | `model.enable_temporal`、`temporal_every` | `aggregator.py`：`_process_temporal_attention`、`temporal_blocks` |
| `L_camera_smooth` | `loss.camera_smooth` | `training/loss.py`：`compute_camera_smooth_loss` |
| Photometric corruption | `corruption.*` | `training/data/photometric_corruption.py`：`corrupt_batch` |
| `L_inf` | `loss.influence` | `training/loss.py`：`compute_influence_loss` |
| 雙前向（teacher/student） | `corruption.enabled`、`corruption.warmup_steps` | `training/trainer.py`：`_step` |
| 動態標籤產生 | `dynamic_source=` | `training/data/preprocess/po_*.py`、`training/data/motion_mask.py` |
| SCARED 抽樣窗 | `nearby_expand_range` | `training/data/datasets/scared.py` |

**評估入口**（契約要穩定，別亂改名）：

| 目的 | 入口 |
|---|---|
| gate 的**唯一**評估入口（quality + pose ablation） | `training/diag/gate_eval.py` |
| Sintel pose + depth 主表 | `training/benchmark/eval_sintel.py` |
| SCARED（對齊 EndoSfM3D / AF 協定） | `training/benchmark/eval_scared.py` |
| 私人資料集 | `training/benchmark/eval_lesion.py` / `eval_gastric.py` |
| MonST3R 對照 | `training/benchmark/eval_monst3r_*.py` |
| 拼接誤差本身 | `training/diag/stitch_error.py` |
| L_inf 尺度診斷 | `training/diag/influence_scale_probe.py` |
| specular 佔多少可匹配紋理 | `training/diag/specular_texture_probe.py` |
