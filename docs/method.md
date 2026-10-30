# Dyn-VGGT 方法

> **本文件只寫機制，不寫數字也不下結論。**
> 結論、訓練設定、評估標準看 [README.md](README.md)；數字看 [results/scared.md](results/scared.md)。
> 2026-10-08 改寫：只留現在主線用到的四個機制（gate、temporal、camera_smooth、`flow_geom`）。
> 照明穩健性 L_inf、已收掉的機制、自然場景的評測細節在 [archive/method_2026-10-08.md](archive/method_2026-10-08.md)。

## 命名

Gate（motion-gated camera aggregation）是方法的第一個正式版本。2026-08 之前的文件稱它 **v3**，
code / config / checkpoint 的名字裡沒有版本號，遇到「v3」時視為本文件描述的同一個東西。

---

## 0. 範圍

VGGT 的 pose 與場景運動在 aggregator 的 global attention 裡**耦合**：camera token
把動態 patch 的運動「平均」進了 pose。本方法用四個機制處理這件事：

| | 機制 | 改哪裡 | 一句話 |
|---|---|---|---|
| ① | **Gate**（§3） | 前向 | 聚合 camera token 時，用 attention bias 壓低動態 patch 的權重 |
| ② | **Temporal**（§4.1） | 前向 | 補上 VGGT 缺的純時間軸 attention |
| ③ | **Camera-Smooth**（§4.3） | loss | 要求輸出軌跡二階平滑 |
| ④ | **`flow_geom`**（§5） | loss | 要求光流對應的兩個像素反投影到同一個 3D 點 |

**範圍鎖定純前饋**：不做 test-time 後處理，不輸出 scene flow，point / track 頭關閉。

---

## 1. 問題定位

### 1.1 pose 在哪裡誕生

**每一幀的 token 佈局**（`Aggregator.forward`，[aggregator.py](../vggt/models/aggregator.py)）：

```
每張圖 → DINOv2 patch embed → patch tokens [P, C]
                                    │
        tokens = cat([ camera_token(1) , register_token(4) , patch_tokens(P) ], dim=1)
                        └─ idx 0 ─┘   └── idx 1..4 ──┘   └── idx 5.. ──┘
                                    │
                        patch_start_idx = 1 + num_register_tokens = 5
```

- `camera_token` / `register_token` 皆 `std=1e-6` 初始化。
- token 有兩組：index 0 只給 frame 0，index 1 給其餘 S−1 幀。VGGT 的 pose 相對於第一幀，frame 0 需要可區分的 query。

| token | idx | 誰讀它 | 下場 |
|---|---|---|---|
| **camera** | 0 | `camera_head`：`pose_tokens = tokens[:, :, 0]` → trunk 迭代 refine → `pose_enc` | **→ pose** |
| **register** | 1..4 | 沒有任何 head 讀 | attention 跑完就丟 |
| **patch** | 5.. | depth head 用 `patch_start_idx` 切出來 | → 幾何 |

### 1.2 為什麼 loss 端的 mask 改不了 pose

pose 在 global attention 裡就已經被污染；等到算 loss 時再屏蔽動態區，只是不去罰它，
並沒有把污染從前向拿掉。所以 gate 把動態屏蔽放在 **attention 端**（§3）。

---

## 2. 總覽

```
            輸入視頻 {I_t}  (S frames)
                  │
       ┌──────────▼───────────┐
       │  DINOv2 Patch Embed   │  (永遠凍結)
       └──────────┬───────────┘
                  │
       ┌──────────▼──────────────────────────────────────┐
       │  Aggregator                                      │
       │    aa_order = [frame, temporal(②), global]       │
       │                                                  │
       │   block 0..7  ──► Gate Predictor g   (① §3.1)    │
       │                        │                         │
       │   block 8..23 global attention 套                │
       │                 camera-query 門控     (① §3.2)   │
       └──────────┬──────────────────────────────────────┘
           ┌──────┴───────┐
           ▼              ▼
       Camera Head    Depth Head
           │              │
           ▼              ▼
       pose_enc        depth                σ(g) = 動態 mask（副產物）
           │              │
           ├── L_cam ─────┤
           ├── L_camera_smooth (③ §4.3)
           ├── L_flow_geom     (④ §5，同時吃 pose 與 depth)
           ├── L_depth
           └── L_gate          (① §3.4，只在有動態標籤的資料)
```

| # | 機制 | 開關 |
|---|---|---|
| ① Gate | `model.enable_gate`；監督方式見 §3.4 |
| ② Temporal | `model.enable_temporal` |
| ③ Camera-Smooth | `loss.camera_smooth` |
| ④ `flow_geom` | `loss.flow_geom` |

`MultitaskLoss.forward` **只在對應 config block 存在時才加該項** → loss 純靠 yaml 開關。

---

## 3. 機制① Gate

> **在聚合 camera token 時，壓低不符合相機運動的 patch。**

### 3.1 中段 Gate Predictor —— `g` 怎麼產生

門控要在 aggregation **進行中**用到動態信號，所以在 aggregator 中段插一個輕量預測器：

```
            中段 aggregator 的 patch token
            (block_iter == gate_block_iter = 7 完成後)
            tokens[:, :, patch_start_idx:, :]  →  [B, S, P_patch, C]
                    │
                    ▼
            LayerNorm → Linear(C→C/4) → GELU → Linear(C/4→1)   # 末層 zero-init
                    ▼
              g  [B, S, P_patch]      # 每 patch 一個動態 logit；σ(g) = 動的機率
```

- **輸入只有 patch token**，沒有其他通道（「幾何殘差輸入通道」的想法未實作）。
- **解析度天然對齊**：`g` 在 patch 解析度，attention 的 key 本來就是 patch token，bias 直接套。
- **warm-start**：末層 zero-init → 第 0 步 `g≡0` → `bias≡0` → 前向 byte-for-byte 等於 pretrained VGGT。

`g` 的去處：

1. **門控** → §3.2 的 attention bias。預設 detach（見 §3.4）。
2. **監督** → §3.4。
3. **輸出**：`σ(g)` 即動態 mask 副產物，在 `predictions["gate_logits"]`。

### 3.2 bias 怎麼施加

attention：`A = softmax(QKᵀ/√d + bias)`。**只**在 camera + register token 的 query 列，對 patch token 的 key 加負偏置：

```
bias[special_query, patch_key_j] = min(0, softplus(0) − softplus(g_j))
bias[其他所有位置]                = 0          # patch↔patch 完全不動
```

| patch j | g_j | bias | softmax 後權重 |
|---|---|---|---|
| 靜態 | →−∞ | 0 | 維持原樣 |
| init | 0 | **=0（精確）** | 逐位元 = pretrained VGGT |
| 動態 | →+∞ | →−∞ | →0（排除） |

**為什麼是這個函數**：

- `−softplus(g) = ln P_靜態`，softmax 後等於「按靜態機率加權」：單邊（≤0）、soft、只壓不升。
- `+softplus(0)` 讓 `g=0` 時 bias 精確為 0，zero-init 的 gate 在 init 時等於 pretrained VGGT。
- `min(0, ·)` 夾上界，避免 offset 引入正 bias。
- **代價：kink 在 σ(g)=0.5**。g<0 的 patch bias 恆為 0，梯度也是 0（§3.4 的死區）。

patch↔patch attention 一字不改 → depth head 的輸入特徵不受 gate 影響。

#### 3.2.1 bias 形式的變體（flag）

| flag | bias | 意義 | 代價 |
|---|---|---|---|
| （預設） | `min(0, softplus(0) − softplus(g))` | kink 在 σ(g)=0.5 | 信心不到 0.5 的 patch 完全不被壓 |
| `gate_leaky=λ>0` | clamp 的平段改成斜率 λ；`λ=1.0` 等於拿掉 clamp | g<0 的 patch 仍有梯度、仍可被輕微壓低 | 無 warm-start 損失 |
| `gate_bias_zero_ref=True` | `−softplus(g)` | 參考點改成 0，bias 處處 ≤0、隨證據平滑變化 | **破壞 warm-start**：g=0 時 bias 是 −log 2 |
| `gate_bias_scale=s`（`gate_bias_a`、`gate_bias_tau`、`gate_bias_learn`） | `−s·softplus(a·(g−τ))` | 可調強度、銳利度、門檻；`gate_bias_learn` 讓 s 可訓練 | 繞過 clamp、leaky 與 softplus(0) 參考點 |

### 3.3 實作：attention 拆兩條 path

把整個 `[N,N]` 浮點 mask 丟給 SDPA 會讓 flash kernel fallback 到 math backend → 顯存暴增。所以：

- **patch↔patch**（佔 (S·P)² 大頭）：維持 flash / memory-efficient attention，**不帶 bias**。
- **camera/register query → 所有 key**（只有 `patch_start_idx·S` 個 query）：單獨小 op，帶 bias。

→ gate 的顯存增加 ≈ 0。

> ⚠️ global attention 的 token 是 **frame-interleaved** `[frame0(special+patch) | frame1(special+patch) | …]`，
> special token 散在每幀開頭。切分必須逐幀 gather special query（2026-07-09 修，commit `e064087`；之前只 gate 到 frame 0）。

### 3.4 gate 的兩種訓練方式

| | 有動態標籤（自然場景：PointOdyssey 訓練） | 沒有動態標籤（SCARED） |
|---|---|---|
| 監督 | `L_gate = BCE(σ(g), m*_patch)`，`gate_logits` 不 detach | 沒有 `L_gate` |
| bias 的梯度 | detach（預設），pose loss 不訓練 gate | `gate_pose_grad=True`：拿掉 detach，由 camera loss 訓練 gate |
| 死區 | 不影響 `L_gate` 的梯度 | g<0 的 patch 從 bias 收不到梯度；用 `gate_leaky`（期中 `b16_gg` 用 0.1）留斜率 |
| 起點 | zero-init | 從 `inst_g` 接續時用 `reset_gate_head` 把末層歸零，讓 g 回到梯度最大的 g=0 |

**動態標籤 `m*`**（自然場景）：

| 名稱 | 是什麼 |
|---|---|
| `m*_inst` | instance × 3D scene-flow 硬 mask，PO 訓練標籤（`dynmask_inst/`） |
| `m_geo` | 幾何光流殘差 `‖f^gt − f^cam‖` 的硬門檻 mask（Sintel / 測試端） |
| `m*_patch` | 硬 mask average-pool 到 patch 格的軟機率，真正餵 `L_gate` 的目標 |

SCARED 沒有動態標註，標籤問題的現況見 [內視鏡整理.md](內視鏡整理.md) 測試 §2。

**判斷 gate 品質一律用 AUC / F1，不看 BCE**：pool 後的軟標籤在 boundary patch 有不可約 floor。

### 3.5 全動態幀的欠定問題（未實證）

若某幀幾乎全動態，門控排除絕大多數 patch → camera token 無料可聚合 → pose 可能欠定。
現行 bias 是 soft（非 −∞），temporal attention 也能從鄰幀借上下文。

---

## 4. 機制② Temporal 與 ③ Camera-Smooth

### 4.1 Temporal Aggregator

VGGT 原生只有 **frame**（單張影像內）與 **global**（所有幀所有 token 攤平）兩種 attention，缺「純時間軸」。

- `enable_temporal=True` → `aa_order` 由 `[frame, global]` 變 `[frame, temporal, global]`，每 `temporal_every=3` 個 aa-block 插一個 → 8 個新 block（100.8M 參數）。
- tokens reshape 成 `(B·P, S, C)`，attention 沿 S 跑：同一空間位置沿它自己的 S 幀互看。
- temporal block 只更新 streaming tokens，不 emit intermediate → head 介面不變。
- 獨立的 1D 時間 RoPE，不動 pretrained 的 2D 空間 RoPE。預設吃 clip 內位置 `arange(S)`；`temporal_frame_pos` 改吃真實幀間隔。

### 4.2 Warm-start

每個 temporal block 的 **LayerScale γ=0** → init 時是身分映射 → 仍能 byte-for-byte 載 VGGT-1B。
因此 config 可以「建了 temporal 但凍住」，效果等於沒有，但 checkpoint 帶著這些權重（期中 `b16` 就是這樣當對照）。

### 4.3 Camera-Smooth Loss

只作用在網路自身輸出的平滑先驗，**不用 GT pose**：

- 對預測 pose 序列罰 **2 階（加速度）**：`accel = v[t+1] − v[t]`，`v = ΔT/Δt`，對平移與四元數各算。
- **Δt 正規化**：clip 抽幀間距不規則（`batch["ids"]` 為真實幀號），用 `ΔT/Δt`；`Δt=0`（重複幀）排除。
- 四元數差分前做半球 sign-fix；FoV 不平滑；多個 refine stage 以 `γ` 加權；需 `S≥3`。

camera_smooth 不需要 temporal 也能算，兩者疊在一起的增益要分開跑才能歸因。

---

## 5. 機制④ `flow_geom`：跨幀 3D 一致性

> 改寫自 ColonAdapter 的 `geometry_loss`。實作：[loss.py `compute_flow_geom_loss`](../pipeline/training/loss.py)、[flow_mask.py `pair_flows`](../pipeline/training/flow_mask.py)。

**做什麼**：frame t 的像素 p 和它在 frame s 的光流對應點 p′ = p + flow(p) 是同一個表面點，
所以模型用 (depth_t, pose_t) 反投影出的世界點，必須和用 (depth_s, pose_s) 反投影出的世界點重合：

```
L_flow_geom = mean_{p 在 s 中可見} ‖ X_t(p) − X_s(p + flow_{t→s}(p)) ‖₁
X = R⁻¹ (K⁻¹ [u, v, 1]ᵀ · depth − T) / mean(depth_window)
```

- **光流**：RAFT，凍結、`no_grad`，在 `flow_proc` 解析度上算再放回原尺寸。
- **配對**：`src_offsets` 指 clip 內的相對位置（clip 依時間排序），±1 = clip 裡相鄰的兩張，不一定是影片上相鄰的幀。
- **可見性**：反向光流對 target 像素的覆蓋率 > `occ_thresh` 才算；可見像素少於 `min_valid` 的配對跳過。
- **尺度**：除以整個 window 的平均預測深度，loss 對全域尺度不變，不能靠縮小場景降低。
- **梯度**：同時流到兩幀的 depth 與 pose；光流與可見性遮罩是常數。
- **內參**：用 GT 內參，不用預測的 FoV。

**為什麼需要**：camera loss 只約束 pose、depth loss 只約束各幀自己的 depth，沒有任何一項直接要求不同幀的**幾何**彼此一致。

**限制**：

- 罰的是**不一致**，不是**錯誤**：所有幀錯得一樣時 loss 為 0。ColonAdapter 另有一項 point-map/pose 錨定，這裡沒搬（沒有 point head）；錨定由 camera loss 與 depth loss 提供。
- 依賴 RAFT 在內視鏡上的品質；低視差、反光區的光流可能不可靠。

---

## 6. 損失函數總表

```
L = λ_c·L_cam + λ_d·L_depth [+ λ_s·L_camera_smooth] [+ λ_f·L_flow_geom] [+ λ_g·L_gate]
```

| 項 | 內容 | 梯度去哪 |
|---|---|---|
| `L_cam` | 逐分量 L1（T / q / FoV），跨 refine stage 以 `γ^{n−i−1}` 加權；只在 valid frame（有效深度點 > 100） | camera 路徑 |
| `L_depth` | confidence-weighted regression + 多尺度梯度 | depth head（凍結時只當 trunk 錨） |
| `L_camera_smooth` | §4.3 | 只進預測 pose |
| `L_flow_geom` | §5 | 兩幀的 depth 與 pose |
| `L_gate` | §3.4，只在有動態標籤時 | gate predictor（不 detach） |

**三個陷阱**：

- **`L_cam` 不經像素** → 動態像素不會污染 pose 的 loss；pose 的改善若來自 gate，只能來自前向門控。
- **`L_depth` 的 confidence 項**：`L_conf = γ·l_reg·c − α·log(c)`，對訓練得很好的 pretrained head，
  `−α·log(c)` 會從第 0 步主宰目標（α 0.2 時 objective 變負）。SCARED 解凍 depth head 時用 α 0.02。
- **`L_cam` 的 valid frame 只看 clip 的第 0 幀**：第 0 幀深度太稀疏時整個 sample 的 camera loss 消失。SCARED loader 只從深度夠密的幀抽錨點。

---

## 7. 文件 ↔ 程式對照表

| 機制 | flag / config key | 程式位置 |
|---|---|---|
| GatePredictor | `model.enable_gate`、`model.gate_block_iter`（預設 7） | `vggt/models/aggregator.py`：`GatePredictor`、`Aggregator.forward` |
| gate attention bias | — | `aggregator.py`：`_process_global_attention` / `_gated_global_block_forward` |
| bias 形式變體 | `model.gate_leaky`、`gate_bias_zero_ref`、`gate_bias_scale`、`gate_bias_a`、`gate_bias_tau`、`gate_bias_learn` | `aggregator.py`：`Aggregator.__init__` + bias 計算處 |
| pose loss 訓練 gate | `model.gate_pose_grad` | `aggregator.py` |
| gate 末層歸零 | `checkpoint.reset_gate_head` | `pipeline/training/trainer.py`：`_reset_gate_head` |
| gate 輸出與 override | — | `vggt/models/vggt.py`：`predictions["gate_logits"]`、`gate_logits_override` |
| `L_gate` | `loss.gate` | `pipeline/training/loss.py`：`compute_gate_loss` |
| oracle mask → override | — | `loss.py`：`oracle_gate_logits_from_mask` |
| Temporal attention | `model.enable_temporal`、`temporal_every`、`temporal_frame_pos` | `aggregator.py`：`_process_temporal_attention`、`temporal_blocks` |
| `L_camera_smooth` | `loss.camera_smooth` | `loss.py`：`compute_camera_smooth_loss` |
| `L_flow_geom` | `loss.flow_geom` | `loss.py`：`compute_flow_geom_loss`；`pipeline/training/flow_mask.py`：`pair_flows` |
| SCARED 抽樣 | `nearby_expand_range`、`min_anchor_valid_frac` | `pipeline/data/datasets/scared.py` |

**評估入口**：

| 目的 | 入口 |
|---|---|
| SCARED（對齊 AF-SfMLearner 協定） | `pipeline/benchmark/eval_scared.py` |
| gate 的唯一評估入口（quality + pose ablation） | `pipeline/diag/gate_eval.py` |
| 拼接誤差本身 | `pipeline/diag/stitch_error.py` |
| 私人資料集 | `pipeline/benchmark/eval_lesion.py` / `eval_gastric.py` |

**chunk_size 陷阱**：任何跨幀指標都要整段一次 forward（`chunk_size=0`），獨立 chunk 拼接會讓接縫主宰 ATE。
權威說明在 `pipeline/eval/vggt_infer.infer_sequence_chunked` 的 docstring。
