# Dyn-VGGT：實作對應、開放決策、與實驗執行

> 本文件是 [dyn_vggt_method.md](dyn_vggt_method.md)（方法/理論）的**工程對應版**：程式落點、關鍵實作細節、開放決策、訓練/評測 SOP，以及**目前實作狀態與驗證結果**。章節編號 §M-x 指向方法文件。

---

## 1. 落地文件清單（對應本倉庫）

| 模塊 | 文件 | 改動 | 狀態 |
|---|---|---|---|
| 時序注意力 + 1D 時間 RoPE | [aggregator.py](../vggt/models/aggregator.py)、[rope.py](../vggt/layers/rope.py) | `RotaryPositionEmbedding1D`、`temporal_blocks`、`_process_temporal_attention`、`aa_order`、`temporal_every` | ✅ 已實作 |
| 雙場輸出組裝 | [vggt.py](../vggt/models/vggt.py) | `motion_head`/`flow_head`、`enable_temporal/motion/flow` 旗標、forward 組裝 `X_{t,p}` | ✅ 已實作 |
| 新 loss | [loss.py](../training/loss.py) | `compute_motion/flow/reproj/tsmooth_loss`、point `(1−m)` 屏蔽、四分支 gating | ✅ 已實作 |
| 尺度正規化（B6） | [normalization.py](../training/train_utils/normalization.py) | `normalize_camera_extrinsics_and_points_batch` 增 `scene_flow` 同步 `/avg_scale` | ✅ 已實作 |
| 數據（PointOdyssey） | [pointodyssey.py](../training/data/datasets/pointodyssey.py) | 返回 `motion_mask`（RNG-replay 對齊） | ✅ 已實作 |
| 數據（TartanAir） | [tartanair.py](../training/data/datasets/tartanair.py) | 靜態場景；不提供 `motion_mask`；S1 防遺忘混入 | ✅ 已實作 |
| 數據 collation | [composed_dataset.py](../training/data/composed_dataset.py) | 轉發 `motion_mask`/`scene_flow_gt` | ✅ 已實作 |
| 訓練 config | [dyn_vggt_po_s0.yaml](../training/config/dyn_vggt_po_s0.yaml)、[dyn_vggt_s1.yaml](../training/config/dyn_vggt_s1.yaml)、[dyn_vggt_s2a.yaml](../training/config/dyn_vggt_s2a.yaml)、[dyn_vggt_s2.yaml](../training/config/dyn_vggt_s2.yaml) + smoke | S0 → S1a/S1b/S2 | ✅ 已實作 |
| warm-start 驗證 | [verify_warmstart.py](../training/verify_warmstart.py) | 載入 VGGT-1B 檢查 missing/unexpected + γ=0 | ✅ PASS |
| 全局對齊（④） | 新建 `dyn_alignment.py` | 前饋初始化的 test-time 優化 | ⏳ 未做（P4） |
| 評測 pipeline | 新建 eval 腳本 | §5 的 ATE/AbsRel/IoU/Chamfer | ⏳ 未做（P5） |
| Demo | [demo_viser.py](../demo_viser.py) | 動靜分色 + 軌跡可視化 | ⏳ 未做 |

---

## 2. 關鍵實作細節

### 2.1 1D 時間 RoPE（§M-3.1）
與 `RotaryPositionEmbedding2D` 同構，但整個 head_dim 用時間座標旋轉、不切 y/x：
```python
# vggt/layers/rope.py
class RotaryPositionEmbedding1D(RotaryPositionEmbedding2D):
    def forward(self, tokens, positions):     # tokens:(B*P,n_heads,S,head_dim); positions:(B*P,S) 整數幀索引
        feature_dim = tokens.size(-1)
        cos, sin = self._compute_frequency_components(feature_dim, int(positions.max())+1, ...)
        return self._apply_1d_rope(tokens, positions, cos, sin)
```
register token 的時間位置設 0 → RoPE 退化為 identity（即「不給時間 RoPE」，對應決策 B4）。

### 2.2 temporal 只更新串流 token、不進 `output_list`（決策 A2，關鍵）
aggregator 裡 token 有兩條去向：
1. **串流隱狀態 `tokens`**：frame→temporal→global 一路被更新（網絡內部計算流）。
2. **回傳給 head 的 `output_list`**：每層把 `frame_intermediates` 與 `global_intermediates` **沿通道維拼接成 `[B,S,P,2C]`**；所有 head 的 `dim_in=2*embed_dim` 正是吃這「frame+global 兩份」。

temporal **必須**更新 `tokens`（讓 global 與後續層看到時序融合），但**絕不可**把中間輸出拼進 `output_list` → 否則通道維變 `3C`、head `dim_in` 被迫改 `3*embed_dim`、**預訓練 head 載入失敗、warm-start 全毀**。故：
```python
def _process_temporal_attention(self, tokens, B, S, P, C, idx, pos=None):
    t = tokens.view(B,S,P,C).permute(0,2,1,3).reshape(B*P,S,C)   # 沿時間軸
    t = self.temporal_blocks[idx](t, pos=pos)
    out = t.view(B,P,S,C).permute(0,2,1,3).reshape(B*S,P,C)
    return out, idx+1        # 不回傳進 output_list 的 intermediate
```

### 2.3 兩個新頭（§M-5）
復用 `DPTHead`。**注意**：`activate_head` 會把最後一個 channel 切成 confidence，故實際 `output_dim` 比語意維度多 1：
```python
# vggt.py __init__（實際採用值）
self.motion_head = DPTHead(dim_in=2*embed_dim, output_dim=2, activation="sigmoid",  conf_activation="sigmoid") if enable_motion else None  # 1 motion + 1 conf
self.flow_head   = DPTHead(dim_in=2*embed_dim, output_dim=4, activation="linear",   conf_activation="expp1")  if enable_flow   else None  # 3 disp  + 1 conf
```
forward 在 point/depth 之後組裝 `predictions["world_points_dyn"] = world_points + motion_prob * scene_flow`。

### 2.4 warm-start γ=0（§M-3.2）
`Block` 在 `init_values` 為 falsy 時用 `nn.Identity()`（不是 LayerScale），故**不能**靠 `init_values=0` 取得 γ=0。做法：用正常 `init_values` 建塊後，把 `ls1.gamma`/`ls2.gamma` 歸零 → temporal 塊初期為 identity。

### 2.5 motion BCE 手寫（AMP 相容）
`F.binary_cross_entropy` 在 AMP autocast 下被禁用；motion 頭輸出已過 sigmoid，故手寫 BCE：`-(gt·log p + (1-gt)·log(1-p))`，autocast 安全。

### 2.6 loss 分支 gating（向後相容）
`MultitaskLoss.forward` 的 **每個**分支（含原 camera/depth/point）都加 `self.X is not None` 守衛 → config 設 `null` 即跳過（S0 用此關掉 camera/depth/point/reproj/tsmooth）。原 VGGT loss 行為不變。

### 2.7 PointOdyssey motion_mask 對齊（RNG-replay）
mask 需與 image/depth 經**相同**的隨機幾何增強。做法：跑 `process_one_image` 前存 `np.random` state，跑完還原 state、把二值 mask 當 pseudo-depth（nearest）再跑一次 → 取其空間輸出。零改動共用 transform helper，不影響既有 co3d/vkitti。

### 2.8 B6 尺度自洽
`scene_flow` 是 world point 的位移 → 與 `world_points` 同 `/avg_scale`（無偏移）。`L_reproj`/`L_flow` 因相機平移也已 `/avg_scale` 而天然尺度不變。不歸一 → `X^can+m·Δ` 量綱錯、**靜默不收斂**。

---

## 3. 開放決策與預設值（Open Design Decisions）

**A 類**影響架構正確性、**必須先拍板**；**B/C 類**可先用預設、列消融；**D 類**為超參。★=建議進 §5 消融表。

### A 類：必須先定（架構正確性）
| # | 決策 | **採用（預設）** | 消融 |
|---|---|---|---|
| A1 | `X^can` 語義 | **per-(t,p) 剛性分支 + 殘差位移**；VGGT 無跨幀對應，「跨幀共享真 canonical」落不了地 | ★（退回單一 world point） |
| A2 | temporal 是否進 `output_list` | **不進**；head 介面維持 `2C`、預訓練權重不動（§2.2） | 否（正確性） |
| A3 | `aa_order` + temporal 數量 | **`["frame","temporal","global"]`，均勻 k=3，共 8 塊** | ★（k / n_temporal） |
| B6 | `Δ`/`X^can` 尺度正規化 | **與 point map 同一 per-scene scale**（§2.8） | 否（正確性） |

### B/C 類：先給預設、列消融
| # | 決策 | **採用（預設）** | 消融 |
|---|---|---|---|
| B4 | special token 進 temporal? | **camera+patch 進且帶時間 RoPE；register 進但時間維補 0**（順序無關摘要槽）；γ=0 暖啟保證最壞無損 | ★（三檔 × pose ATE/depth；可視化 register attn） |
| B5 | 偽標籤 `m*` 自舉穩定 | **detach** depth/pose 算 `m*`（防作弊塌縮，S2 硬性必要）；有 GT 的合成集優先用 GT 算 `f^cam`；`α_m,β_m` 前寬後緊 anneal。curriculum 緩解：S0 的 depth/pose 凍結 → `f^cam` 一開始即可靠 | 可選 |
| B7 | temporal RoPE 長序列外插 | **訓練隨機 frame stride 增廣**；頻率 base 沿用 100，必要時 NTK-aware 插值 | 可選 |
| B8 | causal vs bidirectional | **雙向**（離線重建最準）；保留 causal flag 作串流 future work | 可選 |
| C9 | 全局對齊優化細節 | **先最小版**（只優化 pose、固定 depth）再加碼；長序列滑窗 + 窗間 pose 拼接 | ★（0 迭代 vs 對齊） |

### D 類：超參
| 超參 | 預設 | 備註 |
|---|---|---|
| 動態閾值 `τ` | 0.5 | 僅用於 `valid_frame` 計數；loss 主用連續 `m` |
| 靜態數據混入比例 | 30% | 防遺忘，★ 消融 |
| EMA-teacher 蒸餾 | 開、權重待定 | 只在 `m≈0` 區；是否必要待 ablate |
| `L_flow` 的 SEA-RAFT 配對 | 相鄰 `t→t+1` | 是否加長程影響快取大小 |
| 損失權重 `λ_m,λ_f,λ_rp,λ_ts,λ_tv` | 1.0/0.5/1.0/0.1/0.1 | 必調；reproj 已對角線正規化（`huber_delta≈0.01`） |

**拍板優先序**：實作前必須定 **A1、A2、A3、B6**；其餘可先取預設值開跑。

---

## 4. 訓練流程

### 4.1 模塊耦合度

判準：① 是否共享參數；② 一塊輸出是否進另一塊的 forward/loss。皆否才是真 decoupled。

| 配對 | 耦合度 | 策略 |
|---|---|---|
| 雙場組裝 vs motion/flow 頭 | **強耦合**（組裝吃頭的輸出） | 一起訓 |
| temporal vs 頭 | **弱耦合**（共享 backbone） | 分 stage；合併後須聯合 fine-tune |
| 全局對齊 vs 其餘 | **decoupled**（test-time 只讀凍結輸出） | 獨立開發 |

> **弱耦合陷阱**：頭若在無 temporal 的 backbone 上訓出，temporal 解凍後 feature 分布改變，「各自都 work」不保證拼起來 work → S1a/S1b 分開跑正是為此，合併時須聯合 fine-tune。

### 4.2 訓練課程

Stage 間用 `extract_weights.py` 抽純權重 warm-start（見 [execution §5.4](dyn_vggt_execution.md)）。

```
                        ┌─ S1a ─▶ dyn_vggt_s1a.pt ─┐
VGGT-1B.pt ──S0──▶ s0.pt │  (5 頭 + 7 loss)          ├─ S2 ──▶ ...
                        └─ S1b ─▶ dyn_vggt_s1b.pt ─┘
                           (temporal + 5 loss)
```

#### 總表

| Stage | 凍結 | 解凍 | 啟用 loss | 數據 | lr | epoch | 通過 gate |
|---|---|---|---|---|---|---|---|
| **S0** | aggregator + 原三頭 | 2 新頭（~65M） | `L_motion` `L_flow` | PO | 1e-4 | 8–12 | IoU ≳ 0.5；凍結幾何 bit-identical |
| **S1a** | aggregator | 5 頭（~150–200M） | 全 7 項；`mask_dynamic` 開 | PO 70% + TA 30% | 5e-5 | 20 | motion IoU 維持；val AbsRel/ATE 改善 |
| **S1b** | embed + frame + global | temporal + 5 頭（~448M） | 5 項（無 reproj/tsmooth） | PO 70% + TA 30% | 5e-5 | 20 | 時序一致 ↑；pose 無退化 |
| **S2** | embed + frame | temporal + global + 5 頭（~750M） | 5 項（cam/depth/point/motion/flow；去 reproj/tsmooth） | PO 70% + TA 30% | 5e-5 | 20 | depth/pose 不退化；motion IoU 維持 |

S1a 與 S1b 皆從 S0 出發，是**平行 ablation**：
- **S1a**：固定 backbone，看「解凍全部頭 + 豐富 loss」的效果
- **S1b**：固定 global/frame/embed，看「temporal attention」的效果

**Loss 分類**：

| 項 | S0 | S1a | S1b | S2 |
|---|---|---|---|---|
| `L_motion` `L_flow` | ✔ | ✔ | ✔ | ✔ |
| `L_cam` `L_depth` `L_point` | — | ✔ | ✔ | ✔ |
| `L_reproj` `L_tsmooth` | — | ✔ | — | — |

**記憶體參考（RTX 5090 ~31 GB）**：S0 `img_nums=[2,10]`；S1a `[2,6]`（peak ~25 GB）；S1b `[2,6]`（peak ~25 GB）；S2 `[2,4]`（temporal+global 同開，需壓幀數）。

#### 各 stage 說明

**S0 — 對齊新頭（✅ 已完成）**
在完全不動預訓練幾何的前提下，讓 motion/flow 頭學會分割與 dense flow。temporal 塊 γ=0 → identity，結構存在但不改 feature。`val_metrics: false`（原三頭凍住，depth/pose val 無意義）。
- 實測（epoch-7）：IoU = 0.733，動態 flow +3.8%，凍結 depth/world/pose Δ = 0。

**S1a — 新 loss 效果（✅ 已完成）**
五頭全訓 + 7 loss 全開，在固定 backbone 上看動態 mask + reproj 閉環的效果。
- `mask_dynamic`：`L_point` 乘 `(1−m)`；`L_cam` 用靜態 `valid_frame`（§M-6.1）。
- reproj 閉環可同時更新 camera / point / motion / flow。
- TartanAir 跳過 `L_motion`；其餘 6 項仍監督（靜態 `m≈0`）。
- 實測（epoch-17）：Sintel depth AbsRel ↓26%（vs MonST3R），RPE-rot ↓35%；ATE +58%。

**S1b — temporal attention 效果**
只解凍 temporal blocks（global 凍住），看時序注意力對幾何的貢獻。不含 reproj/tsmooth（S2 觀察到這兩項梯度貢獻極小）。
- S2 同時解凍 temporal + global 導致 pose catastrophic forgetting → S1b 隔離 temporal、保護 global。

**S2 — temporal + global 聯合訓練（ablation，✅ 進行中）**
凍 `patch_embed` + `frame_blocks`，解凍 temporal + global + 5 頭（~750M）。**直接從 S0 warm-start**，跳過 S1 head-only 微調，驗證「temporal+global 一起解凍是否可行」。
- **Loss 5 項**：`L_cam`(w=5) + `L_depth`(w=1) + `L_point`(w=1, `mask_dynamic`) + `L_motion`(w=1) + `L_flow`(w=0.5)。去掉 `reproj`（梯度趨近零）與 `tsmooth`（< 1% 梯度貢獻）。
- **數據**：PO (`len_train=100000`) + TartanAir (`len_train=43000`)；`img_nums=[2,4]`、`max_img_per_gpu=4`。
- **lr**：5e-5 cosine（5% linear warmup → 1e-8）；weight_decay=0.05；grad clip 1.0（aggregator / heads 分組）。
- **AMP**：bfloat16。
- **resume**：`logs/dyn_vggt_s2/ckpts/checkpoint_4.pt`（epoch 4 續訓）。

---

## 5. 評測協議（CVPR 標準）

在 MonST3R 同套 benchmark 同台競技。

| 任務 | 數據集 | 指標 |
|---|---|---|
| 相機 pose | Sintel / TUM-dynamics / ScanNet | ATE、RPE-trans、RPE-rot |
| Video depth | Sintel / Bonn / KITTI | Abs-Rel、δ<1.25、時序一致性（OPW / temporal-AbsRel） |
| 動態分割 | DAVIS / Sintel | IoU、F-measure |
| 點雲/重建 | Dynamic Replica / TUM | Chamfer、Accuracy、Completeness |
| 效率 | 各序列 | 秒/序列 vs MonST3R |

**消融**：①去 temporal-attn；②去雙場分解（退回單一 world point，A1）；③去動態屏蔽（pose `valid_frame`+point `(1−m)`）；④0 迭代 vs 全局對齊（C9）；⑤對比「VGGT 直接微調」（證架構創新非數據紅利）；⑥temporal 數量/間隔 `n_temporal×k`（A3）；⑦special token 進 temporal 三檔 + register attn 可視化（B4）。

**SOTA 對比**：MonST3R、CUT3R、DUSt3R、Robust-CVD、CasualSAM、原始 VGGT。
**賣點**：精度 ≥ MonST3R、速度快一個量級、可擴展到數百幀。

> ⚠️ **trainer 的 validation 只算 val-set 上的 loss，不算上表指標**。ATE/AbsRel/IoU/Chamfer 需獨立 eval pipeline（P5，尚未實作）。

---

## 6. 目前實作狀態與驗證

### 6.1 sintel 評估
| Metric / Model | Depth AbsRel | Depth $\delta < 1.25$ | Depth RMSE | Pose ATE   | Pose RPE-trans | **Pose RPE-rot** |
| -------------- | ------------ | --------------------- | ---------- | ---------- | -------------- | ---------------- |
| base           | 0.2747       | 0.6832                | 5.8274     | 0.1714     | 0.0617         | 0.4706           |
| **S1a**        | 0.2552       | 0.6859                | **5.3843** | 0.1710     | 0.0691         | 0.4792           |
| **S1b**        | 0.2790       | 0.6919                | 5.7328     | **0.1692** | 0.0770         | 0.5502           |
| **s2**         | 0.2840       | 0.7039                | 4.9737     | 0.2035     | 0.0896         | 1.9737           |
| **S1a-full**   | 0.2421       | 0.6869                | 5.2720     | 0.1816     | 0.0675         | 0.5409           |
| **MonST3R**    | 0.3450       | 0.5620                | —          | 0.1080     | 0.0420         | 0.7320           |

### 6.2 待完成
- `dyn_alignment.py`（P4 全局對齊）。
- PointOdyssey `trajs_3d` → 稀疏 `scene_flow_gt`（可選加強）。
- 完整 benchmark（TUM/Bonn/DAVIS/KITTI 等）。
