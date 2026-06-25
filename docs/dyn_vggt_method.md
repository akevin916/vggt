# Dyn-VGGT：運動解耦的 4D 視覺幾何 Transformer

> Motion-Decoupled 4D Visual Geometry Grounded Transformer
>
> 一套在 **VGGT 前饋架構**上、把 **MonST3R 動態建模思想架構化**的動態影片三維重構方法。
> 目標：在動態場景中達到 **精度 ≥ MonST3R、速度快一個量級、可擴展到數百幀**，並滿足 CVPR 投稿的方法新穎性與實驗完整性標準。

本文件每個模塊都對應到本倉庫的真實文件與行號，作為實作藍圖與論文初稿骨架。

---

## 0. TL;DR（一句話 novelty）

> Dyn-VGGT 把 VGGT 的「前饋多視圖幾何聚合」與 MonST3R 的「運動解耦動態建模」在**架構層**統一：用**時空聚合器**引入時序、用**靜態正則點 + 動態場景流的雙場分解**根治單一全局座標系缺陷、用**自監督動態 mask** 解耦相機與物體運動，並以**前饋輸出初始化全局對齊**，實現比 MonST3R 更快、更準、且可擴展到長序列的動態 4D 重建。

---

## 1. 問題診斷：VGGT 為何在動態影片下不如 MonST3R

VGGT 的剛性靜態假設藏在三處，每一處都對應一個失效機制：

### 1.1 單一全局座標系的 point map
[vggt.py:78-83](../vggt/models/vggt.py#L78-L83) 的 `world_points` 形狀為 `[B, S, H, W, 3]`，是把所有幀像素統一回歸到**第一幀相機座標系**下的*靜態*點雲。
對運動物體，同一表面點在不同時刻位於不同的 3D 位置——單一全局座標無法同時表達；網絡被迫「平均」掉運動，導致動態區域深度/點雲糊化、拖影。

### 1.2 Alternating-Attention 無時間序
[aggregator.py:237-253](../vggt/models/aggregator.py#L237-L253) 的 `aa_order = ["frame", "global"]` 把 `S×P` 個 token 完全對稱地拼接做 global attention，而 RoPE 只編碼 **2D 空間位置**（[aggregator.py:221](../vggt/models/aggregator.py#L221) 的 `position_getter` 輸出 `(y, x)`，最後一維為 2；見 [rope.py:55](../vggt/layers/rope.py#L55) `cartesian_prod(y, x)`）。
**沒有任何幀序信息**：網絡不知道哪幀在前哪幀在後，無法建模運動的方向性與連續性，因此 video depth 逐幀獨立、時序閃爍。

### 1.3 相機 pose 與物體運動耦合
[camera_head.py](../vggt/heads/camera_head.py) 回歸 `absT_quaR_FoV`，把整個場景當剛體。動態物體的光流被錯誤歸因到相機運動，污染 pose 估計。
訓練端同樣耦合：[loss.py:97](../training/loss.py#L97) 的 `valid_frame_mask` 只按「有效點數 > 100」篩幀，對動態像素不做屏蔽，pose 監督被動態點污染。

> **MonST3R 的解法**：少量動態數據微調 + 全局優化時引入光流一致性與動態 mask，解耦相機/物體運動。
> **但** MonST3R 繼承 DUSt3R，只能成對處理、需慢速全局對齊、難以擴展到長序列。
> **我們的目標**：保留 VGGT 一次吃數百幀的前饋能力，把 MonST3R 的動態建模思想**架構化**進 VGGT。

---

## 2. 方法總覽：四大貢獻

```
            輸入視頻 {I_t}  (S frames)
                  │
       ┌──────────▼───────────┐
       │  DINOv2 Patch Embed   │  (沿用)
       └──────────┬───────────┘
                  │  + 時間位置編碼 (貢獻①)
       ┌──────────▼─────────────────────────┐
       │  Spatio-Temporal Aggregator         │
       │  frame-attn / temporal-attn /       │  (貢獻① 時序 attn + 1D 時間 RoPE)
       │  global-attn                        │
       └──────────┬─────────────────────────┘
           ┌──────┼───────┬──────────┬─────────────┐
           ▼      ▼       ▼          ▼             ▼
      Camera   Depth   Point     Motion-Seg     Scene-Flow
      Head     Head    Head      Head (新)       Head (新)
       │        │       │          │              │
       │        │  X^can_{t,p}  m_t ∈[0,1]   Δ_{t,p} ∈ R^3
       └────────┴───────┴──────────┴──────────────┘
                         │   雙場組裝 (貢獻②):  X_{t,p}=X^can_{t,p} + m·Δ
            ┌────────────▼─────────────┐
            │  輕量級 4D 全局對齊 (貢獻④)│  (test-time，前饋初始化)
            └────────────┬─────────────┘
                         ▼
        靜態背景點雲 + 每幀動態點雲 + 解耦相機軌跡
```

| # | 貢獻 | 解決的根因 | 核心機制 |
|---|---|---|---|
| ① | **時空聚合器** | §1.2 無時序 | alternating-attn 插入 temporal-attn + 獨立 1D 時間 RoPE |
| ② | **運動解耦雙場表示** | §1.1 單座標系 | 靜態正則點 `X^can` + 每幀殘差場景流 `Δ`，soft-gate by `m` |
| ③ | **動態分割頭 + 場景流頭** | §1.3 運動耦合 | 自監督動態 mask；pose 按靜態 `valid_frame` 篩、point 按 `(1−m)` 加權屏蔽動態 |
| ④ | **前饋初始化的 4D 全局對齊** | MonST3R 慢 | 用前饋輸出初始化，20–50 迭代收斂（MonST3R 需數百次） |

---

## 3. 貢獻①：時空聚合器（Spatio-Temporal Aggregator）

### 3.0 RoPE 概念回顧（理解時間編碼的前提）
Attention 本身對 token 順序無感，必須額外注入「位置」。VGGT 用的是 **RoPE（旋轉式位置編碼）**：不把位置向量「加」到 token 上，而是**把 q、k 向量旋轉一個正比於位置的角度**。旋轉後 `q·k` 只跟相對距離有關，天然就是相對位置編碼、且零參數。

實作要點（對應 [rope.py](../vggt/layers/rope.py)）：
- 取特徵裡的一對數視為 2D 平面點，位置 `m` → 旋轉角 `θ_m = m·ω`；多對特徵用不同頻率 `ω_i = 1/base^(2i/d)`（[rope.py:103-104](../vggt/layers/rope.py#L103-L104)），低維轉得快管近距離、高維轉得慢管遠距離。
- 旋轉以 LLaMA 風格的 rotate-half 實現：`out = x·cos + rotate_half(x)·sin`（[rope.py:133-152](../vggt/layers/rope.py#L133-L152)）。
- **2D 版**把特徵維對半切，前半用 `y` 旋轉、後半用 `x` 旋轉（[rope.py:180-188](../vggt/layers/rope.py#L180-L188)），於是一個 patch token 同時帶上垂直/水平相對位置。
- RoPE 在 attention 內**只作用於 q、k**（[attention.py:56-58](../vggt/layers/attention.py#L56-L58)），v 不旋轉——位置只影響「誰跟誰相關」，不影響「傳遞什麼內容」。
- 頻率表用 `F.embedding(positions, ...)` 查表（[rope.py:148](../vggt/layers/rope.py#L148)），因此 **positions 必須是整數索引**——這直接決定下面時間軸要用整數幀索引。

### 3.1 時間位置編碼（獨立 1D 時間 RoPE，與空間 RoPE 並存）
> **修正先前簡化講法**：不採用「把 pos 從 `(y,x)` 擴成 `(y,x,t)`、最後一維 2→3」的 3D-RoPE。原因：現有 2D RoPE 的「前半 y、後半 x」對半切是寫死的（[rope.py:180-185](../vggt/layers/rope.py#L180-L185)），改成三等分會打亂預訓練權重學到的特徵分工，**warm-start 失效**。

正確做法：**新增一個獨立的 1D 時間 RoPE，只在 temporal attention 內作用**，空間 RoPE 一個字不改。

```python
# vggt/layers/rope.py 新增（與 RotaryPositionEmbedding2D 同構，但不切 y/x）
class RotaryPositionEmbedding1D(nn.Module):
    """整個 head_dim 都用時間座標 t 旋轉。_compute_frequency_components /
       _rotate_features / _apply_1d_rope 可直接複用 2D 版邏輯。"""
    def forward(self, tokens, positions):     # tokens: (B*P, n_heads, S, head_dim)
        feature_dim = tokens.size(-1)         # positions: (B*P, S) 整數幀索引
        cos, sin = self._compute_frequency_components(feature_dim, int(positions.max())+1, ...)
        return self._apply_1d_rope(tokens, positions, cos, sin)
```

接線規則：
- **位置用整數幀索引 `t ∈ {0,…,S−1}`**，不用 `τ=t/(S−1)` 分數——因為 [rope.py:148](../vggt/layers/rope.py#L148) 的 `F.embedding` 要整數索引；RoPE 本就只看相對差 `t_i−t_j`，整數已能完整表達「隔幾幀」。
- **frame / global attention 維持原 2D 空間 RoPE 不變**；時間相位只在 temporal attention 注入，職責乾淨、warm-start 完整保留。
- temporal blocks 構造時掛上這個 1D rope：`block_fn(..., rope=self.temporal_rope)`，其內部 [attention.py:56-58](../vggt/layers/attention.py#L56-L58) 會自動以幀索引旋轉 q、k。
- **special tokens 的時間處理（決策 B4）**：camera token 與 patch token **進** temporal attention 並**帶時間 RoPE**（相機軌跡、像素軌跡都需有序時序）；register token **進** temporal attention 但**不給時間 RoPE**（時間維補 0，沿用 [aggregator.py:223-228](../vggt/models/aggregator.py#L223-L228) 對 special token「空間 pos 補 0」的處理慣例）。如此 register 變成「順序無關的時序全局摘要槽」，camera/patch 則編碼有序軌跡；最壞情況靠 γ=0 暖啟保證無損。詳見 §12-B4。

### 3.2 時序注意力塊（核心；因果為可選）
在每個 aa 迴圈（[aggregator.py:237-248](../vggt/models/aggregator.py#L237-L248)）的 frame / global 之間插入第三種注意力 `"temporal"`，把 token 重排為 `(B*P, S, C)`，**僅沿時間軸**做注意力，捕捉同一空間位置的時序演化（運動連續性、軌跡）。

新增一個與現有 `_process_frame_attention` / `_process_global_attention` 對稱的函數：

```python
# aggregator.py 新增（對稱於 _process_global_attention，但「不」回傳 intermediate）
def _process_temporal_attention(self, tokens, B, S, P, C, idx, pos=None):
    # (B*S, P, C) -> (B, S, P, C) -> (B*P, S, C)：沿時間軸做 attention
    t = tokens.view(B, S, P, C).permute(0, 2, 1, 3).reshape(B * P, S, C)
    # 可選 causal mask：嚴格因果(只看過去) vs 雙向(離線視頻，預設雙向)
    t = self.temporal_blocks[idx](t, pos=temporal_pos)
    out = t.view(B, P, S, C).permute(0, 2, 1, 3).reshape(B * S, P, C)
    return out, idx + 1        # 只更新串流 tokens，不產出進 output_list 的 intermediate
```

**【決策 A2】temporal 只更新串流 `tokens`，不進 `output_list`（維持 head 介面 2C 不變）。**
aggregator 裡 token 有兩條去向：
1. **串流隱狀態 `tokens`**：frame→temporal→global 一路被更新，是網絡內部計算流。
2. **回傳給 head 的 `output_list`**（[aggregator.py:250-253](../vggt/models/aggregator.py#L250-L253)）：每層把 `frame_intermediates` 與 `global_intermediates` **沿通道維拼接成 `[B,S,P,2C]`**；所有 head 的 `dim_in=2*embed_dim` 正是吃這個「frame+global 兩份」。

temporal **必須**更新 `tokens`（去向 1，這樣 global 與後續層才看得到時序融合），但**絕不可**把自己的中間輸出也拼進 `output_list`（去向 2）——否則通道維變 `3C`，head 的 `dim_in` 被迫改成 `3*embed_dim`，而**預訓練 head 權重是 `2C` 形狀 → 載入失敗、warm-start 全毀**。因為 frame/global 兩者都貢獻 intermediate，照「對稱」直覺很容易誤把 temporal 也加進拼接，故此處必須明確規定：**`output_list` 永遠維持 `[B,S,P,2C]`，temporal 的時序增益透過更新後的 `tokens` 間接流入 global 與 head。**

**【決策 A3】`aa_order = ["frame", "temporal", "global"]`，均勻每 `k=3` 個 aa-block 插 1 個 temporal block（depth=24 → 共 8 個 temporal block）。**
- 順序選 frame→temporal→global：讓 global 看到「已時序融合」的 token。
- temporal 放在每組的中間、均勻分布（block 3,6,9,…,24 處），`n_temporal=8`。
- 顯存：`(B*P, S, C)` 序列長僅 `S`、複雜度 `O(B·P·S²)`，遠小於 global 的 `O((S·P)²)`，故插 8 個成本可控。
- `n_temporal` 與插入間隔 `k` 仍列為 §9 消融超參。

**為何這樣設計（三者正交分工）**：
- frame-attn → 單幀內部空間幾何；
- temporal-attn → 同一空間位置的「軌跡 / 運動」；
- global-attn → 跨幀跨空間的全局幾何一致性。
三者復用 VGGT 既有的 `Block` 結構，可從預訓練權重 **warm-start**。

**warm-start 的具體機制（LayerScale 零初始化）**：[Block](../vggt/layers/block.py) 的殘差為 `x = x + γ · attn(x)`，其中 γ 是 LayerScale（由 `init_values` 控制）。把 temporal block 的 γ **初始化為 0**，則 temporal 塊初期輸出恆等於輸入——**整個網絡第一步嚴格等於原 VGGT**，不擾動既有幾何；訓練中 γ 由 0 漸增，時序機制「漸進通電」。這正是 §7 中 S0「temporal 塊存在但等價於不起作用」的落地方式。

---

## 4. 貢獻②：運動解耦的雙場表示（理論支點）

直接解決 §1.1「單一全局座標系無法表達運動物體」。

### 4.1 定義（採 per-frame 剛性分支 + 殘差位移，決策 A1-b）
對像素 p 在時刻 t，**不再**只預測單一全局點，而是分解：

```
X_{t,p} = X^can_{t,p}  +  m_{t,p} · Δ_{t,p}
          └ 剛性分支 ┘     └動態概率┘ └殘差位移┘
```

> **語義釐清（決策 A1）**：VGGT **沒有跨幀像素對應**（無 track），因此無法定義「跨幀共享、同一世界點唯一」的真 canonical 點。本方法改採**逐像素逐幀（per-(t,p)）的剛性分支**語義：
> - `X^can_{t,p}`：**「假設此像素為靜態時」該落的世界座標**（即 VGGT 原 point head 的剛性多視圖預測）。它是 per-frame 輸出，不要求跨幀共享。
> - `Δ_{t,p} ∈ R³`：把剛性預測**修正到第 t 幀真實位置**的 3D 殘差位移，由 Scene-Flow 頭給出。
> - `m_{t,p} ∈ [0,1]`：動態概率，由 Motion-Seg 頭給出。靜態區 `m≈0` 退化為原 VGGT（`X≈X^can`）；動態區 `m≈1` 殘差生效。
>
> **這版可逐像素直接實作、不需任何跨幀對應**。監督方式據此分流（見 §6.1）：靜態區 `(1−m)` 直接拿 GT world point 監督 `X^can`；**動態區**的幾何不直接監督 `X^can`（其為隱變量），而是讓組裝點 `X_{t,p}=X^can+m·Δ` 經 `L_reproj` 對齊到觀測。因 `X^can` 為 per-frame，**不需**強制 `Δ_{0,p}=0` 之類的錨定。

### 4.2 三組輸出與雙路徑融合
| 路徑 | 輸出 | 頭 | 對應強項 |
|---|---|---|---|
| 靜態幾何（剛性分支） | `X^can_{t,p}` + conf | Point Head（[vggt.py:25](../vggt/models/vggt.py#L25), `output_dim=4`） | VGGT 多視圖三角化（數百幀聚合） |
| 動態幾何 | 每幀真實深度 `D_{t,p}` | Depth Head（[vggt.py:26](../vggt/models/vggt.py#L26), `output_dim=2`） | MonST3R 單目深度 + scene flow |

用 `m_{t,p}` 做 **soft gating** 融合兩條路徑——這就是「直接結合 MonST3R 做法」的架構化表達。二者透過相機參數與位移場由 §6 的 reprojection loss 閉環約束。

---

## 5. 貢獻③：動態分割頭 + 場景流頭

新增兩個輕量 DPT 頭（復用 [dpt_head.py](../vggt/heads/dpt_head.py)），在 [vggt.py:24-27](../vggt/models/vggt.py#L24-L27) 同位置接入：

```python
# vggt.py __init__ 新增
self.motion_head = DPTHead(dim_in=2*embed_dim, output_dim=1,
                           activation="sigmoid", conf_activation=None) if enable_motion else None
self.flow_head   = DPTHead(dim_in=2*embed_dim, output_dim=3,
                           activation="linear",  conf_activation="expp1") if enable_flow else None
```

forward 在 [vggt.py:78-83](../vggt/models/vggt.py#L78-L83) 之後組裝 `predictions["motion_prob"]`、`predictions["scene_flow"]`，並輸出組裝後的 `X_{t,p}`（§4.1）。

### 5.1 動態 mask 的自監督信號（無需動態標註）
把 MonST3R 的「光流一致性判定動態」做成**可微自監督 loss**：

由預測深度 + pose + 剛體假設算出相機運動誘導的「期望光流」`f^cam_t`；與離線預算的真實光流 `f^gt_t`（SEA-RAFT / RAFT）對比，殘差大者為動態：

```
r_{t,p} = || f^gt_{t,p} − f^cam_{t,p} ||
m*_{t,p} = σ( α_m · (r_{t,p} − β_m) )      # 偽標籤（α_m, β_m 為光流殘差的尺度/閾值，與 confidence 的 α 無關）
```

- 無動態 GT 的數據集：`m*` 作偽標籤監督 Motion-Seg 頭。
- 有合成動態 GT 的數據集（PointOdyssey / Spring / Dynamic Replica / TartanAir-shibuya）：直接用 GT 監督。

### 5.2 動態屏蔽（pose 提升來源）——兩處獨立改動
動態屏蔽要落在**兩個粒度不同的地方**，缺一不可（兩者皆源自 `point_masks`，但用法不同）：

1. **pose（序列級開關）**：[loss.py:97](../training/loss.py#L97) 的 `valid_frame_mask` 原本只看「anchor 幀有效點數>100」（per-sample 開關，且不排除動態）。改為按**靜態像素**計數 `Σ_p point_masks·𝟙[m<τ] > 100`，使 pose 只由靜態像素監督，從根上解耦相機與物體運動。
2. **正則點（像素級權重）**：`L_point` 走的是 per-pixel `regression_loss`，動態屏蔽要落在**像素權重 `(1−m)`** 上（見 §6.1 的說明），避免動態 GT 污染靜態正則場。

兩者的 `m` 皆取 `detach`，只當開關/權重用，不讓 pose/point 的梯度反灌 motion 頭。

---

## 6. 損失函數（對接 training/loss.py）

在 [MultitaskLoss](../training/loss.py#L17-L78) 上擴展。**原則**：前三項必須沿用既有的 confidence-weighted regression 形式（[loss.py:281-367](../training/loss.py#L281-L367)），不另起爐灶；新四項接在同一套基礎設施上。

### 6.0 符號約定
- 序列 `S` 幀，像素 `p`，幀索引 `t`；`u_{t,p}` 為像素座標。
- `M_{t,p}∈{0,1}`：GT 有效像素 mask（即 `point_masks`，[loss.py:214](../training/loss.py#L214)）。
- `m_{t,p}∈[0,1]`：預測動態概率；`m*_{t,p}`：其偽標籤。
- 相機：外參 `P_t=[R_t|τ_t]`、內參 `K_t`；投影 `π(·)`、反投影 `π⁻¹(u,d)`。
- 雙場組裝點：`X_{t,p} = X^can_{t,p} + m_{t,p}·Δ_{t,p}`。
- `𝒱 = {(t,p): M_{t,p}=1}`。

### 6.1 既有三項（沿用 confidence-weighted 形式）

**核心回歸子** `regression_loss`（[loss.py:307-312](../training/loss.py#L307-L312)），depth/point 共用。對場 `y`（預測 `ŷ`、置信 `c`）：

```
ℓ_reg  = mean_{(t,p)∈𝒱} ‖ y − ŷ ‖₂
ℓ_conf = mean_{(t,p)∈𝒱} [ γ·‖ y − ŷ ‖₂·c − α·log c ]      (預設 γ=1.0, α=0.2)
```
第一項要求「置信高處誤差小」，`−α log c` 防止 `c→0`（不確定性學習）。

**多尺度梯度/法向項**（[loss.py:325-343](../training/loss.py#L325-L343)），`scales=3`：
```
ℓ_grad = mean_s ( 1 − cos∠( n(ŷ^(s)), n(y^(s)) ) )      # n(·)=鄰域叉積法向
```

於是（對應 [loss.py:59,67](../training/loss.py#L59) 三項相加）：
```
L_depth = ℓ_conf^D      + ℓ_reg^D      + ℓ_grad^D       # 監督每幀真實深度 D_{t,p}（動靜皆可，深度在幀 t 可觀測）
L_point = ℓ_conf^{Xcan} + ℓ_reg^{Xcan} + ℓ_grad^{Xcan}  # 監督正則點 X^can，逐像素乘 (1−m) 權重
```

> **為何 `L_point` 必須乘 `(1−m)`**：GT `world_points` 對動態物體記錄的是**第 t 幀的移動位置**，而 `X^can` 定義為**靜態正則點**。直接拿移動的 GT 回歸 `X^can` 在動態區是錯誤監督。故 `X^can` 只在靜態區（`m≈0`）由 GT 監督；**動態區的幾何**改由「組裝點 `X_{t,p}=X^can+m·Δ`」經 `L_reproj`（§6.2-3）約束。`L_depth` 則不需 `(1−m)`：每幀真實深度動靜都可觀測。`m` 在此處取 `detach`（當權重用，不反傳）。

**相機項（動態屏蔽版）** `L_cam^stat`。pose 編碼 `absT_quaR_FoV` 拆成 `T∈ℝ³`(0:3)、四元數 `q∈ℝ⁴`(3:7)、FoV(7:9)，逐分量 L1（[loss.py:175-177](../training/loss.py#L175-L177)），跨 `n` 個 refine stage 以 `γ^{n−i−1}` 加權（[loss.py:117](../training/loss.py#L117)）：
```
L_cam = (1/n) Σ_i γ^{n−i−1} ( w_T‖T̂^(i)−T‖₁ + w_R‖q̂^(i)−q‖₁ + w_FL‖f̂^(i)−f‖₁ )
        (預設 w_T=1, w_R=1, w_FL=0.5, γ=0.6)
```
**本方法改動**：原 [loss.py:97](../training/loss.py#L97) 的 `valid_frame_mask`（有效點數>100）改為**只統計靜態像素**：
```
valid_frame(t) = [ Σ_p M_{t,p}·𝟙[ m_{t,p} < τ ] > 100 ]      (τ=0.5, m 取 detach)
```
即 pose 只由靜態像素監督；`m` 取 `detach` 避免 pose 梯度反灌 motion 頭，保持解耦。

### 6.2 新增四項

**(1) 動態分割** `L_motion`（BCE）：
```
L_motion = − mean_{(t,p)∈𝒱} [ m*·log m + (1−m*)·log(1−m) ]
```
偽標籤 `m*`（無動態 GT 時，§5.1）。先算相機誘導的剛體光流，再與離線 SEA-RAFT 真實光流 `f^gt` 比殘差：
```
f^cam_{t,p} = π( P_{t+1} P_t⁻¹ π⁻¹(u_{t,p}, D_{t,p}) ) − u_{t,p}
r_{t,p}     = ‖ f^gt_{t,p} − f^cam_{t,p} ‖₂
m*_{t,p}    = σ( α_m·(r_{t,p} − β_m) )
```
有合成動態 GT 的數據集則 `m*` 直接取 GT 二值 mask。

**(2) 場景流** `L_flow`（訓練 flow 頭的 `Δ`）：

**主路徑 — dense「組裝點 vs GT world_points」（只需 depth+相機，所有訓練集可用，無需光流）**：
GT `world_points` 在第 t 幀記錄的就是每個像素表面**當下時刻**的真實世界座標（動態物體亦然，深度即抓該瞬間位置），故組裝點 `X_{t,p}=X^can+m·Δ` 有 dense GT 目標。對其做 confidence-weighted regression（沿用 §6.1 形式，置信用 `Δ` 頭的 `scene_flow_conf`）：
```
L_flow^dense = ℓ_conf( X^can_{t,p} + m·Δ_{t,p} ,  world_points^gt_{t,p} ;  conf=scene_flow_conf )
```
搭配 `(1−m)`-masked 的 `L_point`（只在靜態區釘住 `X^can`），動態區的幾何缺口被迫由 `m·Δ` 補上——這就把「動態監督從 `X^can` 搬到 `X^can+m·Δ`」，是雙場解耦的直接落地。

**可選加強 — 稀疏 3D scene-flow GT**（如 PointOdyssey `trajs_3d`，僅在有 `scene_flow_mask` 的像素）：
```
L_flow^sparse = Σ valid · huber( Δ_{t,p} − Δ^gt_{t,p} )
```

> **光流的角色已收斂**：因 dense 主路徑只需 depth+相機，**supervised flow 不再需要 2D 光流 / SEA-RAFT**。2D 光流 `f^gt` 現在**僅**用於「無動態 mask 的數據集」生成 motion 偽標籤 `m*`（§5.1）；有 mask GT 的集（PointOdyssey/Waymo/Sintel）連這個也不需要。原先的光流自監督 `L_flow^self` 只在「無 depth GT」時才需要，本方法訓練集皆有 depth，故不啟用。

**(3) 跨時刻重投影一致性** `L_reproj`（核心閉環，把 point/flow/pose/motion 四頭聯合約束）：
```
L_reproj = mean_{(t,p)∈𝒱}  ‖ π( K_t [R_t|τ_t] (X^can_{t,p} + m_{t,p}·Δ_{t,p}) ) − u_{t,p} ‖_δ
```
靜態區（`m≈0`）退化為「正則點投影回原像素」的多視圖約束；動態區（`m≈1`）位移生效。內部 `π` 對深度 `clamp`、整體用 Huber 以穩定。**像素誤差除以影像對角線正規化**（分數單位，`huber_delta≈0.01`），避免原始像素尺度壓過其他項（P0 實測 reproj 會主導 objective）。

**(4) 時間平滑** `L_tsmooth`（抑制 §1.2 的 video depth 閃爍）：
```
L_tsmooth = mean_{t,p} M·‖ Δ_{t+1} − 2Δ_t + Δ_{t−1} ‖₁    # scene flow 二階平滑
          + λ_tv · Σ_{t,p} | m_{t,p} − m_{t−1,p} |         # 動態 mask 時序 TV
```

### 6.3 總損失與按 stage 的權重開關
```
L = L_cam^stat + L_depth + L_point + λ_m·L_motion + λ_f·L_flow + λ_rp·L_reproj + λ_ts·L_tsmooth
```
建議起始配比（待 §9 消融調）：`λ_m=1.0, λ_f=0.5, λ_rp=1.0, λ_ts=0.1, λ_tv=0.1`。

按 §7 課程學習的啟用順序：
- **S0**：只開 `L_motion + L_flow`，其餘 `λ=0`。
- **S1**：加開 `L_reproj + L_tsmooth`，並切 `L_cam → L_cam^stat`（動態屏蔽）。
- **S2**：全 7 項到位、調最終配比。

**實作**：在 [MultitaskLoss.__init__](../training/loss.py#L27-L33) 加 `self.motion/flow/reproj/tsmooth` config dict（沿用既有 `self.camera/depth/point` 模式），在 [forward](../training/loss.py#L35-L78) 增對應分支；`L_motion/L_flow/L_reproj` 復用 `regression_loss` 與既有 mask 機制，`L_tsmooth` 沿時間軸差分。

---

## 7. 訓練策略（CVPR 可復現的關鍵）

VGGT 是強預訓練 baseline，**絕不從頭訓**。採 MonST3R 式「凍結 encoder、小數據微調」+ 課程學習（凍結用 [train_utils/freeze.py](../training/train_utils/freeze.py)）。

> **關鍵前提：「架構模塊存不存在」≠「該 stage 有沒有訓練/啟用它」。**
> Dyn-VGGT 是**單一模型**——temporal-attn、雙場分解、motion/flow 兩個頭在 S0 就**結構性存在**於網絡中（否則無法載入同一份 checkpoint 續訓）。三個 stage 改變的不是「架構有沒有」，而是 (a)**哪些參數解凍**、(b)**哪些 loss 啟用**。因此 **S0 並非一次打開所有機制**，而是漸進解凍以保護預訓練的靜態幾何先驗。

### 7.1 三階段總覽

| 階段 | 凍結 | 解凍訓練 | 數據 | 目的 |
|---|---|---|---|---|
| S0 | aggregator（含 temporal 塊）+ 原三頭 | 僅 motion / flow 兩個新頭 | 動態合成集 | 對齊新頭，不破壞原幾何 |
| S1 | DINOv2 patch embed | temporal 塊 + 原三頭 + 兩新頭 | 動態為主 + 少量靜態 | 學時序與運動解耦 |
| S2 | 無（全網，小 lr 1e-5） | 全部 | 動靜混合 | 聯合精修 |

### 7.2 各 stage「新增了什麼」（凍結 × loss 啟用 × 機制生效）

| | 解凍訓練 | 新打開的 loss | 新生效的機制 |
|---|---|---|---|
| **S0** | motion / flow 頭 | `L_motion`, `L_flow` | 兩個新頭對齊；雙場 `X=X^can+m·Δ` 僅前向**不參與 loss**；temporal 塊存在但凍結（零初始化殘差門控 ≈ 不起作用） |
| **S1** | + temporal 塊、原 camera/depth/point 三頭 | `L_reproj`, `L_tsmooth` | 雙場閉環正式生效（進 `L_reproj`）；**動態屏蔽開啟**（`L_cam` 改靜態 `valid_frame`、`L_point` 改 `(1−m)` 加權，§5.2）；temporal-attn 解凍開始學幀序 |
| **S2** | + DINOv2 patch embed（全網） | （全 7 項到位、調最終配比） | 端到端聯合精修 + 防遺忘 |

**設計理由（為何不在 S0 全開）**：
- **S0**：兩個新頭隨機初始化。若一開始就接 `L_reproj` 這類閉環 loss，隨機的 `Δ`/`m` 會把梯度灌回已訓好的 depth/point/pose 頭，**污染並破壞 VGGT 既有幾何**。故先凍住其餘部分，只把兩新頭「養」到 mask 大致分得出動靜、flow 大致方向正確。
- **S1**：`m`/`Δ` 已可用，此時才有資格把它們接進 `L_reproj`（四頭聯合約束）與 pose 屏蔽（用可信的 `m` mask 動態像素）；同時解凍 temporal 塊，讓**時序機制與運動解耦一起學**——貢獻①②③在此全部通電，是核心 stage。
- **S2**：放開全網端到端精修，磨平模塊間殘餘不一致；小 lr + 靜態混入 + 蒸餾確保不洗掉原靜態幾何能力。

- **防遺忘**：靜態數據按比例混入（~30%），對 `m≈0` 區保持原 VGGT 行為；可加 **EMA-teacher 蒸餾**舊 VGGT 在靜態幀的輸出。
- **數據集**（公開、含 depth+flow+motion GT）：PointOdyssey、TartanAir、Spring、Dynamic Replica、VKITTI2、Waymo（真實）；Sintel（含動態 GT，僅 eval）。
- **數據載入**：複用 [dynamic_dataloader.py](../training/data/dynamic_dataloader.py)，擴展 `__getitem__` 返回 `flow_gt` / `motion_mask` / `scene_flow_gt`。
- **離線光流（角色已收斂）**：SEA-RAFT 預算 `f^gt` 快取，**僅供無動態 mask 的數據集（Spring/Bonn）生成 motion 偽標籤 `m*`**；supervised flow 改用 dense「組裝點 vs GT world_points」(§6.2-2)，不需光流。有 mask GT 的集（PointOdyssey/Waymo/Sintel）完全不需 SEA-RAFT。

### 7.3 開發與訓練流程（含耦合分析與執行 SOP）

§7.1/7.2 定義「訓練軌跡」（S0→S1→S2）；本節給「**怎麼組織開發與執行**」——先用模塊耦合度判斷哪些塊可平行降風險驗證，再給唯一一條照著跑的 pipeline。

**(a) 模塊耦合度（決定能否平行開發）**
判準（任一成立即耦合）：① 是否**共享參數**；② 一塊輸出是否進另一塊的 **forward / loss**。兩者皆否才是真 decoupled。

```
① temporal-attn（aggregator backbone）── 產出 token features → 餵給「所有」頭
      ▼
 ┌──────────────────────────────┐
 │ ③ motion頭(m) + flow頭(Δ)     │ ── 產出 m, Δ
 │ ② 雙場組裝 X = X^can + m·Δ    │ ◀── 直接吃 ③ 的輸出
 └──────────────────────────────┘
      ▼
④ 全局對齊（test-time，只讀凍結的前饋輸出）
```

| 配對 | 共享參數? | 輸出互餵? | 耦合度 | 開發策略 |
|---|---|---|---|---|
| **② vs ③** | — | 是（② 組裝 ③ 的 `m`、`Δ`） | **強耦合** | 從頭綁一組，不拆 |
| **① vs (②+③)** | 是（共享 backbone） | 間接（① feature 餵 ③ 的頭） | **弱耦合** | 可平行開發，**合併須重對齊** |
| **④ vs 其餘** | 否（不訓） | 單向（只讀前饋輸出） | **近乎完全 decoupled** | 隨時並行、合併零成本 |

> **弱耦合的唯一陷阱——合併後必須重對齊**：②③ 的頭若在原始（無 temporal）backbone feature 上訓出，換成 temporal backbone 後 feature 分布改變，「各自都 work」不保證直接拼起來 work，必須靠一次聯合 fine-tune（即下方 P3）縫合。完全 decoupled 的 ④ 則零成本合併。

**(b) 執行 SOP**——把「平行降風險探針（P1）」與「合併後 curriculum 聯合精修（P3，對應 S0/S1/S2）」縫成一條線。每個 phase 含凍結 / loss / 數據 / **通過門檻（gate）**，gate 不過不進下一階段。

| Phase | 做什麼 | 凍結 / 解凍 | 啟用 loss | 數據 | 通過門檻（gate） |
|---|---|---|---|---|---|
| **P0 準備** | 擴 dataloader 返回 `flow/motion/scene_flow` GT；離線快取 SEA-RAFT 光流；端到端 smoke test | — | 全項各跑一步驗證可降 | 小子集 | forward+backward 不 NaN、shape 對、loss 會降 |
| **P1-A 探針：時序**（可平行） | 在**原 VGGT** backbone 加 temporal 塊（LayerScale γ=0 暖啟），凍其餘 | 凍全部，僅解凍 temporal 塊 | `L_depth`+`L_tsmooth` | 動態合成集 | video depth 時序一致性（OPW/temporal-AbsRel）優於原 VGGT；單幀精度不退 |
| **P1-B 探針：雙場頭**（可平行） | 在**原(無temporal)** backbone 加 motion/flow 兩頭 + 雙場組裝，凍 backbone 與原三頭 | 凍 backbone+原三頭，解凍兩新頭 | `L_motion`+`L_flow` | 動態合成集 | `m` 動靜分割 IoU 基本可用；`Δ` 方向正確（flow EPE 收斂） |
| **P2 合併** | P1-A 的 temporal backbone + P1-B 的兩頭裝進同一模型，載入各自權重 | — | — | — | 載入無誤；合併後前向不崩（指標暫掉正常，待 P3 對齊） |
| **P3a 對齊新頭**（≈S0） | 讓兩新頭適配「新 temporal backbone」的 feature 分布 | 凍 backbone+原三頭，解凍兩新頭 | `L_motion`+`L_flow` | 動態為主 | `m`/`Δ` 指標恢復到 ≥ P1-B 水準 |
| **P3b 運動解耦**（≈S1） | 解凍 temporal+原三頭，接閉環與動態屏蔽 | 凍 DINOv2 patch embed | + `L_reproj` + `L_tsmooth`；`L_cam` 改靜態 `valid_frame`、`L_point` 改 `(1−m)` 加權（§5.2） | 動態為主+少量靜態 | 動態區深度/點雲、相機 pose（ATE）優於「VGGT 直接微調」baseline |
| **P3c 全網精修**（≈S2） | 全網小 lr(1e-5) 端到端 + 防遺忘 | 全部解凍 | 全 7 項到位、最終配比 | 動靜混合（靜態~30%） | 主指標達/超 MonST3R；靜態場景無退化（防遺忘 gate） |
| **P4 全局對齊**（④，decoupled） | 凍結模型上加 test-time 4D 對齊（§8），不再訓網絡 | 模型全凍 | —（優化變量 pose/depth/flow） | eval 序列 | 精修模式 ATE/AbsRel 再升、20–50 迭代收斂 |
| **P5 評測+消融** | 跑 §9 全部 benchmark 與五個 ablation | — | — | eval 集 | 完整表格 + ablation 證每塊貢獻 |

**關鍵執行說明：**
- **平行 vs 串行**：P1-A / P1-B 可平行（弱耦合，互不阻塞）；人力/算力有限也可串行（先 P1-B 再 P1-A）。**P2 之後一律串行**。
- **P1 是降風險探針，不出論文數字**：用小數據短步數早止損（gate 不過就回去改設計）；SOTA 數字在 P3c 產生。
- **P3 = 合併後的 curriculum**：P3a/b/c 一一對應 S0/S1/S2，差別只在起點是已暖身的 P1 權重而非隨機新頭，故 P3a 比從零的 S0 收斂更快。
- **省算力分叉點**：所有 ablation（§9）與 full model **共享 P3a 後的 checkpoint** 起跑，各自永久關掉一個機制再走 P3b→P3c，避免重複前段；這也天然對應「去①／去②③」的消融。
- **防遺忘 gate（硬性）**：P3c 結束必須在純靜態 benchmark（如 ScanNet/原 VGGT 測集）驗證**不退化**；退化則調高靜態混入比例或加重 EMA-teacher 蒸餾權重。

**一句話總結整條線**：`P0 打通 → P1 平行探針各塊止損 → P2 合併 → P3(a/b/c) curriculum 聯合精修到 SOTA → P4 加 test-time 對齊 → P5 評測消融`。

---

## 8. 貢獻④：前饋初始化的 4D 全局對齊（推理時）

MonST3R 慢在「隨機初始化的全局優化」。Dyn-VGGT 前饋輸出已準，全局對齊退化為**少量迭代精修**。

優化變量：每幀 pose `P_t`、video depth `D_t`、scene flow `Δ_t`。目標：

```
min  E_flow(動態區光流一致) + E_static(靜態多視圖一致) + E_smooth(時序平滑)
```

- 用前饋 `m_{t,p}` 把光流項只施加在動態區、把多視圖剛性項只施加在靜態區（`1−m`）。
- 初始化好 → **20–50 次迭代即收斂**（MonST3R 需數百次）。
- 提供兩檔：**快速模式**（純前饋，0 迭代）與**精修模式**（帶對齊），呼應 VGGT demo 設計。

新建 `dyn_alignment.py` 實作 test-time 優化。

---

## 9. 評測協議（CVPR 標準實驗設計）

在 MonST3R 同套 benchmark 上同台競技。

**任務與指標**
| 任務 | 數據集 | 指標 |
|---|---|---|
| 相機 pose | Sintel / TUM-dynamics / ScanNet | ATE、RPE-trans、RPE-rot |
| Video depth | Sintel / Bonn / KITTI | Abs-Rel、δ<1.25、**時序一致性**（OPW / temporal AbsRel） |
| 動態分割 | DAVIS / Sintel | IoU、F-measure |
| 點雲/重建 | Dynamic Replica / TUM | Chamfer、Accuracy、Completeness |
| 效率 | 各序列 | 秒/序列 vs MonST3R |

**必做消融（reviewer 必問）**
1. 去掉 temporal-attn → 證時序模塊作用。
2. 去掉雙場分解（退回單一 world point，對應 §12-A1）→ 證運動解耦必要性。
3. 去掉 motion-mask 的動態屏蔽（pose `valid_frame` + point `(1−m)`）→ 證 pose/幾何 提升來源。
4. 0 迭代 vs 全局對齊（對應 §12-C9）→ 證前饋已強、對齊錦上添花。
5. 對比「VGGT 直接在動態數據微調」→ 證**架構創新**而非數據紅利。

**架構選擇消融（對應 §12 標 ★ 的決策）**
6. temporal 數量/間隔 `n_temporal × k`（§12-A3）→ 證插入密度的取捨。
7. special token 進 temporal 三檔（排除 / camera 進 / 全進且 register 無時間 RoPE，§12-B4）× pose ATE、video depth 一致性；附 register temporal-attention 可視化 → 證 register 是否學到時序語義。

**SOTA 對比**：MonST3R、CUT3R、DUSt3R、Robust-CVD、CasualSAM、原始 VGGT。
**預期賣點**：精度 ≥ MonST3R，速度快一個量級，且可擴展到數百幀（DUSt3R/MonST3R 做不到的長序列）。

---

## 10. 落地文件清單（直接對應本倉庫）

| 模塊 | 文件 | 改動 |
|---|---|---|
| 時序注意力 + 時間 RoPE | [aggregator.py](../vggt/models/aggregator.py)、[rope.py](../vggt/layers/rope.py) | 新增 `temporal_blocks`、`_process_temporal_attention`、`RotaryPositionEmbedding1D`（獨立 1D 時間 RoPE）、`aa_order` |
| 雙場輸出組裝 | [vggt.py](../vggt/models/vggt.py) | 接入 `motion_head` / `flow_head`，forward 組裝 `X_{t,p}` |
| 兩個新頭 | 新建 `vggt/heads/motion_head.py`、`flow_head.py`（或復用 DPTHead 直接在 vggt.py 實例化） | 復用 DPTHead |
| 新 loss | [loss.py](../training/loss.py) | 加 motion/flow/reproj/tsmooth 分支；pose 按靜態像素篩 `valid_frame`、point(`X^can`) 按 `(1−m)` 加權 |
| 數據 | [dynamic_dataloader.py](../training/data/dynamic_dataloader.py) | 返回 flow_gt / motion_mask / scene_flow |
| 尺度正規化（B6） | [normalization.py](../training/train_utils/normalization.py) | `normalize_camera_extrinsics_and_points_batch` 增 `scene_flow` 參數，比照 `world_points` 同步 `/ avg_scale`，確保 `Δ` 與 `X^can` 同尺度空間 |
| 全局對齊 | 新建 `dyn_alignment.py` | 前饋初始化的 test-time 優化 |
| Demo | [demo_viser.py](../demo_viser.py) | 動靜分色顯示 + 軌跡可視化 |

---

## 11. 建議實作順序（最小可跑路徑）

1. **架構地基**：`aggregator.py` + `rope.py` 加 temporal-attn + 時間 RoPE，warm-start 自預訓練權重，跑通 forward shape 測試。
2. **新頭接出**：`vggt.py` 新建並接入 motion / flow 兩頭，forward 組裝 `X_{t,p}`。
3. **loss 打通**：`loss.py` 加四項新損失與 pose 動態屏蔽，跑通一個 forward+backward smoke test。
4. **數據**：`dynamic_dataloader.py` 返回 flow/motion/scene_flow GT；接離線 SEA-RAFT 光流快取。
5. **全局對齊**：`dyn_alignment.py` test-time 優化。
6. **評測 + 消融**：依 §9 協議跑表。

---

## 12. 開放決策與預設值（Open Design Decisions）

實作時需要做選擇的模糊地帶。**A 類**影響架構正確性、**必須先拍板**（錯了訓不起來）；**B/C 類**可先用預設開跑、列為消融；**D 類**為超參。「消融」欄標 ★ 者建議進 §9 實驗表。

### A 類：必須先定（架構正確性）

| # | 決策 | 選項 | **採用（預設）** | 消融 | 對應節 |
|---|---|---|---|---|---|
| A1 | `X^can` 語義 | (a) 跨幀共享真 canonical / (b) per-frame 剛性分支 | **(b)** per-(t,p) 剛性預測 + 殘差位移；VGGT 無跨幀對應，(a) 落不了地 | ★（退回單一 world point，§9-2） | §4.1 |
| A2 | temporal 是否進 `output_list` | 進（→3C）/ 不進（維持 2C） | **不進**；只更新串流 `tokens`，head `dim_in=2*embed_dim` 與預訓練權重不動 | 否（正確性） | §3.2 |
| A3 | `aa_order` + temporal 數量 | 順序、插入間隔 k、`n_temporal` | **`["frame","temporal","global"]`，均勻 k=3，共 8 塊** | ★（k / n_temporal） | §3.2 |
| B6 | `Δ`/`X^can` 尺度正規化 | 各自歸一 / 同一 per-scene scale | **與 point map 同一 per-scene scale**：在 [normalize_camera_extrinsics_and_points_batch](../training/train_utils/normalization.py#L27) 增 `scene_flow` 參數，比照 `world_points` 同步 `/ avg_scale`（[normalization.py:106](../training/train_utils/normalization.py#L106)）；`L_reproj`/`L_flow^self` 因相機平移也已 `/avg_scale`（[normalization.py:107](../training/train_utils/normalization.py#L107)）而天然自洽。不歸一 → `X^can+m·Δ` 量綱錯、靜默不收斂 | 否（正確性） | §6.2 |

### B/C 類：先給預設、列消融

| # | 決策 | 選項 | **採用（預設）** | 消融 | 對應節 |
|---|---|---|---|---|---|
| B4 | special token 進 temporal? | (i) 全排除 / (ii) camera 進·register 排除 / (iii') 全進·register 不給時間 RoPE | **(iii')** camera+patch 進且帶時間 RoPE；register 進但時間維補 0（順序無關摘要槽）；γ=0 暖啟保證最壞無損 | ★（三檔 × pose ATE / depth 一致性；可視化 register temporal attn 佐證） | §3.1 |
| B5 | 偽標籤 `m*` 自舉穩定性 | `f^cam` 是否 detach；閾值是否 anneal | **detach** depth/pose 算 `m*`（防作弊塌縮，S2/P3c 全網訓時硬性必要）；有 GT 的合成集優先用 GT 算 `f^cam`（根除 bootstrapping）；`α_m,β_m` 前寬後緊 anneal（抗早期噪聲）。curriculum 緩解：S0/P1-B 的 depth/pose 凍結預訓練 → `f^cam` 一開始即可靠 | 可選 | §5.1/§6.2 |
| B7 | temporal RoPE 長序列外插 | 直接外插 / stride 增廣 / 頻率縮放 | **訓練隨機 frame stride 增廣**相對距離分布；頻率 base 沿用 100，必要時 NTK-aware 插值 | 可選（train S → test 數百幀） | §3.1 |
| B8 | causal vs bidirectional | 雙向 / 因果 | **雙向**（離線重建最準）；保留 causal flag 作串流 future work | 可選（附加實驗） | §3.2 |
| C9 | 全局對齊優化細節 | 變量範圍、聯合/交替、長序列滑窗 | **先最小版**（只優化 pose、固定 depth）再加碼 depth/flow；長序列滑窗 + 窗間 pose 拼接 | ★（0 迭代 vs 對齊，§9-4） | §8 |

### D 類：超參（預設值見對應節，多數需調）

| 超參 | 預設 | 備註 | 對應節 |
|---|---|---|---|
| 動態閾值 `τ` | 0.5 | 僅用於 `valid_frame` 計數；loss 主用連續 `m` 加權 | §5.2/§6.1 |
| 靜態數據混入比例 | 30% | 防遺忘，★ 消融 | §7 |
| EMA-teacher 蒸餾 | 開、權重待定 | 只在 `m≈0` 區蒸餾；是否必要待 ablate | §7 |
| `L_flow` 的 SEA-RAFT 配對 | 相鄰 `t→t+1` | 是否加 `t→t+k` 長程影響快取大小 | §6.2/§7 |
| 損失權重 `λ_m,λ_f,λ_rp,λ_ts,λ_tv` | 1.0/0.5/1.0/0.1/0.1 | 必調 | §6.3 |

**拍板優先序**：實作前必須定 **A1、A2、A3、B6**；其餘可先取預設值開跑、邊做邊收斂。
