# Dyn-VGGT：運動解耦的 4D 視覺幾何 Transformer（方法）

> ⚠️ **已封存（v1）**：雙場表示 `X=X^can+m·Δ` 已證實有雙線性不可辨識性與偷懶捷徑問題，**不再採用**。
> 沿革摘要見 [../dyn_vggt_history.md](../dyn_vggt_history.md)；現行方法見 [../dyn_vggt_method_v3.md](../dyn_vggt_method_v3.md)。

> Motion-Decoupled 4D Visual Geometry Grounded Transformer
>
> 一套在 **VGGT 前饋架構**上、把 **MonST3R 動態建模思想架構化**的動態影片三維重構方法。
> 目標：在動態場景中達到 **精度 ≥ MonST3R、速度快一個量級、可擴展到數百幀**，並滿足 CVPR 投稿的方法新穎性與實驗完整性標準。

**本文件聚焦「方法／理論設計」**。具體的程式對應、開放決策、訓練/評測執行 SOP 與目前實作狀態，見另一份文件 **[dyn_vggt_implementation.md](dyn_vggt_implementation.md)**。

---

## 0. TL;DR（一句話 novelty）

> Dyn-VGGT 把 VGGT 的「前饋多視圖幾何聚合」與 MonST3R 的「運動解耦動態建模」在**架構層**統一：用**時空聚合器**引入時序、用**靜態正則點 + 動態場景流的雙場分解**根治單一全局座標系缺陷、用**自監督動態 mask** 解耦相機與物體運動，並以**前饋輸出初始化全局對齊**，實現比 MonST3R 更快、更準、且可擴展到長序列的動態 4D 重建。

---

## 1. 問題診斷：VGGT 為何在動態影片下不如 MonST3R

VGGT 的剛性靜態假設藏在三處，每一處都對應一個失效機制：

### 1.1 單一全局座標系的 point map
VGGT 的 `world_points`（形狀 `[B,S,H,W,3]`）把所有幀像素統一回歸到**第一幀相機座標系**下的*靜態*點雲。
對運動物體，同一表面點在不同時刻位於不同的 3D 位置——單一全局座標無法同時表達；網絡被迫「平均」掉運動，導致動態區域深度/點雲糊化、拖影。

### 1.2 Alternating-Attention 無時間序
VGGT 的 `aa_order = ["frame", "global"]` 把 `S×P` 個 token 完全對稱地拼接做 global attention，而 RoPE 只編碼 **2D 空間位置** `(y, x)`。
**沒有任何幀序信息**：網絡不知道哪幀在前哪幀在後，無法建模運動的方向性與連續性，因此 video depth 逐幀獨立、時序閃爍。

### 1.3 相機 pose 與物體運動耦合
Camera head 回歸 `absT_quaR_FoV`，把整個場景當剛體。動態物體的光流被錯誤歸因到相機運動，污染 pose 估計。訓練端同樣耦合：pose 監督未對動態像素做屏蔽。

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

要點：
- 取特徵裡的一對數視為 2D 平面點，位置 `m` → 旋轉角 `θ_m = m·ω`；多對特徵用不同頻率 `ω_i = 1/base^(2i/d)`，低維轉得快管近距離、高維轉得慢管遠距離。
- **2D 版**把特徵維對半切，前半用 `y` 旋轉、後半用 `x` 旋轉，於是一個 patch token 同時帶上垂直/水平相對位置。
- RoPE **只作用於 q、k**，v 不旋轉——位置只影響「誰跟誰相關」，不影響「傳遞什麼內容」。
- 頻率表以整數位置索引查表，故 **positions 必須是整數索引**——這決定時間軸要用整數幀索引。

### 3.1 時間位置編碼（獨立 1D 時間 RoPE，與空間 RoPE 並存）
**不採用** 把 pos 從 `(y,x)` 擴成 `(y,x,t)` 的 3D-RoPE：現有 2D RoPE 的「前半 y、後半 x」對半切是寫死的，改成三等分會打亂預訓練權重學到的特徵分工，**warm-start 失效**。

**做法**：新增一個**獨立的 1D 時間 RoPE，只在 temporal attention 內作用**，空間 RoPE 一個字不改。
- **位置用整數幀索引 `t ∈ {0,…,S−1}`**（RoPE 只看相對差 `t_i−t_j`，整數已能完整表達「隔幾幀」）。
- **frame / global attention 維持原 2D 空間 RoPE 不變**；時間相位只在 temporal attention 注入，warm-start 完整保留。
- **special token 的時間處理**：camera/patch token 進 temporal 並帶時間 RoPE（軌跡需有序時序）；register token 進 temporal 但時間維補 0（不給時間 RoPE）→ 變成「順序無關的時序全局摘要槽」。

### 3.2 時序注意力塊（核心）
在 alternating-attention 的 frame / global 之間插入第三種注意力 `"temporal"`：把 token 重排為 `(B*P, S, C)`，**僅沿時間軸**做注意力，捕捉同一空間位置的時序演化（運動連續性、軌跡）。

**三者正交分工**：
- frame-attn → 單幀內部空間幾何；
- temporal-attn → 同一空間位置的「軌跡 / 運動」；
- global-attn → 跨幀跨空間的全局幾何一致性。

三者復用 VGGT 既有的 `Block` 結構。**warm-start（LayerScale 零初始化）**：Block 殘差為 `x = x + γ·attn(x)`，把 temporal block 的 γ 初始化為 0 → 初期輸出恆等於輸入 → **整個網絡第一步嚴格等於原 VGGT**，不擾動既有幾何；訓練中 γ 由 0 漸增，時序機制「漸進通電」。

> 顯存：temporal 的序列長僅 `S`、複雜度 `O(B·P·S²)`，遠小於 global 的 `O((S·P)²)`。
> temporal 只更新串流 token、不改變交給 head 的特徵介面（實作關鍵見 implementation 文件）。

---

## 4. 貢獻②：運動解耦的雙場表示（理論支點）

直接解決 §1.1「單一全局座標系無法表達運動物體」。對像素 p 在時刻 t，**不再**只預測單一全局點，而是分解：

```
X_{t,p} = X^can_{t,p}  +  m_{t,p} · Δ_{t,p}
          └ 剛性分支 ┘     └動態概率┘ └殘差位移┘
```

> **語義（per-frame 剛性分支）**：VGGT **沒有跨幀像素對應**（無 track），無法定義「跨幀共享、同一世界點唯一」的真 canonical 點。故採**逐像素逐幀（per-(t,p)）的剛性分支**語義：
> - `X^can_{t,p}`：**「假設此像素為靜態時」該落的世界座標**（即原 point head 的剛性多視圖預測），per-frame 輸出、不要求跨幀共享。
> - `Δ_{t,p} ∈ R³`：把剛性預測**修正到第 t 幀真實位置**的 3D 殘差位移，由 Scene-Flow 頭給出。
> - `m_{t,p} ∈ [0,1]`：動態概率。靜態區 `m≈0` 退化為原 VGGT（`X≈X^can`）；動態區 `m≈1` 殘差生效。
>
> **可逐像素直接實作、不需跨幀對應**。監督分流（見 §6）：靜態區 `(1−m)` 直接拿 GT world point 監督 `X^can`；動態區不直接監督 `X^can`（隱變量），而讓組裝點 `X_{t,p}` 經 dense 監督 + `L_reproj` 對齊觀測。

**雙路徑融合**：靜態幾何走「VGGT 多視圖三角化（數百幀聚合）」路徑、動態幾何走「單目深度 + scene flow」路徑，用 `m` 做 **soft gating** 融合——這就是「直接結合 MonST3R 做法」的架構化表達。

---

## 5. 貢獻③：動態分割頭 + 場景流頭

新增兩個輕量 DPT 頭：**Motion-Seg 頭**輸出每像素動態概率 `m`，**Scene-Flow 頭**輸出 3D 殘差位移 `Δ`。

### 5.1 動態 mask 的自監督信號（無需動態標註）
把 MonST3R 的「光流一致性判定動態」做成**可微自監督**：由預測深度 + pose + 剛體假設算出相機運動誘導的「期望光流」`f^cam`，與真實光流 `f^gt`（資料集內建或離線 RAFT）對比，殘差大者為動態：

```
r_{t,p} = ‖ f^gt_{t,p} − f^cam_{t,p} ‖
m*_{t,p} = σ( α_m · (r_{t,p} − β_m) )      # 偽標籤
```

- 無動態 GT 的數據集：`m*` 作偽標籤監督 Motion-Seg 頭。
- 有合成動態 GT（PointOdyssey / Waymo / Sintel-derived）：直接用 GT mask 監督。

### 5.2 動態屏蔽（pose 提升來源）——兩處獨立改動
1. **pose（序列級開關）**：把 `valid_frame` 由「有效點數>100」改為按**靜態像素**計數 `Σ_p point_masks·𝟙[m<τ] > 100`，使 pose 只由靜態像素監督。
2. **正則點（像素級權重）**：`L_point` 走 per-pixel 回歸，動態屏蔽落在**像素權重 `(1−m)`** 上，避免動態 GT 污染靜態正則場。

兩者的 `m` 皆取 `detach`，只當開關/權重用，不讓 pose/point 的梯度反灌 motion 頭。

---

## 6. 損失函數

### 6.0 符號約定
- 序列 `S` 幀，像素 `p`，幀索引 `t`；`u_{t,p}` 為像素座標。
- `M_{t,p}∈{0,1}`：GT 有效像素 mask（`point_masks`）；`m_{t,p}∈[0,1]`：預測動態概率；`m*`：其偽標籤。
- 相機外參 `P_t=[R_t|τ_t]`、內參 `K_t`；投影 `π(·)`、反投影 `π⁻¹(u,d)`。
- 雙場組裝點：`X_{t,p} = X^can_{t,p} + m_{t,p}·Δ_{t,p}`；`𝒱 = {(t,p): M_{t,p}=1}`。

### 6.1 既有三項（沿用 confidence-weighted 形式）
核心回歸子（depth/point 共用），對場 `y`（預測 `ŷ`、置信 `c`）：
```
ℓ_reg  = mean_{(t,p)∈𝒱} ‖ y − ŷ ‖₂
ℓ_conf = mean_{(t,p)∈𝒱} [ γ·‖ y − ŷ ‖₂·c − α·log c ]      (預設 γ=1.0, α=0.2)
```
多尺度梯度/法向項：`ℓ_grad = mean_s ( 1 − cos∠( n(ŷ^(s)), n(y^(s)) ) )`。於是：
```
L_depth = ℓ_conf^D      + ℓ_reg^D      + ℓ_grad^D       # 監督每幀真實深度 D_{t,p}（動靜皆可）
L_point = ℓ_conf^{Xcan} + ℓ_reg^{Xcan} + ℓ_grad^{Xcan}  # 監督 X^can，逐像素乘 (1−m) 權重
```
> **為何 `L_point` 乘 `(1−m)`**：GT `world_points` 對動態物體記錄的是**第 t 幀的移動位置**，而 `X^can` 是**靜態正則點**。直接拿移動 GT 回歸 `X^can` 在動態區是錯誤監督。故 `X^can` 只在靜態區由 GT 監督；動態區幾何改由組裝點經 `L_reproj`（§6.2-3）約束。`L_depth` 不需 `(1−m)`（深度動靜都可觀測）。

**相機項（動態屏蔽版）** `L_cam^stat`。pose 編碼拆成 `T∈ℝ³`、四元數 `q∈ℝ⁴`、FoV，逐分量 L1，跨 `n` 個 refine stage 以 `γ^{n−i−1}` 加權：
```
L_cam = (1/n) Σ_i γ^{n−i−1} ( w_T‖T̂^(i)−T‖₁ + w_R‖q̂^(i)−q‖₁ + w_FL‖f̂^(i)−f‖₁ )   (w_T=1, w_R=1, w_FL=0.5, γ=0.6)
valid_frame(t) = [ Σ_p M_{t,p}·𝟙[ m_{t,p} < τ ] > 100 ]   (τ=0.5, m 取 detach)
```

### 6.2 新增四項

**(1) 動態分割** `L_motion`（BCE）：
```
L_motion = − mean_{(t,p)∈𝒱} [ m*·log m + (1−m*)·log(1−m) ]
```
偽標籤 `m*` 由相機誘導剛體光流與真實光流殘差生成（§5.1）：
```
f^cam_{t,p} = π( P_{t+1} P_t⁻¹ π⁻¹(u_{t,p}, D_{t,p}) ) − u_{t,p}
m*_{t,p}    = σ( α_m·( ‖f^gt − f^cam‖₂ − β_m) )
```
有合成動態 GT 的數據集則 `m*` 直接取 GT 二值 mask。

**(2) 場景流** `L_flow`（訓練 flow 頭的 `Δ`）：

**主路徑 — dense「組裝點 vs GT world_points」（只需 depth+相機，所有訓練集可用，無需光流）**：
GT `world_points` 在第 t 幀記錄的就是每像素表面**當下時刻**的真實世界座標（動態物體亦然），故組裝點 `X_{t,p}` 有 dense GT 目標：
```
L_flow^dense = ℓ_conf( X^can_{t,p} + m·Δ_{t,p} ,  world_points^gt_{t,p} ;  conf=scene_flow_conf )
```
搭配 `(1−m)`-masked 的 `L_point`（只在靜態區釘住 `X^can`），動態區的缺口被迫由 `m·Δ` 補上——把「動態監督從 `X^can` 搬到 `X^can+m·Δ`」，是雙場解耦的直接落地。

**可選加強 — 稀疏 3D scene-flow GT**（如 PointOdyssey `trajs_3d`）：`L_flow^sparse = Σ valid·huber(Δ − Δ^gt)`。

> **光流角色已收斂**：dense 主路徑只需 depth+相機，**supervised flow 不再需要 2D 光流**。2D 光流 `f^gt` 現在**僅**用於「無動態 mask 的數據集」生成 `m*`。

**(3) 跨時刻重投影一致性** `L_reproj`（核心閉環，把 point/flow/pose/motion 四頭聯合約束）：
```
L_reproj = mean_{(t,p)∈𝒱}  ‖ π( K_t [R_t|τ_t] (X^can_{t,p} + m_{t,p}·Δ_{t,p}) ) − u_{t,p} ‖_δ
```
靜態區退化為「正則點投影回原像素」的多視圖約束；動態區位移生效。**像素誤差除以影像對角線正規化**（分數單位）以免壓過其他項。

**(4) 時間平滑** `L_tsmooth`（抑制 video depth 閃爍）：
```
L_tsmooth = mean_{t,p} M·‖ Δ_{t+1} − 2Δ_t + Δ_{t−1} ‖₁  +  λ_tv · Σ | m_{t,p} − m_{t−1,p} |
```

### 6.3 總損失與按 stage 的權重開關
```
L = L_cam^stat + L_depth + L_point + λ_m·L_motion + λ_f·L_flow + λ_rp·L_reproj + λ_ts·L_tsmooth
```
建議起始配比：`λ_m=1.0, λ_f=0.5, λ_rp=1.0, λ_ts=0.1, λ_tv=0.1`。
- **S0**：只開 `L_motion + L_flow`。
- **S1**：加開 `L_reproj + L_tsmooth`，並切 `L_cam → L_cam^stat`（動態屏蔽）。
- **S2**：全 7 項到位、調最終配比。

---

## 7. 訓練策略（課程學習）

VGGT 是強預訓練 baseline，**絕不從頭訓**。採 MonST3R 式「凍結 encoder、小數據微調」+ 課程學習。

> **關鍵前提：「架構模塊存不存在」≠「該 stage 有沒有訓練/啟用它」。**
> Dyn-VGGT 是**單一模型**——temporal-attn、雙場、motion/flow 兩頭在 S0 就**結構性存在**。三個 stage 改變的是 (a) 哪些參數解凍、(b) 哪些 loss 啟用。**S0 並非一次打開所有機制**，而是漸進解凍以保護預訓練的靜態幾何先驗。

### 7.1 三階段總覽
| 階段 | 凍結 | 解凍訓練 | 數據 | 目的 |
|---|---|---|---|---|
| S0 | aggregator（含 temporal 塊）+ 原三頭 | 僅 motion / flow 兩個新頭 | 動態合成集 | 對齊新頭，不破壞原幾何 |
| S1 | DINOv2 patch embed | temporal 塊 + 原三頭 + 兩新頭 | 動態為主 + 少量靜態 | 學時序與運動解耦 |
| S2 | 無（全網，小 lr 1e-5） | 全部 | 動靜混合 | 聯合精修 |

### 7.2 各 stage「新增了什麼」
| | 解凍訓練 | 新打開的 loss | 新生效的機制 |
|---|---|---|---|
| **S0** | motion / flow 頭 | `L_motion`, `L_flow` | 兩頭對齊；雙場僅前向**不參與 loss**；temporal γ=0 ≈ 不起作用 |
| **S1** | + temporal 塊、原三頭 | `L_reproj`, `L_tsmooth` | 雙場閉環生效；動態屏蔽開啟（pose `valid_frame` + point `(1−m)`）；temporal 解凍學幀序 |
| **S2** | + DINOv2 patch embed（全網） | 全 7 項到位 | 端到端聯合精修 + 防遺忘 |

**設計理由（為何不在 S0 全開）**：S0 兩頭隨機初始化，若一開始就接 `L_reproj` 閉環，隨機的 `Δ`/`m` 會把梯度灌回已訓好的幾何頭、污染破壞原 VGGT；故先凍其餘、只養兩頭到可用，S1 才接閉環與屏蔽、解凍 temporal 一起學，S2 全網精修。

**防遺忘**：靜態數據混入（~30%）保持 `m≈0` 區原 VGGT 行為；可加 EMA-teacher 蒸餾舊 VGGT 在靜態幀輸出。

---

## 8. 貢獻④：前饋初始化的 4D 全局對齊（推理時）

MonST3R 慢在「隨機初始化的全局優化」。Dyn-VGGT 前饋輸出已準，全局對齊退化為**少量迭代精修**。優化變量：每幀 pose `P_t`、video depth `D_t`、scene flow `Δ_t`：
```
min  E_flow(動態區光流一致) + E_static(靜態多視圖一致) + E_smooth(時序平滑)
```
用前饋 `m` 把光流項只施加在動態區、把多視圖剛性項只施加在靜態區（`1−m`）。初始化好 → **20–50 次迭代即收斂**（MonST3R 需數百次）。提供**快速模式**（純前饋，0 迭代）與**精修模式**（帶對齊）兩檔。

---

## 一句話總結 novelty

> Dyn-VGGT 把 VGGT 的前饋多視圖聚合與 MonST3R 的運動解耦在架構層統一：時空聚合器引入時序、靜態正則點+動態場景流的雙場分解根治單座標系缺陷、自監督動態 mask 解耦相機與物體運動、前饋初始化全局對齊——比 MonST3R 更快更準、且可擴展到長序列。
