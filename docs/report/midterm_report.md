# 期中報告：方法與公式

> 2026-08-26。方法與公式的完整版，供期中報告撰寫／投影片取材。
> 數字一律引用 [results/medical.md](../results/medical.md)（醫學線）與 [results/natural.md](../results/natural.md)（Sintel／PO 消融），本文不重算、不下新結論。
> 執行指令見 [execution.md](../ops/execution.md)。

---

## 1. 模型概觀

本模型是一個**前饋式（feed-forward）多視角幾何 transformer**：輸入一段影像序列，單次前向同時輸出每一幀的相機參數與深度圖，不需要 per-scene 最佳化、不做 test-time optimization。

**輸入**：影像序列 $\{I_t\}_{t=1}^{S}$，$I_t \in \mathbb{R}^{H\times W\times 3}$。

**輸出**：

$$
\{\hat{\mathbf{g}}_t\}_{t=1}^{S}\ \ (\text{相機參數}),\qquad \{\hat{D}_t\}_{t=1}^{S}\ \ (\text{深度圖})
$$

> **不輸出 point map，也不輸出 track。** `point_head` 與 `track_head` 在設定中關閉（`enable_point: False` / `enable_track: False`），模型裡不建立、checkpoint 中也不帶這兩顆頭的權重。報告中出現的點雲是**評測階段把預測深度以內外參反投影**得到的（`benchmark/eval_lesion.py` 的 `save_cloud`），因此點雲與深度圖必然一致，但它不是模型的第三個輸出。

**模組清單**：

| 模組 | 角色 | 開關 |
|---|---|---|
| DINOv2 patch embed | 影像 → patch token（全程凍結） | — |
| Alternating-attention aggregator | frame／temporal／global 三種 attention 交替，24 個 block | `enable_temporal` |
| **Motion Gate**（模組 A） | 中段預測 per-patch 動態 logit，在後段 global attention 對 camera/register query 加負 bias | `enable_gate` |
| Camera head | 讀 camera token，迭代 refine 4 次輸出相機參數 | `enable_camera` |
| Depth head | 讀 patch token 輸出深度（醫學線全程凍結） | `enable_depth` |

**兩條應用線**：

- **通用動態場景**（Sintel／PointOdyssey／TartanAir／Waymo／Spring）——有動態標註，gate 以 BCE 監督。
- **醫學內視鏡**（SCARED／C3VD／私人病灶・胃資料）——**沒有動態標註**，gate 改由相機 loss 訓練。本報告的主要實驗線。

---

## 2. 架構

### 2.1 影像編碼與 token 佈局

每幀影像經凍結的 DINOv2 ViT-L patch embed（patch size $p=14$，$C=1024$）得到 $P_{\text{patch}} = \frac{H}{14}\cdot\frac{W}{14}$ 個 patch token，再前置 special token：

$$
\mathbf{z}_t = \big[\underbrace{\mathbf{c}_t}_{1\ \text{camera}} ,\ \underbrace{\mathbf{r}_t^{(1..4)}}_{4\ \text{register}},\ \underbrace{\mathbf{x}_t^{(1..P_{\text{patch}})}}_{\text{patch}}\big] \in \mathbb{R}^{P\times C},\qquad P = 5 + P_{\text{patch}}
$$

`patch_start_idx` $= 5$。camera 與 register token 各有兩份參數：index 0 只給 frame 1、index 1 給其餘 $S-1$ 幀——因為相機參數是**相對第一幀**表達的，frame 1 需要一個可區分的 query。

三種 token 的去向：

| token | 數量／幀 | 誰讀它 |
|---|---|---|
| **camera** | 1 | camera head |
| **register** | 4 | 沒有任何 head 讀，是 attention 過程的 scratch space |
| **patch** | $P_{\text{patch}}$ | depth head |

⚠️ **token 在序列軸上是 frame-interleaved 的**：$[\text{frame}_1(\text{special},\text{patch}) \mid \text{frame}_2(\cdots) \mid \cdots]$——special token 散在每幀開頭，不是集中在最前面。這一點在模組 A 的實作中是關鍵（§3.3）。

### 2.2 Alternating Attention

aggregator 共 24 個 block，交替三種 attention：

| 類型 | reshape | 語意 |
|---|---|---|
| **frame** | $(B\!\cdot\!S,\ P,\ C)$ | 每幀內部的空間 attention |
| **temporal** | $(B\!\cdot\!P,\ S,\ C)$ | 每個空間位置沿自己的 $S$ 幀互看（§4.1） |
| **global** | $(B,\ S\!\cdot\!P,\ C)$ | 所有幀所有 token 攤平一起做 |

attention 本體帶一個加性 bias 項 $B$：

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B\right),\qquad \mathbf{o}_i = \sum_j A_{ij}\,\mathbf{v}_j
$$

$B$ 只在 global attention、且只在 camera/register 的 query 列上非零，由模組 A 產生（§3.2）；其餘所有情況 $B \equiv 0$。

### 2.3 Camera head

camera head 取每幀的 camera token $\mathbf{z}_t[0]$，經 4 層 trunk **迭代 refine 4 次**，輸出 4 個 stage 的 pose encoding。編碼採 `absT_quaR_FoV`，9 維：

$$
\mathbf{g}_t = \big[\; \mathbf{t}_t \in \mathbb{R}^3 \;\big|\; \mathbf{q}_t \in \mathbb{R}^4 \;\big|\; (\mathrm{fov}_x, \mathrm{fov}_y) \in \mathbb{R}^2 \;\big]
$$

$\mathbf{t},\mathbf{q}$ 定義 world-to-camera 外參 $E_t = [R(\mathbf{q}_t) \mid \mathbf{t}_t]$，以第一幀為世界座標原點（$E_1 = [I \mid 0]$）。4 個 stage 全部進 loss，以 $\gamma$ 加權（§5.1）。

### 2.4 相機參數與場景運動的耦合

在 global attention 中，每一幀的 camera token 會 attend 到**所有幀的所有 patch token**：

$$
\mathbf{o}[\mathbf{c}_t] = \sum_{s=1}^{S}\sum_{j=1}^{P_{\text{patch}}} A[\mathbf{c}_t,\ \mathbf{x}_s^{(j)}]\cdot \mathbf{v}(\mathbf{x}_s^{(j)}) + (\text{special keys})
$$

其中 $j$ 跑遍所有 patch。若場景中有移動的物體或形變的組織，那些 patch 的運動會被一併加權平均進 camera token 的表示，再送進 camera head。**模組 A 的作用就是在這個加權上動手**：讓 camera token 對「不符合相機剛體運動」的 patch 降低權重。

在 loss 端遮罩動態區無法達到同樣效果——那只是不去懲罰它，相機參數在前向時就已經聚合完成了。因此遮罩必須施加在 **attention 端**。

---

## 3. 模組 A：Motion-Gated Camera Aggregation

### 3.1 中段 Gate Predictor

門控訊號必須在**聚合進行中**就可用，而不是等最終特徵算完。因此在 aggregator 中段（第 $k=7$ 個 aa-block 完成後，約全深度 $1/3$）插入一個輕量 MLP：

$$
g_{t,j} \;=\; W_2\,\phi\big(W_1\,\mathrm{LN}(\mathbf{x}^{(k)}_{t,j})\big) \;\in\; \mathbb{R},
\qquad W_1 \in \mathbb{R}^{\frac{C}{4}\times C},\ W_2 \in \mathbb{R}^{1\times \frac{C}{4}},\ \phi = \mathrm{GELU}
$$

- **輸入**是中段的 contextualized patch token（已跑過 8 個 aa-block），切掉 camera+register，只留 patch。
- **輸出** $g \in \mathbb{R}^{B\times S\times P_{\text{patch}}}$，每個 patch 一個動態 logit。$\sigma(g)$ 同時是一張免費的動態機率圖。
- **末層 zero-init**（$W_2 = 0,\ b_2 = 0$）→ 訓練第 0 步 $g \equiv 0$ → bias $\equiv 0$ → 門控在初始化時是**恆等的 no-op**，隨訓練漸進通電。
- **解析度天然對齊**：$g$ 定義在 patch 格上，而 attention 的 key 本來就是 patch token → bias 直接套用，零插值。
- **放在 $1/3$ 深度**：太早則特徵尚未成熟，太晚則後面沒有幾個 global block 可供門控作用。$1/3$ 讓後續仍有 $2/3$ 的 global block 受其影響。

### 3.2 Attention Bias

bias 只加在 **camera + register token 的 query 列**、對 **patch token 的 key**；其餘位置為 0：

$$
B[i, j] =
\begin{cases}
b(g_j), & i \in \text{camera/register queries},\ j \in \text{patch keys}\\[2pt]
0, & \text{otherwise}
\end{cases}
$$

$$
\boxed{\;b(g) \;=\; \min\big(0,\ \ \mathrm{softplus}(0) - \mathrm{softplus}(g)\big) \;=\; \min\big(0,\ \ \ln 2 - \ln(1+e^{g})\big)\;}
$$

**這個形式同時滿足三個性質**：

1. **機率意義**。$-\mathrm{softplus}(g) = \ln \sigma(-g) = \ln P(\text{static})$。加進 softmax 的 logit 等價於把 attention 權重乘上靜態機率：

$$
A[\mathbf{c}, \mathbf{x}_j] \;\propto\; \exp\!\left(\tfrac{\mathbf{q}_c^\top \mathbf{k}_j}{\sqrt d}\right)\cdot \underbrace{2\,\sigma(-g_j)}_{\text{經 clamp 後} \le 1}
$$

即「按靜態機率重新加權」——單邊、soft、只壓不升。

2. **零點精確**。$+\mathrm{softplus}(0) = \ln 2$ 的 offset 讓 $g=0$ 時 $b$ **精確為 0**，門控在初始化時完全不擾動前向。（若用裸的 $-\mathrm{softplus}(g)$，$g=0$ 會給 $-\ln 2$，把每個 patch key 相對 special key 的權重減半。）

3. **Suppress-only**。$\min(0,\cdot)$ 夾上界，使有信心的靜態 patch（$g<0$）得到 bias $=0$（滿權重），不會被**放大**。

極限行為：

| patch | $g$ | $b(g)$ | softmax 後的權重 |
|---|---|---|---|
| 有信心靜態 | $\to -\infty$ | $\to 0$ | 維持原樣 |
| 初始化（zero-init） | $0$ | **$=0$（精確）** | 門控為 no-op |
| 動態 | $\to +\infty$ | $\to -\infty$ | $\to 0$（排除） |

**結果**：camera token 在後 $2/3$ 的 global block 中結構性地只聚合靜態 patch（訓練與推論皆然）。**patch↔patch 的 attention 完全不受影響**，depth head 的輸入特徵不變。

### 3.3 實作：attention 依 query 拆兩條路徑

把整個 $[N,N]$（$N = S\cdot P$）的浮點 bias 矩陣交給 `F.scaled_dot_product_attention`，會讓 flash kernel fallback 到 memory-heavy 的 math backend，顯存成長為 $O(N^2)$。因此把 attention 依 query 拆成兩條：

$$
\underbrace{\mathrm{Attn}(Q_{\text{patch}},\,K,\,V)}_{\text{Path 1: } S\cdot P_{\text{patch}}\ \text{queries，無 bias，走 flash}}
\quad\Big\Vert\quad
\underbrace{\mathrm{Attn}(Q_{\text{special}},\,K,\,V;\ B)}_{\text{Path 2: } 5S\ \text{queries，帶 bias，小 op}}
$$

Path 2 只有 $5S$ 個 query（相對 $S\cdot P_{\text{patch}}$ 可忽略），即使不走 flash 也無所謂——**門控的顯存成本因此 $\approx 0$**。

因為 token 是 frame-interleaved 的（§2.1），拆分必須**逐幀 gather**：

```
q.view(B, H, S, P, D)[:, :, :, :5, :]   →  special queries
q.view(B, H, S, P, D)[:, :, :, 5:, :]   →  patch queries
```

key 端的 bias 也照 interleaved 順序組：每幀的 patch 欄位填 $b(g)$、special 欄位填 0，再 reshape 成 $[B,1,1,N]$ 廣播到所有 head 與所有 special query。

### 3.4 Gate 的兩種訓練方式

**(a) 有動態標註的域——BCE 監督**

$$
L_{\text{gate}} = \mathrm{BCE}\big(\sigma(g),\ m^*_{\text{patch}}\big),
\qquad m^*_{\text{patch}} = \mathrm{AvgPool}_{14\times 14}\big(m^*\big) \in [0,1]
$$

此路徑上 $g$ 不 detach，梯度回流 predictor。同時 **bias 路徑 detach**——相機 loss 不會反灌門控，避免門控為了讓 pose 好看而任意標記動態區。

**(b) 無動態標註的域（SCARED／C3VD）——由相機 loss 訓練**

醫學資料沒有動態標註，$L_{\text{gate}}$ 無從計算；若 bias 仍 detach，門控將完全不可訓練。因此打開兩個設定：

**`gate_pose_grad = True`** —— 移除 detach，讓 $L_{\text{cam}}$ 直接訓練 gate：

$$
\frac{\partial L_{\text{cam}}}{\partial g_j} = \frac{\partial L_{\text{cam}}}{\partial b_j}\cdot b'(g_j),
\qquad b'(g) = \begin{cases} -\sigma(g), & g > 0 \\ 0, & g < 0 \end{cases}
$$

門控學到的量因此從「哪裡在動」變成「**哪些 patch 對相機估計有害**」。

**`gate_leaky = 0.1`** —— 上式的 clamp 在 $g<0$ 整段是平的，梯度恆為 0：任何被推到 $g<0$ 的 patch 從此不再收到梯度，是一個吸收態（dying-ReLU 的同型問題）。改用 leaky clamp 給那一半一個斜率：

$$
b_\lambda(g) = \begin{cases} x, & x \le 0\\ \lambda\,x, & x > 0\end{cases},
\qquad x = \ln 2 - \mathrm{softplus}(g),\ \ \lambda = 0.1
$$

代價是有信心的靜態 patch 最多被放大 $\lambda\ln 2 \approx 0.07$（而非精確 0）。分界用 $x \le 0$ 而非 $x<0$，讓 $g=0$（zero-init 的起點）走非 leaky 分支。

> ⚠️ (a) 與 (b) 監督的是**不同的量**，報告中不宜混為一談。(b) 的門控是否收斂到有意義的東西、或只是退化成一個 attention 溫度旋鈕，目前是**開放問題**——`gate_g_median` 與 `gate_frac_active` 兩個 log 即為監控此事而存在。

### 3.5 動態標籤 $m^*$（給有標註的域）

標籤的定義必須是**運動**而非**外觀**，否則跨域即失效。兩種來源，語義一致（世界真的在動）：

**(a) instance × GT scene-flow（訓練用）** —— 以 GT world track 判斷每個 instance 連通塊是否位移，動則整塊填滿。靜態世界點的位移恆為 0，因此沒有門檻取值上的快慢兩難。

**(b) 幾何光流殘差（測試／跨域用）** —— 假設場景全靜態時，相機誘導的光流為

$$
\mathbf{f}^{\text{cam}}(\mathbf{u}) \;=\; \pi\!\big(P_{t+1}P_t^{-1}\,\pi^{-1}(\mathbf{u}, D_t)\big) - \mathbf{u}
$$

$$
m_{\text{geo}}(\mathbf{u}) = \mathbf{1}\Big[\ \big\Vert \mathbf{f}^{\text{gt}}(\mathbf{u}) - \mathbf{f}^{\text{cam}}(\mathbf{u})\big\Vert > \max\big(\tau_{\text{abs}},\ \alpha\,\Vert \mathbf{f}^{\text{gt}}(\mathbf{u})\Vert\big)\ \Big]
$$

相對門檻 $\alpha\Vert\mathbf{f}^{\text{gt}}\Vert$ 是必要的：純絕對門檻在高速鏡頭序列會讓整片背景被判為動態。

**跨域證據**（只訓練 gate predictor，其餘全凍）：

| 標籤定義 | PO AUC | Sintel AUC |
|---|---|---|
| 光流殘差（單一絕對門檻） | 0.984 | **0.584** |
| **instance × scene-flow** | 0.926 | **0.767** |

外觀式的 instance 分割（連牆、地板、靜止前景都標記）會讓門控學到「這個資料集的室內外觀 → 動態」，換域即失效；改用運動定義的標籤後 Sintel AUC 提升 0.183，PO 只掉 0.058。

> **判讀門控品質一律用 AUC／F1，不看 BCE**：pool 後的軟標籤在 boundary patch 有不可約的 floor，val BCE 會看似 overfit 卻與高 AUC 並存。

---

## 4. 模組 B：Temporal Attention 與軌跡平滑

### 4.1 Temporal Attention

在 frame 與 global 之外，加入一種沿**純時間軸**的 attention：每個空間位置 attend 自己的 $S$ 幀。

$$
(B, S, P, C) \;\xrightarrow{\ \text{permute}\ }\; (B\cdot P,\ S,\ C)
\;\xrightarrow{\ \text{Attn along } S\ }\;
(B\cdot P,\ S,\ C) \;\xrightarrow{\ \text{permute}\ }\; (B\cdot S,\ P,\ C)
$$

| 項目 | 設定 |
|---|---|
| `aa_order` | `[frame, temporal, global]` |
| 插入頻率 | 每 `temporal_every=3` 個 aa-block 一個 → $24/3 = 8$ 個 temporal block |
| head 介面 | temporal block 只更新 streaming token、**不 emit intermediate** → head 輸入維持 $[B,S,P,2C]$，下游維度不變 |
| 位置編碼 | 獨立的 1D 時間 RoPE，與空間 2D RoPE 分開。camera token 與 patch token 給真實 frame index，register token 給 0 |

**Identity 初始化**：每個 temporal block 的 LayerScale $\gamma$ 初始為 0，

$$
\mathbf{x} \leftarrow \mathbf{x} + \gamma \cdot \mathrm{Attn}\big(\mathrm{LN}(\mathbf{x})\big),\qquad \gamma|_{t=0} = 0
$$

因此整塊在初始化時是恆等映射；訓練中 $\gamma$ 漸長、temporal 漸進通電。

**與模組 A 的互補關係**：若某幀幾乎全是動態區，門控排除大多數 patch 後 camera token 可能資訊不足。temporal attention 讓它從**較靜態的鄰幀**取得幾何上下文。兩者正交但互補。

### 4.2 Camera-Smooth Loss

對**預測的相機序列**施加一個時序平滑先驗，罰其二階（加速度）不連續。此項**不使用 GT pose**，是純自參照的正則。

令第 $i$ 個 refine stage 的預測為 $(\hat{\mathbf{t}}^{(i)}_s, \hat{\mathbf{q}}^{(i)}_s)$，$\Delta t_s = \mathrm{id}_{s+1} - \mathrm{id}_s$ 為真實時序間隔：

**速度（一階）**

$$
\mathbf{v}^T_s = \frac{\hat{\mathbf{t}}_{s+1} - \hat{\mathbf{t}}_s}{\Delta t_s},
\qquad
\mathbf{v}^R_s = \frac{\tilde{\mathbf{q}}_{s+1} - \tilde{\mathbf{q}}_s}{\Delta t_s}
$$

**加速度（二階）**

$$
\mathbf{a}^{T}_s = \mathbf{v}^T_{s+1} - \mathbf{v}^T_s,\qquad \mathbf{a}^{R}_s = \mathbf{v}^R_{s+1} - \mathbf{v}^R_s
$$

$$
\boxed{\;
L_{\text{camera\_smooth}} = \frac{1}{n}\sum_{i=1}^{n} \gamma^{\,n-i}\Big[\ w_T\cdot \overline{\vert \mathbf{a}^{T,(i)}\vert}\ +\ w_R\cdot \overline{\vert \mathbf{a}^{R,(i)}\vert}\ \Big],
\qquad n=4,\ \gamma=0.6,\ w_T=w_R=1
\;}
$$

$\overline{\vert\cdot\vert}$ 為對所有有效 $(b,s)$ 的遮罩平均。三個實作細節：

**(1) $\Delta t$ 正規化** —— 訓練 clip 的抽幀間距不規則、且可能重複（`batch["ids"]` 為真實時序 index），故用 $\Delta T/\Delta t$ 而非裸的 $\Delta T$。$\Delta t = 0$ 的 pair 排除；$k$ 階差分要求其底下 $k$ 個 pair 全部有效。

**(2) 四元數半球對齊** —— $\mathbf{q}$ 與 $-\mathbf{q}$ 表示同一旋轉，差分前先逐幀對齊：

$$
\tilde{\mathbf{q}}_1 = \mathbf{q}_1,\qquad
\tilde{\mathbf{q}}_s = \mathrm{sign}\big(\langle \tilde{\mathbf{q}}_{s-1}, \mathbf{q}_s\rangle\big)\cdot \mathbf{q}_s
$$

否則 sign flip 會被誤判為巨大的旋轉跳動。這是逐分量的簡化平滑，不是嚴格的 geodesic。

**(3) FoV 不平滑**，且需 $S \ge 3$（故取樣張數下界取 4）。

一階項（罰速度本身）也已實作（`orders=(1,2)`）但預設關閉：一階會與合法的等速運動對抗，需要 $L_{\text{cam}}$ 當煞車，尚未在醫學線調校。

> **機制與目標的區分**：temporal attention 是**機制**（讓網路能跨幀看），camera-smooth 是**目標**（要求輸出時序連貫）。兩者可獨立開關，但目前的實驗把它們綁在同一個 arm，**因子尚未拆解**（§10）。

---

## 5. 損失函數

$$
\boxed{\;
L \;=\; \lambda_{\text{cam}} L_{\text{cam}} \;+\; \lambda_{d} L_{\text{depth}} \;+\; \lambda_{g} L_{\text{gate}} \;+\; \lambda_{s} L_{\text{camera\_smooth}} \;}
$$

loss dispatcher 只在該項的 config block 存在時才把它加入總和，因此損失組合純由 yaml 開關控制，不需改程式。

### 5.1 $L_{\text{cam}}$

$$
L_{\text{cam}} = \frac{1}{n}\sum_{i=1}^{n}\gamma^{\,n-i}\Big[
w_T \big\Vert \hat{\mathbf{t}}^{(i)} - \mathbf{t}\big\Vert_1
+ w_R \big\Vert \hat{\mathbf{q}}^{(i)} - \mathbf{q}\big\Vert_1
+ w_f \big\Vert \widehat{\mathrm{fov}}^{(i)} - \mathrm{fov}\big\Vert_1 \Big]
$$

$n = 4$（refine stage 數），$\gamma = 0.6$，$w_T = w_R = 1.0$，$w_f = 0.5$。

- 平移項逐元素 clamp 至 $\le 100$，防止離群值主導梯度。
- 只在 **valid frame**（該幀有效深度點數 $>100$）上計算。
- 採 L1 而非 smooth-L1／L2：實測較穩定。
- 此項直接比對相機參數，**不經像素**，因此動態區的像素不會污染相機的 loss。

### 5.2 各設定的實際組合

| 設定 | $L$ |
|---|---|
| 通用 `inst_g` | $L_{\text{cam}} + L_{\text{gate}}$ |
| 通用 `inst_gts` | $L_{\text{cam}} + L_{\text{gate}} + \lambda_s L_{\text{camera\_smooth}}$ |
| **醫學 `*_vanilla` / `*_b16` / `*_b16_gg`** | $5.0\cdot L_{\text{cam}}$ |
| **醫學 `*_smooth_temporal` / `c3vd_cam_gts`** | $5.0\cdot L_{\text{cam}} + 3.0\cdot L_{\text{camera\_smooth}}$ |

醫學線沒有 $L_{\text{depth}}$（depth head 全程凍結，保留既有的深度能力、避免在小資料集上過擬合），也沒有 $L_{\text{gate}}$（無動態標註可餵）。

---

## 6. 訓練設定

### 6.1 凍結策略

模型從預訓練權重 warm-start，只解凍必要的部分。

**永遠凍結**

| 模組 | 理由 |
|---|---|
| `patch_embed`（DINOv2） | 最強的預訓練特徵，也是深度品質的來源 |
| `frame_blocks` | 相機與運動的耦合發生在 global attention，不在幀內空間 attention |
| `global_blocks[0..7]` | 門控之前的前段，保留通用幾何特徵 |
| `depth_head` | 本線的 scope 是純相機估計，深度不動 |

**可解凍（依 arm 而異）**：`global_blocks[8..23]`、`camera_head`、`camera_token`／`register_token`、`gate_predictor`、`temporal_blocks`。

### 6.2 SCARED 的 arm 設計

所有 arm 都建成 `enable_gate=True + enable_temporal=True`（即使該 arm 把它們凍住），使 checkpoint 的 key 集合一致，後續 arm 能直接接續載入而不會 missing key。

| arm | warm-start | temporal | gate | $L_{\text{smooth}}$ | 取樣張數 |
|---|---|---|---|---|---|
| `scared_cam_vanilla` | 預訓練權重 | 不建立 | 不建立 | ✗ | 4–12 |
| `scared_cam_b16` | `inst_g` | 凍結（$\gamma=0$ 恆等） | 凍結 | ✗ | 4–16 |
| `scared_cam_b16_gg` | `inst_g` | 凍結 | **可訓練** | ✗ | 4–16 |
| `..._smooth_temporal` | 承上 | **可訓練** | 可訓練 | **✓ $\lambda_s=3$** | 4–12 |
| `..._wide` | 承上 | 可訓練 | 可訓練 | ✓ | 4–12，`nearby_expand_range=120` |

C3VD 線同理：`c3vd_cam_vanilla`（不含兩個模組）對 `c3vd_cam_gts`（兩個模組全開）。

> ⚠️ **已知混淆**：`vanilla`(12 張) 與 `b16`(16 張) 的輸入張數不同，兩者之差不是純架構效果。`smooth_temporal` 同時開了 temporal 解凍與 $L_{\text{smooth}}$，無法歸因是哪一半在起作用。`wide` 的唯一 delta 是取樣窗（$\pm24 \to \pm120$），與架構正交。

### 6.3 超參數（SCARED 線）

| 項目 | 值 |
|---|---|
| optimizer | AdamW，base lr $3\times10^{-5}$ |
| LR schedule | Composite：前 5% linear warmup $10^{-8}\to 3\times10^{-5}$，後 95% cosine 至 $10^{-8}$ |
| `max_epochs` | 10 —— 它驅動 `where = epoch/max_epochs`、即整條 cosine 曲線，**同組比較的每個 arm 必須共用同一個值** |
| `limit_train_batches` | 1500 |
| `len_train` | 6000 samples／epoch |
| gradient clip | max_norm 1.0（L2），分四組：global+camera / camera_token+register_token / gate_predictor / temporal_blocks |
| 精度與記憶體 | bf16 autocast、gradient checkpointing、`accum_steps=1` |
| 硬體 | 單張 RTX 5090（32 GB） |
| `depth_max` | 655.35 mm（SCARED）／100 mm（C3VD 上游即在此截斷） |

> **顯存的真正決定因素是 `img_nums` 的上界，不是 `max_img_per_gpu`**：dataloader 算的是 `batch_size = max(1, ⌊max_img_per_gpu / random_image_num⌋)`。當抽到的張數大於 cap，floor 變 0、`max()` 抬成 1，該 batch 就帶著超過 cap 的影像數。最壞情況永遠是 $\max(\texttt{img\_nums})$。

---

## 7. 資料集

| 資料集 | 性質 | GT | 切分 | 用途 |
|---|---|---|---|---|
| **SCARED** | 腹腔鏡（da Vinci），結構光重建 | 稀疏 depth（mm）+ camera pose（w2c） | train = keyframe1/2、val = keyframe3、**test = keyframe4** | 主要訓練與評測（17,824 幀 / 48 GB） |
| **C3VD** | 大腸鏡（真實 Olympus 鏡頭 + 矽膠假體），對 CT mesh 註冊 | **稠密** depth（mm，上游截斷 100 mm）+ pose | 按整段序列、以貼圖紋理切分；test 另含唯一的降結腸序列（未見解剖結構） | 第二個有 GT 的醫學域 |
| 私人病灶（lesion） | 內視鏡，3 段 | **無標定、無 GT** | — | warp PSNR 自洽性 + 點雲視覺對照 |
| 私人胃（gastric） | 內視鏡，單目 | **無標定、無 GT** | — | 僅點雲視覺對照 |
| Sintel / PointOdyssey / TartanAir / Waymo / Spring | 通用動態場景 | depth + pose (+ 動態標註) | — | 通用線訓練與消融 |

**SCARED 的 split 細節**

- `test/`（550 幀 / 7 序列）是官方**稀疏抽樣**（間隔 1–296 幀）→ **不能算 snippet ATE**（幀不連續）。
- `pose_seq/dataset{5,3}/keyframe4`（411 / 834 幀）是同一段影片**逐幀重抽**（連續）→ 相機評測用這個。
- ⚠️ `test/dataset3/keyframe4`（79 幀）與 `pose_seq/dataset3/keyframe4`（834 幀）**同名不同物**，引用必須帶 split。
- 兩者皆屬 keyframe4 = test，**沒進訓練**。
- SCARED 有空深度幀，會靜默吃掉相機 loss（valid frame 判準是有效點數 $>100$），取樣時要避開當 anchor。

**協定對齊**（逐幀驗證過）：depth 用 AF-SfMLearner 的 `test_files.txt`，**幀數與幀號完全相同**；pose 用 `test_files_sequence1/2.txt` 的同兩段序列。

---

## 8. 評測協定與公式

### 8.1 Pose

**(a) 全序列 ATE（evo；Sintel / C3VD 用）**

把預測的 w2c 外參轉成相機中心軌跡 $\{\hat{\mathbf{p}}_i\}$，與 GT $\{\mathbf{p}_i\}$ 做 **Sim(3) 對齊**（`align=True, correct_scale=True`）：

$$
(s^\star, R^\star, \mathbf{t}^\star) = \arg\min_{s, R, \mathbf{t}} \sum_i \big\Vert s R \hat{\mathbf{p}}_i + \mathbf{t} - \mathbf{p}_i \big\Vert^2
$$

$$
\mathrm{ATE} = \sqrt{\frac{1}{N}\sum_i \big\Vert s^\star R^\star \hat{\mathbf{p}}_i + \mathbf{t}^\star - \mathbf{p}_i\big\Vert^2}
$$

同時報 RPE（`delta=1 frame, all_pairs=True`）：$\mathrm{RPE}_t$（平移）與 $\mathrm{RPE}_r$（旋轉角，度），以及 $\mathrm{ate\_rel} = \mathrm{ATE} / \Vert\text{軌跡 bbox 對角線}\Vert$，讓不同尺度的序列可比。

**(b) Snippet ATE（AF-SfMLearner 協定；SCARED 用）**

對每個長度 $n=5$ 的滑動窗（起點 $i$），把窗內位姿表示到**窗首座標系**：

$$
\mathbf{x}_j = \big(C_i^{-1} C_j\big)_{[:3,3]},\qquad j = i,\dots,i+n-1
$$

（$C = E^{-1}$ 為 c2w。此式等價於 AF 的 `dump_xyz` 對相對變換逐次相乘，只是繞開了其 pose network 的符號慣例。）

錨定到原點後擬合單一 scale：

$$
\hat{\mathbf{x}}'_j = \hat{\mathbf{x}}_j + (\mathbf{x}_i - \hat{\mathbf{x}}_i),
\qquad
s = \frac{\sum_j \langle \mathbf{x}_j,\ \hat{\mathbf{x}}'_j\rangle}{\sum_j \Vert\hat{\mathbf{x}}'_j\Vert^2}
$$

$$
\boxed{\;\mathrm{ATE}_{\text{snippet}} = \frac{1}{n}\sqrt{\sum_{j} \big\Vert s\,\hat{\mathbf{x}}'_j - \mathbf{x}_j \big\Vert^2}\;}
$$

⚠️ **分母是 $n$ 不是 $\sqrt{n}$** —— 這不是筆誤，是 AF 原始碼的定義；為了讓數字可與其表格對比必須照抄。窗的枚舉也照抄：對 $S-1$ 個相對位姿跑 `range(0, n_rel-1)`，尾端的窗被**截斷**（4、3、2 個相對位姿）而非丟棄，共 $S-2$ 個窗。

旋轉誤差同 AF 的 `compute_re`：

$$
\mathrm{RE} = \frac{1}{n}\sum_j \arctan2\big(\Vert\mathbf{s}(R_j)\Vert,\ \mathrm{tr}(R_j) - 1\big),
\quad R_j = R^{\text{gt}}_{i\to j}\big(R^{\text{pred}}_{i\to j}\big)^{-1}
$$

**(c) 長序列：chunk 與 Sim3 拼接**

單次前向的幀數上限約 80（受顯存限制）。長序列有兩種處理：

- **`chunk_size = 0`（整段一次前向）** —— 任何跨幀指標的唯一正確作法。獨立 chunk 拼接會讓接縫主宰 ATE，且數字不再隨模型變化。
- **重疊 chunk + 逐接縫 Sim3** —— 相鄰 chunk 有 `overlap` 幀重疊，用 **Umeyama** 在重疊段的相機中心上解 Sim(3)，把後一個 chunk 對齊到前一個的座標系：

$$
(s, R, \mathbf{t}) = \mathrm{Umeyama}\big(\{\mathbf{c}^{\text{new}}_k\}_{k \in \text{overlap}},\ \{\mathbf{c}^{\text{out}}_k\}_{k\in\text{overlap}}\big)
$$

拼接**只用於 pose**，depth 不回傳——每個 chunk 有自己的尺度，按接縫因子縮放深度會靜默混合不同尺度。

⚠️ **SCARED 不報全序列 evo ATE**：實測拼接誤差讓 ATE 擺動 $-1\% \sim +32\%$，訊號被拼接本身淹沒。因此 EndoSfM3D 那一系（全序列 evo ATE）的 pose 欄我們**填不了**——這是**選擇不估，不是估錯**。SCARED 只報 snippet ATE，只與 AF-SfMLearner 系比較。

⚠️ **chunk 長度是預測本身的一部分**：`chunk 64` 的每個窗有 64 幀 context，`chunk 5` 只有 5 幀（更貼近單目設定）。兩者算指標的方式相同但預測不同，**不可互相取代**，必須分開記錄。

### 8.2 Depth

預測深度只有相對尺度，先對齊再算指標。

**(a) `afsfm`（AF-SfMLearner 協定，論文可比欄位用）** —— **per-frame** 中位數縮放，1 個自由度：

$$
s_t = \frac{\mathrm{median}\big(D^{\text{gt}}_t[\mathcal{M}_t]\big)}{\mathrm{median}\big(\hat{D}_t[\mathcal{M}_t]\big)},\qquad
\hat{D}^{\text{aligned}}_t = \mathrm{clip}\big(s_t \hat{D}_t,\ \ge 10^{-2}\big)
$$

有效範圍 $\mathcal{M}_t = \{10^{-2} < D^{\text{gt}} \le 150\ \text{mm}\}$。中位數必須用 `np.median` 語意（偶數個元素取中間兩者的**平均**）；`torch.median` 取較小者，數字會與參考實作對不上。

**(b) `monst3r`（lad2）** —— 對整段序列的所有有效像素共同擬合 scale + shift（2 DOF），以 Adam 最小化

$$
(s^\star, b^\star) = \arg\min_{s,b} \sum_{\mathbf{u}} \big\vert s\,\hat{D}(\mathbf{u}) + b - D^{\text{gt}}(\mathbf{u})\big\vert
$$

**指標**（對齊後）：

$$
\mathrm{AbsRel} = \frac{1}{|\mathcal{M}|}\sum \frac{|\hat{D} - D|}{D},\qquad
\mathrm{SqRel} = \frac{1}{|\mathcal{M}|}\sum \frac{(\hat{D} - D)^2}{D}
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{M}|}\sum (\hat{D}-D)^2},\qquad
\mathrm{RMSE}_{\log} = \sqrt{\frac{1}{|\mathcal{M}|}\sum (\ln\hat{D}-\ln D)^2}
$$

$$
\delta_k = \frac{1}{|\mathcal{M}|}\Big|\Big\{\ \max\big(\tfrac{\hat D}{D}, \tfrac{D}{\hat D}\big) < 1.25^{\,k}\ \Big\}\Big|,\quad k=1,2,3
$$

⚠️ **輸入張數是隱藏變因**。發表的醫學表格幾乎都是單目（一次一張）。只有 `--single_view` 那一欄可與其對比；多張輸入（`chunk 64 / overlap 16`）是本模型的原生設定。零樣本權重光靠 multi-view 就把 AbsRel 從 0.0758 拉到 0.0488 —— **排名有很大一部分是輸入設定決定的**，每張表都必須標註。

### 8.3 無 GT 資料的自洽性指標：Warp PSNR

私人資料沒有標定、沒有 GT，因此沒有精度指標。改量 **depth 與 pose 是否合得起來**：用預測的 $\hat{D}_{\text{dst}}$ 與相對位姿把來源影像重投影合成目的視角，比對真實影像。

$$
T_{\text{dst}\to\text{src}} = E_{\text{src}} E_{\text{dst}}^{-1},\qquad
\hat{I}_{\text{dst}}(\mathbf{u}) = I_{\text{src}}\Big(\pi\big(T_{\text{dst}\to\text{src}}\,\pi^{-1}(\mathbf{u}, \hat D_{\text{dst}}(\mathbf{u}))\big)\Big)
$$

$$
\mathrm{PSNR} = 10\log_{10}\frac{1}{\mathrm{MSE}\big[\hat{I}_{\text{dst}},\ I_{\text{dst}}\big]_{\mathcal{V}}},\qquad
\mathcal{V} = \{z>0\ \wedge\ \text{投影落在界內}\}
$$

另報 $\mathrm{PSNR}_{\text{no spec}}$：額外排除飽和高光像素（任一通道 $\ge 250/255$）。內視鏡光源黏在鏡頭上，高光跟著相機移動，幾何 warp 原理上重現不了它。

⚠️ **這不是精度指標**，只是自洽性；且它是相對本模型自身的 depth scale 定義的，**與其他模型不可比**。

### 8.4 Checkpoint 選擇

- **channel A（windowed val）**：每 epoch 隨機抽 4–16 幀算 loss。⚠️ seed 隨 epoch 變 → 變動 = 模型 + 資料，分不開。
- **channel B（pose_eval）**：每 epoch 在 val 上跑固定的全序列 ATE（SCARED：6 個 keyframe × 50 幀），**完全決定性**，`best_ate.pt` 由此選出。
- **test**：`benchmark/eval_*.py`，與 channel B 不同 split。

---

## 9. 結果

> 單位：SCARED / C3VD 的 ATE 與 depth 誤差皆為 **mm**。空白 = 還沒跑，不是 0。
> `VGGT-1B 零樣本` 一列是**未經任何內視鏡微調**的預訓練權重，只作為 sanity reference，不是 baseline。

### 9.1 SCARED — Depth（test split，afsfm 協定，AbsRel）

| ckpt | 單張輸入 | 多張輸入 |
|---|---|---|
| VGGT-1B 零樣本 | 0.0758 | 0.0488 |
| `vanilla`（微調 baseline） | 0.0561 | 0.0454 |
| `b16` | 0.0563 | 0.0459 |
| `b16_gg` | 0.0550 | 0.0449 |
| `smooth_temporal` | **0.0537** | 0.0434 |
| `wide` | 0.0550 | **0.0419** |

只有**單張**那欄可與論文的單目表對比。depth head 全程凍結，此處的改善來自 aggregator 特徵本身的變化。

### 9.2 SCARED — Pose（pose_seq split，snippet ATE）

**chunk 64 / overlap 16**

| ckpt | ds3 | ds5 | **mean** |
|---|---|---|---|
| VGGT-1B 零樣本 | 0.1159 | 0.1258 | 0.1209 |
| `vanilla` | 0.0626 | 0.0902 | 0.0764 |
| `b16` | 0.0586 | 0.0826 | 0.0706 |
| `b16_gg` | 0.0565 | 0.0832 | 0.0699 |
| `smooth_temporal` | 0.0478 | 0.0716 | **0.0597** |
| `wide` | 0.0585 | 0.0825 | 0.0705 |

**chunk 5 / overlap 4**（context 只有 5 幀，最貼近單目設定）

| ckpt | ds3 | ds5 | **mean** |
|---|---|---|---|
| VGGT-1B 零樣本 | 0.1294 | 0.1549 | 0.1421 |
| `vanilla` | 0.0712 | 0.1017 | 0.0865 |
| `b16` | 0.0675 | 0.0973 | 0.0824 |
| `b16_gg` | 0.0663 | 0.0956 | 0.0810 |
| `smooth_temporal` | 0.0567 | 0.0844 | **0.0705** |
| `wide` | 0.0723 | 0.1007 | 0.0865 |

兩件可讀的事：(i) context 從 5 幀放大到 64 幀，**每個 arm 都變好**（$-12\%\sim-19\%$）；(ii) 排名大致一致（`smooth_temporal` 兩邊都最好），但 `wide` 在 chunk 5 掉到與 `vanilla` 並列最差。

### 9.3 C3VD（test split，5 序列 × 50 幀）

| ckpt | ATE | ate_rel | rpe_t | rpe_r | AbsRel | $\delta_1$ |
|---|---|---|---|---|---|---|
| VGGT-1B 零樣本 | 2.6343 | 0.0985 | 1.1681 | 0.7625 | 0.2252 | 0.6186 |
| `c3vd_cam_vanilla` best_ate | 0.5978 | 0.0218 | 0.2356 | 0.2173 | 0.0874 | 0.9584 |
| `c3vd_cam_gts` best_ate | **0.4329** | 0.0176 | 0.1865 | 0.1859 | **0.0736** | 0.9649 |

⚠️ 兩個 run 的 `best_ate.pt` **都是 epoch 0**，之後再沒更新過。test 也同向（ep0 0.4329 < ep5 0.4642）。**C3VD 上訓練越久越差**，這件事本身需要解釋，不能只報最好的那格。

### 9.4 私人資料集（無 GT）

- **lesion warp PSNR**：所有 arm 之間差 $\le 0.56$ dB，排序幾乎不變 → **在這個指標上分辨不出模型差異**。零樣本權重在其中兩段上仍是最高。
- **gastric**：只有點雲；`ply_points` 由幀數 × 像素數決定，**不帶模型資訊**，只能做視覺對照。

---

## 10. 限制與缺口

| # | 限制 |
|---|---|
| 1 | **「最好的版本」沒有單一答案**：選 ckpt 用的 val ATE 選出 `wide`，test 上 `smooth_temporal` 較好。報告要先決定用哪個指標敘事。 |
| 2 | **因子未拆解**：`smooth_temporal` 同時開了 temporal 解凍與 $L_{\text{camera\_smooth}}$，無法歸因；`wide` 與它的差是**取樣**而非架構。 |
| 3 | **輸入張數混淆**：`vanilla`(12) 與 `b16`(16) 張數不同，兩者之差不是純架構效果。 |
| 4 | **模組 A 在醫學線的邊際效果很小**：`b16` → `b16_gg`（唯一 delta 是門控可訓練）只有 0.0706 → 0.0699（chunk 64）與 0.0824 → 0.0810（chunk 5）。方向一致但幅度小。 |
| 5 | **無標註門控的語意未驗證**：靠相機 loss 訓練，學到的是「哪些 patch 對相機估計有害」；是否收斂到有意義的東西未經驗證。 |
| 6 | **SCARED 沒有全序列 evo ATE**（拼接誤差 $-1\%\sim+32\%$），EndoSfM3D 系的 pose 欄填不了。 |
| 7 | **baseline 政策**：對比對象必須是**微調過的**模型。MonST3R 至今未在內視鏡微調，故目前不算有效 baseline。 |
| 8 | **成本軸完全沒有資料**：本方法是單次前饋（秒級），MonST3R 是 test-time optimization（分鐘級／序列）。wall-clock / VRAM / 是否需 per-scene 最佳化是核心宣稱，**待補**。 |
| 9 | **兩個 run 沒跑完**：`scared_cam_b16_gg` 在 ep8 被中斷，其 `best_ate.pt` 是 8-epoch 產物；C3VD 在 ep6 之後暫停。 |
| 10 | 通用線（Sintel）**沒有 held-out**，且單次比較的雜訊門檻為 16.9%（2σ）。 |
