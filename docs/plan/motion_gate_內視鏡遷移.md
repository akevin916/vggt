# Motion gate 遷移到內視鏡：PAGE-4D 對照與計畫

> 2026-10-12。依據：`/mnt/project-files/page4D.pdf`（PAGE-4D, ICLR 2026, arXiv 2510.17568v8）、
> repo `akevin916/vggt` 分支 `illu-vggt`（commit `63ebf98`）的 `vggt/models/aggregator.py`、
> `docs/status.md`、`docs/experiments.md`、`docs/results/medical.md`、`docs/內視鏡整理.md`、`docs/計劃書_2026-10.md`。
> 本文只有閱讀與比對，沒有跑任何程式。標「推論」的是我的判斷，不是量到的。

## 一句話

你期中的 gate 和 PAGE-4D 的 mask attention 是同一個機制；要遷移的不是架構，而是「在內視鏡裡，什麼區域該被 camera token 忽略、用什麼當標籤」。

---

## 1. PAGE-4D 的 motion gate 在做什麼

用一個例子說：一段 8 幀影片裡有個人在走。camera token 在 global attention 裡會去看所有幀的所有 patch；
走動的人那些 patch 的位移不符合「只有相機在動」的幾何（極線約束），會把 pose 拉偏。
PAGE-4D 讓模型先猜「哪些 patch 在動」，再只對 camera / register token 把這些 patch 的 attention 壓低；
depth / point 用的 patch token 不壓，因為動態區的位移反而是重建它的線索。

機制拆開：

- **三段式 aggregator**：前 N1 層照舊 → 中間 N2 層（約 10 層）換成「帶 mask 的 global attention」→ 後 N3 層照舊。
- **mask 預測**：取第 N1 層輸出的 patch token，linear 降維 + depthwise conv → 每個 patch 一個 logit。
  **不用任何 mask 標籤**，完全由 camera / depth / point 的 loss 反傳學出來（論文 §3.2.1、Fig. 6）。
- **mask 怎麼加**：`softmax(QKᵀ/√d + M)`，M 只作用在 camera / register 的 query row，patch query 不加。
  實作是 rank-1 加法：query 一個向量 r、key 一個向量 c，`q' = [q·√(d'/d), r·√d']`、`k' = [k, c]`，
  多一維就等於加上 `r_i·c_j`，O(N) 記憶體、可走 fused SDPA。
- **只微調中間 10 層 global attention**（約 30% 參數），其他凍結。
- **訓練才用 mask 的版本**：推論時拿掉 mask，回到原 VGGT 架構，成績與一直開著相近（附錄 §8.1）。
  也就是 mask 可以只當「訓練期的正則化」。

論文自己的消融（Odyssey、Sintel、Bonn）：

| 設定 | 結果 |
|---|---|
| 只微調中間層，不加 mask | Sintel video depth AbsRel 0.409 |
| 中間層 + mask attention | 0.357（−12.7%） |
| DD-MSK（讓動態 patch 彼此看不到，所有 query 都套） | pose 變好，但靜態區 depth 0.085 → 0.566 |
| 輸入端直接把動態區塗掉 | pose 沒變好，depth 變差 |

論文的結論是「只對 pose 遮、不對 geometry 遮」，這和你期中 gate 的設計原則一樣。

---

## 2. 與你現有 `gate` 的差異

你現有的 gate（`aggregator.py` 的 `GatePredictor` 與 `_gated_global_block_forward`）已經做到：
mid-aggregator 預測 per-patch logit `g`、只對 camera / register query row 加 bias、patch↔patch 走 flash 不加 bias。
差別在下面五點：

| 項目 | PAGE-4D | 你的 gate |
|---|---|---|
| 監督來源 | 無標籤，由下游 loss 自學 | BCE 對 instance×scene-flow 標籤；`gate_pose_grad=True` 才讓 pose loss 也教它 |
| bias 函數 | 直接把 logit 加進去，沒有死區 | 預設 `min(0, softplus(0) − softplus(g))`，σ(g) < 0.5 時 bias 恆為 0 |
| 作用的層 | 中間約 10 層，之後的層不遮 | `gate_block_iter=7` 之後的 global block 8–23 全部（16 層） |
| 預測頭 | linear + depthwise conv（看鄰近 patch） | LayerNorm + 2 層 MLP（每個 patch 各自判斷） |
| 推論 | 有「訓練才用 mask」的版本 | 推論一定開著 |

這些差異對應到 `status.md` 已經量到的問題：

- **死區**：Sintel 上 soft gate 只換到 ATE −0.15%，同一份預測二值化後 −4.76%（oracle −5.85%）。
  PAGE-4D 的「直接加 logit」正好沒有這個死區。repo 已有 `gate_bias_scale`（`−s·softplus(a·(g−τ))`）與 `gate_bias_zero_ref` 可以近似它。
- **SCARED 沒有標籤**：gate 在 SCARED 上 σ(g) 平均 0.005–0.010、沒有 patch 越過 0.5（`status.md` B.3）。
  PAGE-4D 的自學監督就是為了這種情況；你的 `scared_cam_b16_gg`（`gate_pose_grad`）是唯一試過的一次，
  val ATE 0.9464 vs `b16` 0.9814，但 ep8 被中斷，test c5 0.0810 也沒贏 `smooth_temporal` 0.0705。

---

## 3. 內視鏡和自然場景哪裡不一樣

PAGE-4D 的推導（式 4）給了一個與資料域無關的定義：**一個像素該被 pose 忽略，是因為它的跨幀位移離開了相機運動所預測的極線**。
自然場景裡造成這件事的主要是移動的人和車；內視鏡裡來源不同：

| 來源 | 會違反極線嗎 | 在 SCARED 上有多少 | 現有證據 |
|---|---|---|---|
| 組織變形（呼吸、心跳、被器械推） | 會 | 推論：很少。SCARED 是離體豬隻，沒有呼吸心跳 | 無 |
| 器械 | 會（剛體、自己動） | 推論：部分 keyframe 有，比例未量 | 無 |
| 反光 | 會（亮點隨光源和視角移動，不黏在表面上） | 每幀 8–17% patch | oracle 抑制反光 patch，pose 只動 ±1%（`內視鏡整理.md`） |
| 光照變化（光源跟著鏡頭） | 不直接違反幾何，影響的是外觀匹配 | 普遍 | 光照 token 是負結果（`status.md` E.2） |
| 無紋理、模糊、煙霧 | 不違反，但匹配不可靠 | 未量 | 無 |

兩個推論：

1. **在 SCARED 上，gate 能贏的空間可能本來就很小**。自然場景的 oracle 頭空是 −5.85%；SCARED 沒有大面積移動物體，
   反光 oracle 又只有 ±1%。若頭空 < 2%（你計劃書的擺動參考量），在 SCARED 上做 gate 就量不出效果。
2. **內視鏡的 gate 應該定義成「這個 patch 對 pose 不可靠」，而不是「這個 patch 在動」**。
   它的標籤可以直接照 PAGE-4D 式 4 來算，不需要 instance mask。

---

## 4. 遷移計畫

### 步驟 0：先量頭空（不訓練）

- **要回答的問題**：在 `tfs` 上，若 camera token 完美忽略「違反極線的 patch」，SCARED test 的 pose 能好多少？
- **標籤怎麼來**：每對相鄰幀用 GT 相對 pose 算 essential matrix E，對 RAFT flow（`flow_geom` 已在用的同一套）
  算每個像素的 Sampson 距離（即式 4 的 δ），pool 到 patch grid，超過門檻的 patch 標 1。
  這個標籤一次涵蓋變形、器械、反光和 flow 本身算錯的地方。
- **怎麼量**：用現有的 `gate_logits_override` 把這個標籤送進 `tfs`（override 不需要 `gate_predictor`），
  跑 pose c64 與 c5，對照 override 全 0 的同一次 forward。另外報「被標 1 的 patch 比例」。
- **怎麼判讀**（事前定）：頭空 ≥ 4% → 做步驟 1；< 2% → SCARED 上不做 gate，改看步驟 3；中間 → 回報討論。
- **成本**：GPU 數分鐘到數十分鐘；標籤 script 約 50–80 行，寫在 scratchpad。
- 實作上要注意：override 目前從 global block 0 就作用（`aggregator.py` 第 743 行附近沒檢查 `gate_block_iter`），
  要另外加 oracle 只在 8–23 或 8–17 層作用的版本，否則量到的不是方法真正能做到的上限。

### 步驟 1：在 `tfs` 上照 PAGE-4D 的方式訓 gate

建議的 config 差異（每一項都對應 PAGE-4D 的一個設計）：

- bias 改成沒有死區的形式：`gate_bias_scale` 參數化版（或 `gate_bias_zero_ref`）。
- 只在中間層作用：global block 8–17（10 層），18–23 不遮。需要加一個 `gate_block_end` 參數，約 5 行。
- 監督：
  - A 版：只用 `gate_pose_grad`，不給標籤（照 PAGE-4D）。
  - B 版：步驟 0 的極線殘差標籤做 BCE，加上 `gate_pose_grad`。
- 其餘照 `tfs`（`docs/計劃書_2026-10.md` §0.1），比 pose c64、depth 多張，參考 c5、depth 單張。
- 另外評一次「推論時關掉 gate」，對應 PAGE-4D 的「訓練才用 mask」版本；如果一樣好，論文可以宣稱零推論成本。

### 步驟 2：預測頭（只在步驟 1 有效時做）

把 MLP 換成 linear + depthwise conv，讓 gate 看得到鄰近 patch。反光、器械邊緣都是區塊狀的，推論上有幫助，但不是第一優先。

### 步驟 3：換一個真的有動態的資料（步驟 0 頭空 < 2% 時）

SCARED 的動態太少，可能不適合當 gate 的主場。候選是 in vivo、有呼吸與器械、又有相機 pose 的資料，
例如 StereoMIS（in vivo 豬隻、達文西運動學給 pose；**資料集細節我沒有查證**）。
這會動到論文的主實驗設定，要你決定。

---

## 5. 與 PAGE-4D 的區隔（論文怎麼寫）

PAGE-4D 已經發表了「只對 pose 遮動態區」的想法，所以單純搬過來不能當貢獻。可以主張的差異：

- **內視鏡的「不可靠」不等於「在動」**：反光、器械、變形來源不同，PAGE-4D 只處理移動物體。
- **用式 4 的極線殘差當監督**：PAGE-4D 只推導了式 4、沒有拿它當標籤；這在沒有 instance mask 的醫學域是自然的做法。
- **你已量到的死區問題**：soft bias 在欠自信時失效，PAGE-4D 沒有討論 calibration。

（推論，未經確認。）

---

## 附錄：步驟 0 的驗證計畫（2026-10-13，待 Kevin 同意）

依據：`pipeline/benchmark/eval_scared.py`（已有 `gate_logits_override` 管線與 `--gate_mode off|predicted`）、
`pipeline/training/flow_mask.py` 的 `pair_flows`（RAFT + 遮擋 mask）、`pipeline/data/datasets/scared.py`（`extrinsics.txt` 是 world→cam 3×4）、
`pipeline/training/loss.py` 的 `oracle_gate_logits_from_mask`（k = 30）。

- **要回答的問題**：在 `tfs` 上，若 camera token 完全不看「違反極線」的 patch，SCARED test 的 pose c64 / c5 能比不遮好多少？
- **怎麼量**：
  1. 標籤（scratchpad 一次性 script，約 80 行）：test/pose 每一幀 i 和 i±k 配對（k 從 1 往上加，直到 GT 平移 ≥ 中位相鄰平移的 3 倍，避開純旋轉），
     GT extrinsics 算 F（R = R₂R₁ᵀ、t = t₂ − R t₁），RAFT flow 得 x₂，逐像素算 δ（只取可見像素），多個配對取最大，
     平均到 518×420 的 37×30 patch 格。門檻 τ = 全部 patch δ 的中位數 + 3×MAD，**看 pose 之前先定**。
  2. 遮罩送進 `tfs`：`eval_scared.py` 加 `--gate_mode oracle_file`（讀步驟 1 的 npy，標 1 → +30、標 0 → −30），
     aggregator 加一個「override 從第幾個 global block 開始」的參數（預設 0，與現狀 byte-identical），設 8。
     跑 pose c64 與 c5，和同一 ckpt 的 `off` 比。
- **sanity check（批次第一步，不過就停）**：全部 patch 的 δ 中位數必須 < 2 px。超過代表 pose 慣例或配對方向錯了，不往下跑。
- **怎麼判讀（事前定）**：
  - c64 與 c5 都 ≤ −4%，且兩條序列同方向 → 做步驟 1（訓練）。
  - c64 與 c5 都 > −2% → SCARED 上不做 gate，改看步驟 3（換有動態的資料）。
  - 其他 → 回報討論。
  - 另外報「被標 1 的 patch 比例」，只當參考，不參與判讀。
- **成本**：GPU。標籤是 RAFT 跑 test/pose 全部幀，估計數分鐘到十幾分鐘；eval 的 c64 數分鐘、c5（830 個窗）約十幾到二十幾分鐘。都要在 75 上跑（資料與 GPU 在那裡）。
- **要改的 repo 程式**：`eval_scared.py`（新 mode，約 20 行）、`aggregator.py`（override 起始 block，約 5 行，預設不改行為）。標籤 script 留在 scratchpad。

### 步驟 0 修訂（2026-10-13，依 Kevin 的建議，待同意）

Kevin 的觀點：醫學影像上 camera token 能聚合的資訊可能比較少，只遮 camera→patch 效果有限；要試不同遮法，並同時看 depth；ckpt 除了 `tfs` 也測 `VGGT-1B`，因為 `tfs` 是在沒有遮罩的情況下訓練的，可能已經適應（依賴）了那些 patch。

| 遮法 | 誰被限制 | 對應 |
|---|---|---|
| off | 不遮 | 對照 |
| M1 camera-only | camera / register query 不看標 1 的 patch | PAGE-4D、現有 gate |
| M2 all-query | 所有 query 都不看標 1 的 patch | 最強；PAGE-4D 說會傷 depth，但內視鏡的「不可靠」可能也傷 depth |
| M3 cross-frame | 所有 query 都不看「其他幀」裡標 1 的 patch，同一幀內照看 | 不可靠的 patch 仍能幫自己那幀，但不當跨幀對應 |

- 網格：2 個 ckpt（`VGGT-1B`、`tfs`）× 4 種遮法，指標 pose c64、c5、depth 多張。
- 判讀：pose 照原本的 ≥4% / <2%；depth 多張 ≥2% 算有改善（計劃書的擺動參考量）。
- 實作：M1 走現有的 special-query 路徑；M2、M3 要對 patch query 也加 bias，沿用 `dual_stream_scope: all` 的 flex_attention 路徑（只推論、不需反傳），約 40–60 行，推論變慢約 2 倍。
- 解讀限制：`VGGT-1B` 沒在 SCARED 上微調，它的誤差主要來自域差，遮罩在它身上的效果不一定能轉移到訓練後的模型。
