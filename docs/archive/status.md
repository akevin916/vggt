# 研究進度：現在信什麼

> **建立 2026-08-31。這是全 repo 唯一的「當前狀態」來源。**
> 機制怎麼運作看 [method.md](method.md)；每個 run 的身分與下場看 [experiments.md](experiments.md)；
> 數字本身看三份數據表（[results/natural.md](results/natural.md) 自然場景 ／ [results/medical.md](results/medical.md) 醫學
> ／ [results/sota.md](results/sota.md) 對外對照）。**這四份都不重複下結論，結論只寫在這裡。**

## 一句話

Gate（正式版 v1）在自然場景把 Sintel ATE 從 0.1714 帶到 0.1343，但**增益幾乎全部來自
temporal + camera_smooth，gate 本身的 soft bias 目前近乎無作用**（predicted −0.15%，oracle −5.85%）；
SCARED 的成分消融（§E）已成形：最佳組合 pose −16.7%，光照 token 是負結果。
醫學域（SCARED）是另一條較晚開的線，pose 相對微調 baseline 有 17–24% 改善、depth 幾乎沒動；
照明穩健性（L_inf）機制已實作、第一次 run 失敗、**結論未定**。

---

## 五條線的狀態

| 線 | 域 | 狀態 | 最後一個可信數字 | 依據 |
|---|---|---|---|---|
| **A. Gate（正式版 v1）** | Sintel / PointOdyssey | 🟡 機制成立、兌現不足 | Sintel ATE **0.1343**（`inst_gts` ep30） | [results/natural.md](results/natural.md) 表 1 |
| **B. 醫學域適配** | SCARED / C3VD / 私人內視鏡 | 🟢 有正向結果，但排名未定 | snippet ATE **0.0597**（chunk 64）、單張 AbsRel **0.0537** | [results/medical.md](results/medical.md) §1 |
| **C. 照明穩健性（L_inf）** | SCARED | ⚪ 機制已實作，**尚未有結論** | — | [method.md](method.md) §5 |
| **D. 反光的影響** | C3VD | 🔴 負結果（C3VD 上反光不傷重建） | 點雲角度校正後 **2.07 vs 2.23 mm** | 本頁 §D.2 |
| **E. SCARED 成分消融** | SCARED | 🟢 主線，最佳組合成立；光照 token 為負結果 | snippet ATE **0.0709**（chunk 5，`tfs`） | [results/medical_part2.md](results/medical_part2.md)、本頁 §E |

「已收掉」的線見文末。

---

## A. Gate（正式版 v1，舊稱 v3）

### A.1 已經成立的

**跨域標籤問題已解決。** 用 instance×scene-flow 標籤取代 RAFT flow-residual，Sintel AUC
從 0.584 拉到 0.767（S0，只訓 gate_predictor）。這是這條線最紮實的證據，**但它證明的是
gate 品質，不是 pose**。

**gate 的排序能力可用。** `inst_gts` ep30 在 Sintel 14 seq / 50 幀上：
macro AUC **0.844**、micro AUC 0.823（`outputs/gate_quality/inst_gts/quality_f50.json`）。

**顯存成本 ≈ 0，推論成本量不到。** attention 拆兩條 path 的設計成立：Ours 與 VGGT-1B 在
五條 64 幀序列上都是 7.2 s，差異落在量測雜訊內（±0.1 s）。

### A.2 目前的主瓶頸（**這是這條線最重要的一件事**）

**soft bias 幾乎沒有兌現到 pose，但同一份預測二值化後就兌現了。**
`outputs/gate_bias_ablation/inst_gts/`，`inst_gts` ep30、Sintel 14 seq、`max_frames=50`、
`chunk_size=0`，2026-08-24：

| 模式 | ATE | Δ% vs off |
|---|---|---|
| `off`（無 gate 對照） | 0.13447 | — |
| **`predicted`（模型自己的 soft gate）** | **0.13426** | **−0.15%** |
| `predicted_hard@0.1`（同一份預測，σ(g)>0.1 二值化） | 0.12874 | **−4.26%** |
| `predicted_top0.2`（同一份預測，逐幀前 20%） | 0.12807 | **−4.76%** |
| `oracle`（GT mask，頭空上界） | 0.12660 | −5.85% |

同一份 `g`，只換「怎麼把它變成 bias」，就從 −0.15% 變成 −4.76%——**已經走掉 oracle 頭空的
81%**。這指向病因不是「gate 判斷錯」而是「gate 的數值落在 bias 函數的死區」：

- calibration 顯示 gate **系統性欠自信**（預測 0.145 的 bin，實際動態頻率 0.345）。
- 現行 bias 是 `min(0, softplus(0) − softplus(g))`，kink 在 σ(g)=0.5。動態 patch 的平均信心
  只到 ~0.36 → bias 恆為 0 → gate 結構性失效。

兩個已試的修法**都只回收一小部分**（同一 ckpt、同一協定）：

| 變體 | predicted Δ% | 檔案 |
|---|---|---|
| `gate_leaky=1.0`（拿掉 clamp） | −1.79% | `gate_bias_ablation/inst_gts/leaky1.0/` |
| `gate_bias_zero_ref=True`（拿掉 softplus(0) 參考點） | −1.10% | `gate_bias_ablation/inst_gts/zero_ref/` |

> ⚠️ **上面「病因是死區」是我的解讀，尚未經確認流程。** 已測的是數字本身，
> 「所以下一步該做什麼」還沒拍板。

### A.3 增益歸因仍未拆開

`inst_gts` 相對 `inst_g` 的 −12%（0.1533 → 0.1343）**混著 temporal 結構與 camera_smooth loss
兩個因子**，從 2026-07 掛到現在沒拆。考慮到 A.2 顯示 gate 本身只值 −0.15%，
**這個未拆的因子很可能就是這條線的全部增益**，拆它的優先序因此比原本更高。

### A.4 已知的量測限制

- 單次比較的 2σ 雜訊門檻 **16.9%**，且沒有 held-out 資料。
- `inst_gts` 在 ep30 後平台化在 0.134–0.136，**沒有「再訓久一點就追上 MonST3R（0.108）」的空間**。
- windowed `best.pt` 不可信（`inst_gts` 的 best.pt = ep17，full-seq 0.1651，比 ep30 差 23%）。
  這條線一律取 `epoch_30.pt`。

---

## B. 醫學域適配（SCARED / C3VD / 私人內視鏡）

### B.1 已經成立的

**pose 是這條線的賣點，方向正確。** 相對「微調過的 VGGT baseline」（不是 pretrained）：

| 指標 | baseline | 本期 | 改善 |
|---|---|---|---|
| depth 單張 AbsRel | 0.0561 | 0.0537 | −4.3% |
| pose snippet ATE（5 幀 context） | 0.0865 | 0.0705 | −18% |
| pose snippet ATE（64 幀 context） | 0.0764 | 0.0597 | **−22%** |

**depth 幾乎沒動（4%），pose 有 18–22%。**

**速度是結構性優勢。** 64 幀一條序列：Ours 7.2 s、MonST3R 450.7 s（**63 倍**）。
MonST3R 每條要跑 550 對 pairwise + 300 次全域對齊 + RAFT 光流，不是實作差異。

### B.2 未定的

**「哪個版本最好」沒有單一答案。** `wide` 在 channel B val ATE（0.8150）與 depth 多張輸入
（0.0419）領先，`smooth_temporal` 在 test pose（0.0597）與 depth 單張（0.0537）領先。
**報告要先決定用哪個指標敘事。**

**公平設定下 depth 還是輸。** 單張 0.0537 贏 AF-SfMLearner（0.059）但輸 EndoSfM3D（0.050）、
Endo-FASt3r（0.051）、上期 ESRT-Row（0.048）。全序列的 0.0434 **主要來自 multi-view 不是方法**
——pretrained VGGT-1B 完全沒微調、光靠 multi-view 就 0.0488。

**pose 追上但沒贏。** 64 幀下 Seq.2 的 0.0478 追平 AF-SfMLearner，但那是 64 幀 context；
公平的 5 幀設定（0.0844 / 0.0567）兩條都還輸給全部 published 方法。

**全序列 evo ATE 不報。** Sim3 拼接讓該指標擺動 −1%~+32% 且不可預測。**是不估，不是估錯。**
這不妨礙對外比較：EndoSfM3D 表上的 pose 欄是經 EndoDAC 沿用的 AF snippet 協定，可以和我們的 5 幀列
逐序列比；只有 EndoSfM3D 自己那一列的協定論文未載明。前提與規則見 [results/sota.md](results/sota.md) §2.1。

### B.3 兩個負結果

- **depth head 解凍失敗**：`..._depth` arm 的 test depth 反而比凍結的頭差
  （AbsRel 0.0434 → 0.0468），pose 退 18%（0.0597 → 0.0707）。病因已定位到 confidence loss
  的 `−alpha·log(c)` 項（見 [experiments.md](experiments.md)）。修正版 `_lowalpha` **尚未跑**。
- **gate 在內視鏡上幾乎不點火**：`outputs/gate_medical/*/stats.json`，`inst_g` 在 SCARED
  keyframe3/4 的 σ(g) 平均只有 0.005–0.010，`frac_gt_half = 0.0`——**沒有任何一個 patch 越過
  kink**。這與 A.2 的死區診斷同向，但在醫學域更極端（本來就沒有動態物體）。

### B.4 自監督（Colon版，`scared_selfsup_ca3`）

- **不用任何 pose / depth 標註，Colon版在 test snippet ATE 上贏過以 GT pose 訓練的 `vanilla`**（兩種 chunk 設定、兩條序列皆同向），但仍輸 `smooth_temporal`。
- **ColonAdapter 式的跨幀 3D 一致性、光流遮擋遮罩與 conf 加權修好了純自監督的 depth 退步，且幾乎不犧牲 pose**（單張 depth 由差於零樣本轉為優於零樣本）。

數字見 [results/medical.md](results/medical.md) §1.1、§1.2。

---

## C. 照明穩健性（L_inf）— **只有機制，沒有結論**

動機是一個實際觀察到的失效：胃部影片裡**單一幀** AWB 重鎖變綠，整段重建就跑掉。
機制（photometric corruption → 雙 forward → L_inf）寫在 [method.md](method.md) §5。

**現況：**
- 機制已實作並跑得動（`scared_point_inf`，10 epoch）。
- 第一次 run 的 val ATE 從 2.03 惡化到 3.24，遠差於同家族的 0.91。
- `diag/influence_scale_probe.py` 已把「正規化回饋」與「真的發散」兩個假說分開，
  並據此改了 loss（quantile filter + median 除數）與權重（30 → 10）。
- **修正後尚未重跑，weight=0 的 augmentation-only 對照組也還沒跑。**

> **在那兩個 run 出來之前，這條線不下任何結論**，也不拿去支撐其他實驗設計。

---

## D. 反光（specular）的影響 — **負結果，只在 C3VD 成立**

在 C3VD 上，反光**沒有**讓重建變差。點雲指標原本顯示反光區差 17%，該差距被
「反光只長在正對鏡頭的表面上、而點雲指標對正對表面特別嚴格」完全解釋掉；
校正後反光區反而好 7%。深度指標本來就顯示反光略優。

**D 與 C 是兩個不同的問題，不是前提關係。** D 測的是公開資料集（C3VD / SCARED）上的
反光，那裡的照明變化幅度小；C 針對的是真實臨床影片中大幅度的照明偏移（整幀 AWB 重鎖，
全幀色彩位移而資訊仍在）。兩者的失效方式、幅度、影響範圍都不同，D 的結果不能拿來
支持或否定 C。

**論文定位上，D 對公開資料集的針對性偏弱**：公開資料集本身就沒有臨床影片那種幅度的
照明失效可測，所以在這些資料上測出「反光不傷重建」，說明的是資料集的性質多於方法的價值。

### D.1 方法

**偵測器**（`pipeline/data/specular_mask.py`，全 repo 唯一）：像素平均亮度 > 0.82
**且** rg 色度距白點 < 0.07，再 7×7 dilate。判準是「亮且接近白」——反光帶的是光源的
顏色，不是黏膜的紅。門檻經 36 張 SCARED/C3VD overlay 人工確認（`outputs/specular_check/`）。

**為什麼場地是 C3VD 而不是 SCARED。** 理由是真值密度，不是真值成因。SCARED 的結構光
只打在每段影片的**一幀**（key-frame），其餘幀的深度由「幀間位姿 + 深度 warp」擴充，
所以缺洞主要來自 warp 的遮擋與出界，**與反光沒有已證實的因果關係**——實測的分層覆蓋率
在兩個 split 方向相反（val：clean 0.4051 / specular 0.3391；test：clean 0.3469 /
specular 0.3974），沒有系統性偏向。但覆蓋率本身只有 34–40%，六成像素缺席而缺席原因
未查清，分層統計就不可信；SCARED 上 val 與 test 的結論也互相矛盾。C3VD 的深度由已知
CT 網格算繪而來，覆蓋率 99.5–100%，沒有這個問題。

C3VD 的**逐像素深度**獨立於影像品質（來自網格），但用來算繪的**位姿**是以 GAN 深度
估計加邊緣對齊配準出來的，仍與影像品質有關（原文報告平移誤差 0.321 mm、旋轉 0.159°）。

**量法**（`pipeline/diag/washout_impact.py --arm observational`）：C3VD test 全部
18 個 64 幀窗，逐像素分 `clean` / `specular` 兩層，各自算深度 abs_rel 與融合點雲的
accuracy。點雲用真值位姿擺放、全序列單一尺度、兩朵雲限制在同一組有真值的像素。
ckpt 用 C3VD 微調的 `logs/c3vd_cam_vanilla/ckpts/best_ate.pt`。

### D.2 數據

| | clean | specular |
|---|---|---|
| 深度 abs_rel（全序列單一 scale） | 0.0977 | 0.0960 |
| 點雲 acc_median（點數加權） | 2.229 mm | 2.601 mm |
| 點雲 acc_mean | 3.026 mm | 2.981 mm |
| 點數 | 1,480,606 | 25,682 |

中位數差 17%、平均幾乎相同（反光略低），代表反光區是「典型值稍差、極端離群值沒有比較多」。

**表面朝向**（由真值深度算法線，不經模型）：

| | \|cos θ\| 中位數 | 平均 | 下四分位 | >0.9 的比例 |
|---|---|---|---|---|
| clean | 0.6557 | 0.6435 | 0.4902 | 13.3% |
| specular | 0.8228 | 0.7839 | 0.6738 | 32.3% |

`|cos θ|` 是表面法線與視線夾角的餘弦：1 = 正對鏡頭，0 = 擦過。反光像素落在
「幾乎完全正對」表面上的機率是乾淨像素的 **2.4 倍**——這是反光形成的必要條件。

### D.3 幾何論證

點雲 accuracy 量的是點到真值**表面**的最短距離，深度誤差量的是沿**視線**的偏差。
兩者的關係是 `距離 ≈ 沿視線誤差 × |cos θ|`：正對的表面付全額，斜的表面被折扣。

中位數比值 0.8228 / 0.6557 = **1.255**，即光靠角度差異就會把反光區的距離放大 25.5%，
**大於實際觀察到的 17%**。校正後反光區 2.601 / 1.255 = 2.07 mm，乾淨區 2.229 mm。

這是一階估計（角度中位數比值除距離中位數，非逐點校正），方向與量級可信，小數點後第二位不可信。

### D.4 範圍限制與未解

- **C3VD 是矽膠模型**：照明受控，沒有 AWB 重鎖、體液、煙霧。負結果不保證轉移到
  gastric / lesion。
- **跨幀一致性未測**。融合點雲的分數把「每幀各自算偏」與「幀間彼此不一致」混在同一個
  數字裡，總量相抵不代表成分相同。唯一能隔離它的工具是 `diag/vis/point_track_glare.py`
  （追同一個 3D 點跨幀的深度標準差），目前未跑。
- **目視印象與數字方向相反**：3D 點雲上可見部分反光區一致性較差、極少數較遠的反光表面
  有凸起；數字則說校正後反光略優。凸起在 acc_mean 上沒有反映，與「極少數」一致。
- **注入 arm 從未跑過**。自然反光沒有可測傷害，不代表更強的反光沒有。

---

## E. SCARED 成分消融（2026-09）— 主線在這裡

12 支 arm 建在同一個底座（`scared_cam_vanilla_depth`：depth head 解凍 + `loss.depth` α=0.02）。
數字全在 [results/medical_part2.md](results/medical_part2.md)。判斷顯著性一律用該節列的擺動上限
（pose c5 是 2.4%），**全部單一 seed**。

**pose 主指標是 c5**（2026-10-02 定）：它是唯一能和論文逐序列對比的設定（見 [results/sota.md](results/sota.md) §2.1）。
c64 當作參考。

### E.1 已經成立的

**最佳組合是 temporal + flow_geom + camera_smooth（`tfs`）**，pose c5 0.0709，比底座 −16.7%，
多張 depth 與兩種 pose 都是全表第一。它超過 temporal 單獨（−11.5%）加 flowgeom_smooth 單獨
（−4.7%）的總和，**架構時序與 loss 端幾何約束是互補的**。

**單張 depth 分不出勝負。** 同底座 12 支全部落在 0.0493–0.0504 的 1% 區間內。唯一真實的落差
是兩個底座之間（凍結 0.0561 對解凍 0.0501），那是解凍 depth head 買到的，不是任何成分。

**temporal 的效果約三分之二來自「幀順序資訊」，不是它的 100.8M 參數。** 零新模組、768 個參數的
時間相位 RoPE 在 pose c5 拿到 −7.6%，完整 temporal 是 −11.5%；兩者疊加 −10.1%，重疊而非互補。

**載體比資訊重要。** 同樣注入幀索引，重切 head_dim（`rope3d`）是 +11.0%（全表唯一比底座差的），
加在原有角度上（`ropet`）是 −7.6%。差別在於重切改掉每個通道的位置語意，而 32 個凍結的
attention block 無法適應——零訓練探測顯示光是重切就讓 pose c5 掉 112%。

### E.2 負結果

**光照 token 對 pose 沒有貢獻通道。** 證據四項互相印證：單獨跑四項全在雜訊帶（c5 −6.2% 是唯一
超過門檻的）；疊到 `tfs` 上讓 −16.7% 退成 −12.0%；SCARED 與 C3VD 兩次切斷檢驗都顯示切掉它
pose 只動 0.7% / 1.5%，比切一個普通 register token 還小；weight 1.0 與 0.2 的差距全在雜訊內。

病因量到了，而且修法已驗證一半：`L_illu` 佔訓練目標 76%，在共享 trunk 上的梯度範數是 camera 的
20 倍、camera_smooth 的 440 倍，但兩兩餘弦落在 −0.047 至 +0.025，**是量級擠壓不是方向衝突**。
把權重降到 0.05（1/20.6，梯度對齊 camera）之後，`tfsi_w005` 的 pose c5 回到 **0.0717**，
落在事前定的「不損害」帶（≤ 0.0726），pose c64 甚至是 0.0603（`tfs` 是 0.0609）。
**代價是光照本身學得較差**：illu corr 由 weight 1.0 的 0.8432 降到 0.7347。所以這是一個
「光照不再傷害 pose、但也不再是有力的賣點」的交換，不是白拿的改善。單一 seed。

**時間相位 RoPE 的效果依賴 context 長度。** 三種組合（單獨、加 temporal、加兩個 loss）在
pose c64 上都是 +8% 左右的傷害，幅度一致；而在 c5 上都是有效的。alpha 只在 t ≤ 12 的 clip 上
學過（訓練 `img_nums [4,12]`），chunk 64 讓它進入沒見過的相位範圍。**這是外插，不是組合方式
的問題。** 最便宜的修法是把時間頻率底數由 10 降到 3，未試。

### E.3 attention 先驗（跨底座，尚未對齊）

幀距高斯帶 bias 明顯優於前兩代：凍結底座上 −14.8%，對比只看幀號差的 −8.8% 與看 DINO 相似度
的 −6.4%。搬到解凍底座後縮到 −8.7%。per-token 自適應 DINO 距離帶（`tokdino`）在解凍底座上是
0.0850，與底座 0.0851 相同，**目前看不到效果**。patch query 也吃 bias 的版本技術上已可行
（flex_attention，12 幀 fwd+bwd 17.3 GiB），但訓練慢 7 倍、推論慢 2.28 倍，未跑。

### E.4 缺口

- 三個貢獻全開（temporal + 兩個 loss + attn bias）在另一台機器進行中（`tfsd`、`tfsdi`），主表的「完整模型」那一列還是空的。
- 全部單一 seed；`tfs` 的 −16.7% 只有一次觀測。
- 依 baseline 政策要有的「微調過的 MonST3R」至今沒訓練過。
- 私人資料集（lesion / gastric）沒有用這批 ckpt 更新。

---

## 已收掉的線（不再投入，保留記錄）

| 線 | 為什麼收 | 記錄在 |
|---|---|---|
| v1 雙場 `X = X^can + m·Δ` | 雙線性不可辨識 + 動態區偷懶捷徑 | [archive/checkpoints.md](checkpoints.md) §2 |
| v2 學習式 mask | 跨域崩塌（PO AUC 0.88 → Sintel 0.45） | 同上 |
| Motion Loss / static-photo（`_photo`） | clean 重跑是負結果：最佳 0.1583 比自己的起點 0.1533 還差，且單調惡化 | [method.md](method.md) §6 |
| `L_ego_flow` | 陰性收束。低視差序列任何 flow loss 都無效（目標無關） | [ego_flow.md](topics/ego_flow.md) |

---

## 下一步（依優先序，未經確認）

1. **拆 `inst_gts` 的兩個因子**（temporal-only / smooth-only）。A.2 之後這件事的重要性升高了：
   如果 gate 只值 −0.15%，那這條線的賣點就在這個沒拆的因子裡。
2. **決定 gate 的 bias 函數要不要改**。A.2 已經量到 hard/top-k 能走掉 81% 的頭空，
   但那是 eval-only override；要變成方法就得決定是改 bias 函數、改 loss 校準、還是兩者。
3. **跑 `_lowalpha`**（depth arm 的修正版），以及 L_inf 的 weight=0 對照組。
4. **決定醫學線用哪個指標敘事**（`wide` vs `smooth_temporal`），B.2 的第一項。
