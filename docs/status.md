# 研究進度：現在信什麼

> **建立 2026-08-31。這是全 repo 唯一的「當前狀態」來源。**
> 機制怎麼運作看 [method.md](method.md)；每個 run 的身分與下場看 [experiments.md](experiments.md)；
> 數字本身看三份數據表（[results/natural.md](results/natural.md) 自然場景 ／ [results/medical.md](results/medical.md) 醫學
> ／ [results/sota.md](results/sota.md) 對外對照）。**這四份都不重複下結論，結論只寫在這裡。**

## 一句話

Gate（正式版 v1）在自然場景把 Sintel ATE 從 0.1714 帶到 0.1343，但**增益幾乎全部來自
temporal + camera_smooth，gate 本身的 soft bias 目前近乎無作用**（predicted −0.15%，oracle −5.85%）；
醫學域（SCARED）是另一條較晚開的線，pose 相對微調 baseline 有 17–24% 改善、depth 幾乎沒動；
照明穩健性（L_inf）機制已實作、第一次 run 失敗、**結論未定**。

---

## 三條線的狀態

| 線 | 域 | 狀態 | 最後一個可信數字 | 依據 |
|---|---|---|---|---|
| **A. Gate（正式版 v1）** | Sintel / PointOdyssey | 🟡 機制成立、兌現不足 | Sintel ATE **0.1343**（`inst_gts` ep30） | [results/natural.md](results/natural.md) 表 1 |
| **B. 醫學域適配** | SCARED / C3VD / 私人內視鏡 | 🟢 有正向結果，但排名未定 | snippet ATE **0.0597**（chunk 64）、單張 AbsRel **0.0537** | [results/medical.md](results/medical.md) §1 |
| **C. 照明穩健性（L_inf）** | SCARED | ⚪ 機制已實作，**尚未有結論** | — | [method.md](method.md) §5 |

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

**全序列 evo ATE 不報。** Sim3 拼接讓該指標擺動 −1%~+32% 且不可預測 → EndoSfM3D 系的 pose 欄填不了。
**是不估，不是估錯。**

### B.3 兩個負結果

- **depth head 解凍失敗**：`..._depth` arm 的 test depth 反而比凍結的頭差
  （AbsRel 0.0434 → 0.0468），pose 退 18%（0.0597 → 0.0707）。病因已定位到 confidence loss
  的 `−alpha·log(c)` 項（見 [experiments.md](experiments.md)）。修正版 `_lowalpha` **尚未跑**。
- **gate 在內視鏡上幾乎不點火**：`outputs/gate_medical/*/stats.json`，`inst_g` 在 SCARED
  keyframe3/4 的 σ(g) 平均只有 0.005–0.010，`frac_gt_half = 0.0`——**沒有任何一個 patch 越過
  kink**。這與 A.2 的死區診斷同向，但在醫學域更極端（本來就沒有動態物體）。

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
