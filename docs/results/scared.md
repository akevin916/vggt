# 數據表 — SCARED

> 2026-10-08 由 `medical.md`（SCARED 部分）、`medical_part2.md`、`sota.md` 合併（原檔在 [archive/results/](../archive/results/)）。
> **只記數字，不下結論**，結論寫在 [README.md](../README.md)。C3VD 在 [c3vd.md](c3vd.md)，私人資料集在 [private.md](private.md)。
> 三個收起的分支（attention bias、光照 token、時間 RoPE）的數字只留在 archive。

---

## 1. 協定

| 協定 | split | 設定 | 指標 |
|---|---|---|---|
| depth 單張 | `test/depth`（550 幀 / 7 序列） | `--single_view` | AbsRel，afsfm 協定 |
| depth 多張 | `test/depth` | `chunk_size 64 overlap 16` | AbsRel，afsfm 協定 |
| pose c64 | `test/pose`（ds3 834 幀、ds5 411 幀） | `chunk_size 64 overlap 16` | 5 幀 snippet ATE（mm） |
| pose c5 | `test/pose` | `chunk_size 5 overlap 4` | 5 幀 snippet ATE（mm） |

- **afsfm 協定**：per-frame median scaling、範圍 (0.01, 150] mm、scaling 後 clip、per-frame 指標無權重平均。
- **snippet ATE**：滑動 5 幀窗、每窗獨立平移對齊 + 最小二乘 scale、誤差除以 N 而非 √N。與 AF-SfMLearner `evaluate_pose.py` 相同。
- split 與 AF-SfMLearner 逐幀相同。`test/pose` 是連續影片，`test/depth` 是官方稀疏抽樣（間隔 1–296 幀），**不能算 snippet ATE**。
- ⚠️ `test/depth/dataset3/keyframe4`（79 幀）與 `test/pose/dataset3/keyframe4`（834 幀）同名不同物，引用要帶 split。
- **ds5 = Seq.1**（`dataset5/keyframe4`），**ds3 = Seq.2**（`dataset3/keyframe4`）。
- 程式：`pipeline/benchmark/eval_scared.py --depth_protocol afsfm`，指令見 [README.md](../README.md)。

**全序列 evo ATE 不報**：VGGT 在此解析度單次上限 80 幀，Sim3 拼接讓該指標擺動 −1% ~ +32%、不可預測（`diag/stitch_error.py`）。
published 的 SCARED pose 表用的都是 AF snippet 協定，所以不影響對外比較。

---

## 2. 對外對照

已發表方法挑 4 個代表；完整 11 個方法的表在 [archive/results/sota.md](../archive/results/sota.md)。

| 方法 | 為什麼列 | 輸入 | depth 單張 | pose ds5（Seq.1） | pose ds3（Seq.2） |
|---|---|---|---|---|---|
| AF-SfMLearner（MIA'22） | 協定來源 | 單目 | 0.059 | 0.0742 | 0.0478 |
| Endo-FASt3r（MICCAI'25） | published pose 最好 | 單目 | 0.051 | **0.0702** | **0.0438** |
| EndoSfM3D（MICCAIW'25） | published depth 最好 | 單目 | 0.050 | 0.0791 | 0.0529 |
| ESRT-Row（上期） | 我們上一期 | pair | **0.048** | — | — |
| `smooth_temporal`（期中） | 期中方法 | 5 幀 / 64 幀 | 0.0537 | 0.0844 / 0.0716 | 0.0567 / 0.0478 |
| `tfs`（現在的基準） | 現在的基準 | 5 幀 / 64 幀 | 0.0498 | 0.0848 / 0.0746 | 0.0571 / 0.0472 |

引用規則：

1. **逐序列比，不用兩條平均。**
2. **只有 5 幀（c5）對應 published 設定**；64 幀（c64）沒有文獻對應，只能當補充。
3. published 方法一次 forward 看 2 幀，我們的 c5 看 5 幀；這是方法的一部分，不做對齊版，但表上「輸入」欄要保留。
4. EndoSfM3D 自己那一列的協定論文沒寫（它釋出的程式算全序列 evo ATE，但數字量級像 snippet），引用時加註。
5. 不照抄 EndoDAC 的 95% 信賴區間：相鄰窗共用 4 幀，`st.sem` 會偏窄；要報不確定性就在序列內做 block bootstrap。

協定傳承與 GT 核對的完整記錄見 [archive/results/sota.md](../archive/results/sota.md) §2.1。

---

## 3. 自家 arm 主表

**每一格取 `epoch_10` 與 `best_ate` 中較好的值**；ᵇ = 來自 `best_ate`，無標記 = `epoch_10`。這等於用 test 挑 ckpt，數字偏樂觀。
兩支的原始值見 §5。

| run | 深度頭 | depth 單張 | depth 多張 | pose c64 | pose c5 |
|---|---|---|---|---|---|
| `VGGT-1B` 零樣本 | — | 0.0758 | 0.0488 | 0.1209 | 0.1421 |
| `vanilla_frozen` | 凍結 | 0.0561ᵇ | 0.0454ᵇ | 0.0764ᵇ | 0.0865ᵇ |
| `b16` | 凍結 | 0.0563 | 0.0459 | 0.0706 | 0.0824 |
| `b16_gg` | 凍結 | 0.0550 | 0.0449 | 0.0699 | 0.0810 |
| `smooth_temporal` | 凍結 | 0.0537ᵇ | 0.0434ᵇ | 0.0597ᵇ | 0.0705ᵇ |
| `wide` | 凍結 | 0.0550 | **0.0419** | 0.0705 | 0.0865 |
| `vanilla` | 解凍 | 0.0499ᵇ | 0.0450 | 0.0711 | 0.0851 |
| `temporal` | 解凍 | 0.0501 | 0.0426ᵇ | 0.0669 | 0.0753 |
| `flowgeom` | 解凍 | 0.0502 | 0.0420 | 0.0696 | 0.0822 |
| `smooth` | 解凍 | **0.0495** | 0.0437ᵇ | 0.0681 | 0.0859 |
| `flowgeom_smooth` | 解凍 | 0.0501 | 0.0426ᵇ | 0.0671 | 0.0811ᵇ |
| `tfs` | 解凍 | 0.0498 | **0.0409ᵇ** | **0.0609** | **0.0709** |

- **凍結組**（期中，2026-08-25 評）：
  - `vanilla_frozen` = `scared_cam_vanilla`，從 VGGT-1B 微調，只有 `best_ate`。
  - `b16` / `b16_gg` 從 `inst_g` 起訓，gate 與 temporal 都建了但凍結；`b16_gg` 唯一的差別是 gate 可訓練。⚠️ `b16_gg` 在 ep8 被中斷，是 8-epoch 產物。
  - `smooth_temporal` = `scared_cam_b16_gg_smooth_temporal`，從 `b16_gg` 接續，temporal 解凍 + camera_smooth（weight 3.0），gate 可訓練。
  - `wide` 與 `smooth_temporal` 只差取樣窗（`nearby_expand_range=120`）。
  - `vanilla` 取 4–12 張，`b16` 系列取 4–16 張，兩者之差不是純架構效果。
- **解凍組**（2026-09）：
  - 共用底座 `scared_cam_vanilla_depth` = 從 VGGT-1B 起，depth head 解凍 + `loss.depth` α 0.02。
  - 成分字母：t = temporal block、f = `flow_geom`、s = camera_smooth；`tfs` = t+f+s。
  - 這組都沒有 gate。
- ⚠️ **凍結組的 json 已遺失**：`b16`、`b16_gg`、`smooth_temporal`、`wide` 的數字取自舊 `medical.md`，`b16` / `b16_gg` / `wide` 當時沒記是哪支 ckpt；`smooth_temporal` 依 sota.md 為 `best_ate`。
  `VGGT-1B` 與 `vanilla_frozen` 的 json 還在，和文件一致。
- 凍結組與解凍組的 depth 起點不同（單張 0.0561 對 0.0499），**depth 欄不可跨組比**。

---

## 4. pose 逐序列值（ds3 / ds5）

取主表同一支 ckpt。

| run | c64 | c5 |
|---|---|---|
| `VGGT-1B` 零樣本 | 0.1159 / 0.1258 | 0.1294 / 0.1549 |
| `vanilla_frozen` | 0.0626 / 0.0902 | 0.0712 / 0.1017 |
| `b16` | 0.0586 / 0.0826 | 0.0675 / 0.0973 |
| `b16_gg` | 0.0565 / 0.0832 | 0.0663 / 0.0956 |
| `smooth_temporal` | 0.0478 / 0.0716 | 0.0567 / 0.0844 |
| `wide` | 0.0585 / 0.0825 | 0.0723 / 0.1007 |
| `vanilla` | 0.0565 / 0.0857 | 0.0694 / 0.1008 |
| `temporal` | 0.0531 / 0.0807 | 0.0596 / 0.0910 |
| `flowgeom` | 0.0545 / 0.0848 | 0.0656 / 0.0988 |
| `smooth` | 0.0537 / 0.0825 | 0.0692 / 0.1027 |
| `flowgeom_smooth` | 0.0527 / 0.0816 | 0.0646 / 0.0976ᵇ |
| `tfs` | 0.0472 / 0.0746 | 0.0571 / 0.0848 |

---

## 5. 兩支 ckpt 的原始值（`epoch_10` / `best_ate`）

只有解凍組兩支都有。

| run | depth 單張 | depth 多張 | pose c64 | pose c5 |
|---|---|---|---|---|
| `vanilla` | 0.0501 / 0.0499 | 0.0450 / 0.0451 | 0.0711 / 0.0736 | 0.0851 / 0.0861 |
| `temporal` | 0.0501 / 0.0505 | 0.0432 / 0.0426 | 0.0669 / 0.0689 | 0.0753 / 0.0764 |
| `flowgeom` | 0.0502 / 0.0506 | 0.0420 / 0.0429 | 0.0696 / 0.0713 | 0.0822 / 0.0822 |
| `smooth` | 0.0495 / 0.0500 | 0.0440 / 0.0437 | 0.0681 / 0.0699 | 0.0859 / 0.0864 |
| `flowgeom_smooth` | 0.0501 / 0.0504 | 0.0427 / 0.0426 | 0.0671 / 0.0681 | 0.0821 / 0.0811 |
| `tfs` | 0.0498 / 0.0501 | 0.0413 / 0.0409 | 0.0609 / 0.0624 | 0.0709 / 0.0724 |

來源 `outputs/eval_scared/{test_depth_0f_single_afsfm,test_depth_0f_afsfm,test_pose_0f,test_pose_0f_chunk5}/<exp>_<ckpt>/results.json`（2026-10-08 重新從 json 讀出）。

**ckpt 漂移**（同一個 run 的 `best_ate` 與 `epoch_10` 之間的擺動上限，2026-09 的 7–8 個 run 取最大值）：
depth 單張 1.0%、depth 多張 2.3%、pose c64 3.4%、pose c5 2.4%。

**seed 擺動**（seed 43 對 seed 42，`epoch_10`，另一台跑，本機無 json）：
`vanilla` 單張 −0.6%、多張 +2.2%、c64 −0.3%、c5 +0.1%；`tfs` 單張 +1.0%、多張 **+6.5%**、c64 −1.1%、c5 +1.1%。

---

## 6. 缺口

| 缺口 | 現況 |
|---|---|
| seed | 全部單一 seed；`tfs` 的多張 depth seed 擺動 6.5%，超過 ckpt 漂移 |
| 凍結組 json | 已遺失，數字只剩文件；`b16` / `b16_gg` / `wide` 的 ckpt 未記錄 |
| gate 在解凍底座 | 解凍組沒有任何 run 開 gate |
| 微調過的 MonST3R | baseline 政策要求，至今未訓練 |
| EndoSfM3D pose 列 | 協定論文未載明 |
| depth 最小深度 | AF 與 EndoDAC 用 1e-3 mm，EndoSfM3D 與我們用 1e-2 mm；應無像素落在其間，未實際量 |
