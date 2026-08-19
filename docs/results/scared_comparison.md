# SCARED 對照表與消融

> 狀態：2026-08-19 首批數字。ckpt = `logs/scared_cam_b16_gg/ckpts/best_ate.pt`
> （SCARED 微調，訓練跑到 epoch 8/10 被中斷，此 ckpt 是 epoch 7）。
> 結論尚未經過確認流程，引用前請先與最新討論核對。

協定完全依照 AF-SfMLearner / EndoSfM3D，逐行核對過其原始碼（見 `docs/planning/` 與 project memory）：

- **depth**：`test_files.txt` 550 幀、per-frame `np.median` scaling、深度範圍 (0.01, 150] mm、
  scaling 後 clip 回範圍、per-frame 指標**無權重平均**。
- **pose**：`test_files_sequence{1,2}.txt`（410 / 833 幀）、滑動 5-frame 窗口、
  每窗獨立平移對齊 + 最小二乘 scale、誤差除以 N 而非 sqrt(N)、尾端窗口截短不丟棄。
- 程式：`training/benchmark/eval_scared.py --depth_protocol afsfm`。

---

## 1. Depth

| Method | Venue | 輸入 | AbsRel↓ | SqRel↓ | RMSE↓ | RMSELog↓ |
|---|---|---|---|---|---|---|
| Fang et al. | WACV'20 | 單張 | 0.078 | 0.794 | 6.794 | 0.109 |
| Monodepth2 | ICCV'19 | 單張 | 0.069 | 0.577 | 5.546 | 0.094 |
| Endo-SfM | MIA'21 | 單張 | 0.062 | 0.606 | 5.726 | 0.093 |
| AF-SfMLearner | MIA'22 | 單張 | 0.059 | 0.435 | 4.925 | 0.082 |
| DUSt3R | CVPR'24 | pair | 0.059 | 0.516 | 4.947 | 0.080 |
| DARES | ECCVW'24 | 單張 | 0.052 | 0.356 | 4.483 | 0.073 |
| EndoDAC | MICCAI'24 | 單張 | 0.052 | 0.362 | 4.464 | 0.073 |
| Endo-FASt3r | MICCAI'25 | 單張 | 0.051 | **0.354** | 4.480 | — |
| EndoSfM3D | MICCAIW'25 | 單張 | 0.050 | 0.389 | 4.749 | 0.070 |
| ESRT-Full（上期） | — | pair | 0.053 | 0.454 | 4.791 | 0.076 |
| ESRT-Row（上期） | — | pair | 0.048 | 0.372 | 4.242 | 0.069 |
| VGGT-1B zero-shot | — | 單張 | 0.0758 | 0.7671 | 6.5298 | 0.1046 |
| VGGT-1B zero-shot | — | 全序列 | 0.0488 | 0.3742 | 4.6048 | 0.0735 |
| **本期（微調）** | — | 單張 | 0.0550 | 0.4440 | 4.9193 | 0.0785 |
| **本期（微調）** | — | 全序列 | **0.0449** | **0.3176** | **4.2263** | **0.0663** |

δ₁：本期全序列 0.9852、本期單張 0.9757、VGGT-1B 全序列 0.9784、VGGT-1B 單張 0.9472。

### per-sequence（本期全序列）

| keyframe | AbsRel |
|---|---|
| dataset2/keyframe4 | 0.0199 |
| dataset4/keyframe4 | 0.0369 |
| dataset6/keyframe4 | 0.0380 |
| dataset1/keyframe3 | 0.0433 |
| dataset7/keyframe4 | 0.0476 |
| dataset5/keyframe4 | 0.0620 |
| dataset3/keyframe4 | 0.0626 |

最好與最差差 3 倍。`dataset3/keyframe4` 同時是 pose 的 Seq.2。

---

## 2. Pose（5-frame snippet ATE）

| Method | Venue | Seq.1 | Seq.2 |
|---|---|---|---|
| Monodepth2 | ICCV'19 | 0.0769 | 0.0554 |
| Endo-SfM | MIA'21 | 0.0759 | 0.0500 |
| AF-SfMLearner | MIA'22 | 0.0742 | 0.0478 |
| EndoSfM3D | MICCAIW'25 | 0.0791 | 0.0529 |
| Endo-FASt3r | MICCAI'25 | **0.0702** | **0.0438** |
| VGGT-1B zero-shot | — | 0.1258 | 0.1159 |
| **本期（微調）** | — | 0.0832 | 0.0565 |

Seq.1 = `dataset5/keyframe4`（411 幀），Seq.2 = `dataset3/keyframe4`（834 幀）。

---

## 3. 四個結論

### 3.1 公平設定下我們輸

單張輸入是 published 方法的實際設定。我們 0.0550：

- **贏**：AF-SfMLearner 0.059、Endo-SfM 0.062、Monodepth2 0.069、Fang 0.078、DUSt3R 0.059、ESRT-Full 0.053
- **輸**：DARES / EndoDAC 0.052、Endo-FASt3r 0.051、EndoSfM3D 0.050、**上期 ESRT-Row 0.048**

### 3.2 那個 0.0449 主要來自 multi-view，不是方法

同一 ckpt，單張 0.0550 → 全序列 0.0449，改善 **18%**。
更關鍵：**zero-shot 的 VGGT-1B 只靠 multi-view 就跑到 0.0488**，已贏過表上除 ESRT-Row 外全部方法——
一個沒看過任何內視鏡影像的自然影像模型。這說明這張表的排名主要由「一次看幾張影像」決定。
把 0.0449 放進單目方法的排名裡會被 reviewer 直接駁回；輸入欄不能省。

### 3.3 微調有效，幅度不小

| 指標 | VGGT-1B | 微調後 | 改善 |
|---|---|---|---|
| depth 單張 AbsRel | 0.0758 | 0.0550 | −27% |
| depth 全序列 AbsRel | 0.0488 | 0.0449 | −8% |
| pose Seq.1 | 0.1258 | 0.0832 | −34% |
| pose Seq.2 | 0.1159 | 0.0565 | **−51%** |

domain gap 是真的，微調是解方——這條敘事成立，而且 pose 的改善幅度大於 depth。

### 3.4 pose 是短板，且 multi-view 救不了

微調後仍是表上最差。depth 能靠 multi-view 大贏、pose 卻不行。
**而 pose 正是這條研究線的主張。** 這個不對稱是目前最值得追的問題。

---

## 4. 未報告的項目與原因

**全序列 ATE**：需要多次 forward（VGGT 在此解析度單次上限 80 幀，`diag/frame_capacity.py`），
而 Sim3 拼接會讓該指標擺動 −1% ~ **+32%**，且與 seam 數無關、不可預測
（`diag/stitch_error.py`，80 連續幀實測）。根因是模型只預測出 0.368 mm 位移而 GT 走 17.10 mm，
ATE 的 `correct_scale` 對齊把預測空間 1% 的抖動放大 47 倍。**選擇不估，而非估錯。**

snippet ATE 不受影響：每個 5 幀窗口自己重新歸零與配尺度，不需要全域座標系，
所以用重疊 chunk 逐窗計算與整段計算 **bit-identical**（已驗證）。

**EndoSfM3D pose 那一列存疑**：其論文引用的 baseline 確定是 snippet 協定（AF Table 10 明文），
但其釋出的程式碼 `dares/evaluate_pose_and_intrinsics.py` 算的是全序列 evo ATE，
論文未說明自己那列用哪個。與 AF / Endo-FASt3r 的比較紮實，與 EndoSfM3D 那列存疑。

---

## 5. 復現指令

```bash
cd training
B=logs/scared_cam_b16_gg/ckpts/best_ate.pt; V=checkpoints/VGGT-1B.pt

# depth, 全序列 multi-view
python benchmark/eval_scared.py --ckpts $B $V --split test --n_frames 0 \
  --chunk_size 64 --overlap 16 --depth_protocol afsfm

# depth, 單張輸入（公平設定）
python benchmark/eval_scared.py --ckpts $B $V --split test --n_frames 0 \
  --single_view --depth_protocol afsfm

# pose, snippet ATE
python benchmark/eval_scared.py --ckpts $B $V --split pose_seq --n_frames 0 \
  --chunk_size 64 --overlap 16 --no_depth
```

輸出在 `outputs/eval_scared/<split>_<n>f[_single][_afsfm]/<ckpt>/results.json`。
