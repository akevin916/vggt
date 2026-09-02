# 數據表 — 對外對照（SCARED vs 已發表方法）

> **角色：只放「我們 vs 已發表方法」的對照與其協定辯護。**
> 自家 arm 之間的消融數字在 [results/medical.md](medical.md)，自然場景在 [results/natural.md](natural.md)，
> 每個 arm 的身分與下場在 [experiments.md](../experiments.md)。
> **同一個數字只准存在一份表裡**——引用要用連結，不要複製貼上。
>
> 狀態：2026-08-25 重跑。
> **本期 =** `logs/scared_cam_b16_gg_smooth_temporal/ckpts/best_ate.pt`（gate + temporal + camera_smooth）
> **baseline =** `logs/scared_cam_vanilla/ckpts/best_ate.pt`（stock VGGT-1B 微調，v3 機制全關）
> 結論尚未經過確認流程，引用前請先與最新討論核對。

**為什麼 baseline 是微調過的 VGGT，不是 pretrained**：打贏一個沒看過內視鏡的自然影像模型不是
值得主張的事。pretrained VGGT-1B 只當 sanity reference（數字見 §3.2），不列進對照表。

協定完全依照 AF-SfMLearner / EndoSfM3D，**逐幀核對過**：

- **depth**：`test_files.txt` 550 幀（幀數與幀號與我們的 `test/` 完全相同）、per-frame `np.median`
scaling、範圍 (0.01, 150] mm、scaling 後 clip 回範圍、per-frame 指標**無權重平均**。
- **pose**：`test_files_sequence{1,2}.txt` → `pose_seq/dataset{5,3}/keyframe4`（411 / 834 幀，連續）、
滑動 5-frame 窗、每窗獨立平移對齊 + 最小二乘 scale、誤差除以 N 而非 sqrt(N)。
- 程式：`training/benchmark/eval_scared.py --depth_protocol afsfm`。

⚠️ **輸入張數欄不能省**。published 方法都是單目（一次一張）；我們有些數字讓模型一次看很多張，
那不是同一個設定。詳見 §3.2。

---



## 1. Depth


| Method            | Venue      | 輸入   | AbsRel↓    | SqRel↓     | RMSE↓      | RMSELog↓   |
| ----------------- | ---------- | ---- | ---------- | ---------- | ---------- | ---------- |
| Fang et al.       | WACV'20    | 單張   | 0.078      | 0.794      | 6.794      | 0.109      |
| Monodepth2        | ICCV'19    | 單張   | 0.069      | 0.577      | 5.546      | 0.094      |
| Endo-SfM          | MIA'21     | 單張   | 0.062      | 0.606      | 5.726      | 0.093      |
| AF-SfMLearner     | MIA'22     | 單張   | 0.059      | 0.435      | 4.925      | 0.082      |
| DUSt3R            | CVPR'24    | pair | 0.059      | 0.516      | 4.947      | 0.080      |
| ESRT-Full（上期）     | —          | pair | 0.053      | 0.454      | 4.791      | 0.076      |
| DARES             | ECCVW'24   | 單張   | 0.052      | 0.356      | 4.483      | 0.073      |
| EndoDAC           | MICCAI'24  | 單張   | 0.052      | 0.362      | 4.464      | 0.073      |
| Endo-FASt3r       | MICCAI'25  | 單張   | 0.051      | **0.354**  | 4.480      | —          |
| EndoSfM3D         | MICCAIW'25 | 單張   | 0.050      | 0.389      | 4.749      | 0.070      |
| ESRT-Row（上期）      | —          | pair | **0.048**  | 0.372      | 4.242      | 0.069      |
| VGGT 微調（baseline） | —          | 單張   | 0.0561     | 0.4503     | 4.9296     | 0.0792     |
| VGGT 微調（baseline） | —          | 全序列  | 0.0454     | 0.3305     | 4.2538     | 0.0675     |
| **本期**            | —          | 單張   | 0.0537     | 0.4256     | 4.7842     | 0.0767     |
| **本期**            | —          | 全序列  | **0.0434** | **0.3089** | **4.0898** | **0.0647** |


δ₁：本期全序列 0.9855、本期單張 0.9779、baseline 全序列 0.9831、baseline 單張 0.9757。

### per-sequence（本期，全序列）


| keyframe           | AbsRel |
| ------------------ | ------ |
| dataset2/keyframe4 | 0.0197 |
| dataset1/keyframe3 | 0.0352 |
| dataset4/keyframe4 | 0.0365 |
| dataset6/keyframe4 | 0.0368 |
| dataset7/keyframe4 | 0.0478 |
| dataset3/keyframe4 | 0.0591 |
| dataset5/keyframe4 | 0.0645 |


最好與最差差 3.3 倍。`dataset5/keyframe4` 同時是 pose 的 Seq.1、`dataset3/keyframe4` 是 Seq.2
——**depth 最差的兩條，正好就是 pose 用的那兩條**。

---



## 2. Pose（5-frame snippet ATE）

Seq.1 = `dataset5/keyframe4`（411 幀），Seq.2 = `dataset3/keyframe4`（834 幀）。


| Method            | Venue      | 輸入   | Seq.1      | Seq.2      |
| ----------------- | ---------- | ---- | ---------- | ---------- |
| EndoSfM3D         | MICCAIW'25 | 單目   | 0.0791     | 0.0529     |
| Monodepth2        | ICCV'19    | 單目   | 0.0769     | 0.0554     |
| Endo-SfM          | MIA'21     | 單目   | 0.0759     | 0.0500     |
| AF-SfMLearner     | MIA'22     | 單目   | 0.0742     | 0.0478     |
| Endo-FASt3r       | MICCAI'25  | 單目   | **0.0702** | **0.0438** |
| VGGT 微調（baseline） | —          | 5 幀  | 0.1017     | 0.0712     |
| VGGT 微調（baseline） | —          | 64 幀 | 0.0902     | 0.0626     |
| **本期**            | —          | 5 幀  | 0.0844     | 0.0567     |
| **本期**            | —          | 64 幀 | 0.0716     | 0.0478     |


**5 幀 vs 64 幀**：每個 5 幀窗的位姿是從多寬的 context 推出來的。5 幀 = 窗自己一次 forward
（最貼近 published 方法的設定）；64 幀 = 從 64 幀的 chunk 取出。兩者算指標的方式相同，
但預測本身不同——64 幀對每個 ckpt 都好 12~19%。

---



## 3. 結論



### 3.1 公平設定下 depth 仍然輸

單張輸入是 published 方法的實際設定。本期 0.0537：

- **贏**：AF-SfMLearner 0.059、Endo-SfM 0.062、Monodepth2 0.069、Fang 0.078、DUSt3R 0.059
- **平**：ESRT-Full 0.053
- **輸**：DARES / EndoDAC 0.052、Endo-FASt3r 0.051、EndoSfM3D 0.050、**上期 ESRT-Row 0.048**



### 3.2 全序列那個 0.0434 主要來自 multi-view，不是方法

同一 ckpt，單張 0.0537 → 全序列 0.0434，改善 **19%**。

決定性的證據：**pretrained VGGT-1B 完全沒微調，只靠 multi-view 就跑到 0.0488**
（單張時是 0.0758），已贏過表上除 ESRT-Row 外的全部方法。這說明這張表的排名很大一部分是
「一次看幾張影像」決定的。把 0.0434 放進單目方法的排名裡會被 reviewer 直接駁回。

### 3.3 相對微調 baseline，增益集中在 pose


| 指標               | baseline | 本期     | 改善       |
| ---------------- | -------- | ------ | -------- |
| depth 單張 AbsRel  | 0.0561   | 0.0537 | −4.3%    |
| depth 全序列 AbsRel | 0.0454   | 0.0434 | −4.4%    |
| pose Seq.1（5 幀）  | 0.1017   | 0.0844 | −17%     |
| pose Seq.2（5 幀）  | 0.0712   | 0.0567 | −20%     |
| pose Seq.1（64 幀） | 0.0902   | 0.0716 | −21%     |
| pose Seq.2（64 幀） | 0.0626   | 0.0478 | **−24%** |


**depth 幾乎沒動（4%），pose 有 17–24%。** 而 pose 正是這條研究線的主張，方向是對的。

⚠️ **這個 −24% 混了兩件事**：baseline 從 stock VGGT-1B 暖啟動，本期從 `inst_g` 暖啟動
（帶 PointOdyssey 的 gate 預訓練）。用 `scared_cam_b16`（同 lineage、但 temporal γ=0 恆等 +
gate 凍結 ≈ 原版架構）可以拆開：pose 64 幀 Seq.2 = 0.0626 → 0.0565 → 0.0478，
**lineage 約佔 −10%、架構約佔 −15%**；depth 上 lineage 買不到東西（0.0561 → 0.0563）。

### 3.4 pose 追上了，但還沒贏

64 幀設定下 Seq.2 的 0.0478 **追平 AF-SfMLearner**，Seq.1 的 0.0716 逼近 Endo-FASt3r 的 0.0702。
但那是 64 幀 context；公平的 5 幀設定（0.0844 / 0.0567）兩條都還輸給全部 published 方法。

### 3.5 depth 最差的序列就是 pose 用的序列

`dataset5` / `dataset3` 是 depth per-sequence 的後兩名（0.0645 / 0.0591，其餘五條都在 0.048 以下），
而它們正是 pose 的 Seq.1 / Seq.2。這兩件事有沒有共同成因，未查。

---



## 4. 未報告的項目與原因

**全序列 evo ATE**：需要多次 forward（VGGT 在此解析度單次上限 80 幀，量測工具 `diag/frame_capacity.py` 已於 2026-08-19 刪除），
而 Sim3 拼接會讓該指標擺動 −1% ~ **+32%**，且與 seam 數無關、不可預測
（`diag/stitch_error.py`，80 連續幀實測）。根因是模型只預測出 0.368 mm 位移而 GT 走 17.10 mm，
`correct_scale` 對齊把預測空間 1% 的抖動放大 47 倍。**選擇不估，而非估錯。**
代價是 EndoSfM3D / DARES / Endo-FASt3r 那一系的 pose 欄我們填不了——本表的 pose 只對得上
AF-SfMLearner 的 snippet 協定。

**test split 的 snippet ATE**：算不了。`eval_scared.py:168` 檢查幀是否 stride-1，
test split 間隔 1–296 幀，守衛會 skip。pose 只能用 `pose_seq`。

**EndoSfM3D pose 那一列存疑**：其論文引用的 baseline 確定是 snippet 協定（AF Table 10 明文），
但其釋出的 `dares/evaluate_pose_and_intrinsics.py` 算的是全序列 evo ATE，論文未說明自己那列用哪個。
與 AF / Endo-FASt3r 的比較紮實，與 EndoSfM3D 那列存疑。

---



## 5. 復現指令

```bash
cd training
O=logs/scared_cam_b16_gg_smooth_temporal/ckpts/best_ate.pt   # 本期
B=logs/scared_cam_vanilla/ckpts/best_ate.pt                  # baseline

# depth, 單張輸入（公平設定）
python benchmark/eval_scared.py --ckpts $O $B --split test --n_frames 0 \
  --single_view --depth_protocol afsfm

# depth, 全序列 multi-view
python benchmark/eval_scared.py --ckpts $O $B --split test --n_frames 0 \
  --chunk_size 64 --overlap 16 --depth_protocol afsfm

# pose, snippet ATE, 5 幀 context（公平設定）—— 必須給 --out_dir，否則蓋掉 64 幀那份
python benchmark/eval_scared.py --ckpts $O $B --split pose_seq --n_frames 0 \
  --chunk_size 5 --overlap 4 --no_depth \
  --out_dir ../outputs/eval_scared/pose_seq_0f_chunk5

# pose, snippet ATE, 64 幀 context
python benchmark/eval_scared.py --ckpts $O $B --split pose_seq --n_frames 0 \
  --chunk_size 64 --overlap 16 --no_depth
```

一次跑完全部：`bash benchmark/run_report_tables.sh [scared|pose5|lesion|all]`

輸出在 `outputs/eval_scared/<split>_<n>f[_single][_afsfm]/<exp>_<stem>/results.json`。
子目錄名走 `paths.exp_name_from_ckpt`——**不是** ckpt 檔名，否則多個 run 的 `best_ate.pt`
會全部撞成同一個 `best_ate/` 目錄互相覆蓋（2026-08-25 發生過）。