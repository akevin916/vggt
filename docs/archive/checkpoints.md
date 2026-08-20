# 權重登記表 —— 每支權重做了什麼、數字多少

> 建立 2026-08-20。**用途**：checkpoint 檔案本身是純 `state_dict`，裡面沒有 epoch、沒有 config、
> 沒有任何 metadata，檔名一旦說不清就永遠說不清。這份表是權重身分的唯一去處。
>
> 刪權重之前先確認它在這裡有一列。**沒有數字的不刪**（見 §1.4）。

## 判讀規則

- **Sintel 欄**一律是 `benchmark/eval_sintel.py`、14 序列、`chunk_size=0`（完整序列，平均 45.9 幀）、
  `max_depth=80`。數字全部搬自 [../table.md](../table.md) 表 1，該表是唯一權威來源。
  ATE(12) = 去掉 `cave_2`/`temple_3` 兩個離群序列後的平均。
- **SCARED 欄**是 `benchmark/eval_scared.py`、val split、6 keyframe × 50 幀、單位 **mm**，
  括號是佔 GT 軌跡尺度的比例。
- ⚠️ **雜訊帶 ±0.02 ATE（±13%）**（table.md 缺口 #7）—— 兩支權重差距小於此就不能說誰比較好。
- **世代**：`v3-bug` 指 gate 前向 bug 修正（commit `e064087`, 2026-07-09）之前啟動的 run，
  數字不可信、只能當歷史紀錄；`v3-clean` 是修正後。

---

## 1. 登記表

### 1.1 現役 —— `training/checkpoints/`（weights-only，~4.7 G，可 warm-start / eval，**不能 resume**）

| 權重 | 世代 | 做了什麼 | Sintel ATE | ATE(12) | RPE-t | RPE-r | AbsRel | SCARED ATE (mm) |
|---|---|---|---|---|---|---|---|---|
| `VGGT-1B.pt` | base | 上游 pretrained，未經任何微調 | 0.1714 | 0.0618 | 0.0617 | 0.4706 | 0.2747 | 1.818 (4.39%) |
| `inst_gate_init.pt` | v3-clean | 只訓 `gate_predictor`，其餘全凍；camera 路徑仍是 base。gate zero-init ≈ no-op，所以數字該等於 base。**所有 v3 run 的 warm-start 起點，孤本** | 0.1715 † | — | — | — | — | 1.817 |
| `inst_g.pt` | v3-clean | gate + 後段 global block + camera head（§8.3 run 1，ep15）。**Sintel depth 七項全贏 base** | **0.1533** | **0.0517** | 0.0671 | 0.4923 | **0.2136** | **1.734 (4.15%)** |
| `inst_gts.pt` | v3-clean | run 1 再加 temporal 解凍 + `L_camera_smooth`（`smooth_temporal` ep30）。**Sintel pose 最好** | **0.1343** | **0.0500** | 0.0633 | **0.3943** | ‡ | 1.887 |

† `inst_gate_init` 的 0.1715 來自 ckpt 命名整理的紀錄，**未在 table.md 立表**，屬單一來源。
‡ `inst_gts` 的 depth 欄用的是 `max_depth=70` 協定，與本表 max80 不同欄，不可同欄比。

> **注意 Sintel 與 SCARED 的排序相反**：Sintel 最好的 `inst_gts`（0.1343）在 SCARED 是四支裡**最差**
> （1.887，比未微調的 VGGT-1B 還差）。跨域不是自動成立的。

### 1.2 已封存 —— `archive/checkpoints/`（全部是孤本：`logs/` 沒有對應 ckpts 目錄）

原本 10 支 48.6 G，2026-08-20 刪掉 8 支（38.4 G），**現存只剩最後兩列**。刪掉的那 8 支數字全部
在下表，`table.md` 表 1 有對應列可交叉核對；但**權重本身回不來了**——它們在 `logs/` 沒有 ckpts
目錄，重現要重跑當年的 config。

| 權重 | 世代 | 做了什麼 | ATE | ATE(12) | RPE-t | RPE-r | AbsRel | 狀態 |
|---|---|---|---|---|---|---|---|---|
| `dyn_vggt_s1b.pt` | v1 | 雙場 `X=X^can+m·Δ`，S1 第二版（表 1 的 `v1 S1b`） | 0.1692 | 0.0680 | 0.0770 | 0.5502 | 0.2790 | 已刪 08-20 |
| `dyn_vggt_s1_v2.pt` | v2 | v2 的 S1（架構同 v1，見 §2.2） | 0.1732 | 0.0688 | 0.0685 | 0.4857 | 0.2558 | 已刪 08-20 |
| `dyn_vggt_s2.pt` | v1 | 雙場 S2（全網解凍）。**pose 全面崩壞**，ATE(12) 0.1113 是全表最差 | 0.2035 | 0.1113 | 0.0896 | 0.8909 | 0.2840 | 已刪 08-20 |
| `dyn_vggt_v3_s1_inst.pt` | v3-**bug** | v3 gate 的第一版，在 frame-interleaved attention bug 下訓的 | 0.1790 | 0.0706 | 0.0828 | 0.5200 | 0.2892 | 已刪 08-20 |
| `gate_badmask.pt` | v3-**bug** | **與上一列 byte-identical**（同一份權重存兩個名字） | 0.1790 | 0.0706 | 0.0828 | 0.5200 | 0.2892 | 已刪 08-20（重複） |
| `dyn_vggt_v3_s1_inst_photo.pt` | v3-**bug** | 加 static-photo loss（該 run 的 ep19）。sqRel **6.210** 全場最差（base 2.348）→ 少數像素巨大誤差 | 0.1620 | 0.0666 | 0.0813 | 0.5474 | 0.2732 | 已刪 08-20 |
| `dyn_vggt_v3_s1_smooth_temporal.pt` | v3-**bug** | temporal + camera_smooth。⚠️ **是該 run 的 ep2**（計畫 20 epoch），不是收斂值 | 0.1568 | 0.0535 | 0.0720 | 0.4634 | 0.2458 | 已刪 08-20 |
| `dyn_vggt_v3_oracle.pt` | v3-**bug** | `oracle_camera_only` 消融（ep19）：**只留 camera_head**，用 GT mask 當完美 gate。depth 全面崩潰（AbsRel 0.563、δ1 0.430），pose 數字須在此前提下讀 | 0.1757 | 0.0685 | 0.0649 | 0.4769 | **0.5631** | 已刪 08-20 |
| `dyn_vggt_s1a.pt` | v1 | 雙場 S1 第一版（推測）。**檔名未出現在 table.md** | | | | | | **保留待驗** |
| `dyn_vggt_s1_full.pt` | v1 | 雙場，全部 loss 開啟（推測）。**檔名未出現在 table.md** | | | | | | **保留待驗** |

### 1.3 訓練 run —— `training/logs/<run>/ckpts/`（含 optimizer state，6.5–9.2 G，可 resume）

> **2026-08-20 清理**：45 個 checkpoint 共 313 GB 刪除。已判陰性或棄用的線（egoflow ×3、
> `photo_smooth_temporal`、`smooth_temporal_photo`）**只留 `best_ate.pt`**；4 個 smoke run 的
> ckpt 全刪；byte-identical 的 `epoch_N` / `last` 對留 `epoch_N`（檔名帶身分，沒有 metadata 時
> 那是唯一線索），刪 `last` —— 代價是那兩個 run 不能直接 `--resume`，要 resume 得先把
> `epoch_N.pt` 複製成 `last.pt`。

逐 epoch 的數字在 `logs/<run>/pose_eval/epoch_*/results.json`（全部 2.7 MB，**永遠保留**，
刪 ckpt 不影響它）；`log.txt` 與 `tensorboard/` 同樣全數保留（共 646 MB）。下表只記每個 run
最好的那一點，「現存」欄是清理後還在磁碟上的 ckpt。

| run | 世代 | 做了什麼 | 最佳 | ATE | 現存 | 備註 |
|---|---|---|---|---|---|---|
| `dyn_vggt_v3_s1_inst` | v3-clean | §8.3 run 1，gate + camera。24 epoch | ep15 | 0.1533 | best, e10, e20, last | = `inst_g.pt` |
| `..._smooth_temporal` | v3-clean | run 1 + temporal + `L_camera_smooth`。50 epoch | ep30 | **0.1343** | best, best_loss, e10–e50 | = `inst_gts.pt`。ep30 後平台化在 0.134–0.136；windowed `best.pt` (ep17) 是 0.1651，**不可用** |
| `..._photo_smooth_temporal` | v3-**bug-init** | 三因子全開，但 warm-start 自 buggy ckpt → lineage 不乾淨 | ep16 | 0.1534 | best_ate | 全序列 0.1743，**比 base 還差，已確認棄用**。48 epoch 單調惡化到 0.2126 |
| `..._smooth_temporal_photo` | v3-clean | 在收斂的 smooth_temporal 上加 static-photo | ep3 | 0.1413 | best_ate | 只跑 3 epoch |
| `..._smooth_o1` | v3-clean | `camera_smooth` 改一階（`orders=(1,2)`） | ep4 | 0.1371 | 全 5 支 | e4 後劣化到 0.1627 |
| `..._egoflow` | v3-clean | `L_ego_flow` + 解凍 depth_head + 開 `L_depth` | ep4 | 0.1340 | best_ate | **陰性**：起點就是最好，e20 劣化到 0.1551 |
| `..._egoflow_gt` | v3-clean | 同上但 `use_gt_depth`（depth_head 保持凍結） | ep4 | 0.1346 | best_ate | 陰性，→ 0.1429 |
| `..._egoflow_gt_mask` | v3-clean | 同上 + `use_dynamic_mask: False` | ep4 | 0.1325 | best_ate | 陰性，→ 0.1388 |
| `scared_cam_b2` | v3-clean | SCARED 適應，只解凍 global block 8,9 | — | — | best_loss, last | **無 eval 數據**（當時 `pose_eval` 未開），跑到 ep3 被 b16 取代 |
| `scared_cam_b16` | v3-clean | SCARED 適應，解凍 global block 8–23 | ep8 | 0.9814 mm | best_ate, best_loss, e5, e10 | b2/b16 只差解凍層數，其餘全同 |
| `scared_cam_b16_gg` | v3-clean | b16 + `gate_pose_grad`（讓 pose loss 教 gate） | ep8 | 0.9464 mm | 全 4 支 | 中斷於 ep8/10 |
| `scared_cam_b16_gg_smooth_temporal` | v3-clean | b16_gg + temporal + camera_smooth | ep8 | **0.9108 mm** | 全 4 支 | **2026-08-20 訓練中**，尚未收斂 |
| smoke ×4 | — | wiring 驗證用（`egoflow_smoke`、`egoflow_gt_smoke`、`scared_cam_smoke`、`scared_cam_smoke_prev_0800`） | — | — | **無** | 目的是「跑不跑得起來」，權重無保留價值 |

### 1.4 為什麼 `s1a` / `s1_full` / `scared_cam_b2` 先不刪

- `dyn_vggt_s1a.pt`、`dyn_vggt_s1_full.pt` —— table.md 表 1 有 `v1 S1`(0.1710) 這一列，但**沒有任何東西
  把它綁到某個檔名**；`s1_full` 更是完全沒有對應列。檔案內部無 metadata，權重 key 只能告訴你
  「這是 v1 架構」（有 `motion_head` + `flow_head`），分不出是哪一次 run。
  **要定案**：跑 `benchmark/eval_sintel.py` 拿 mean ATE 去比對
  `v1 S1`(0.1710) / `v1 S1b`(0.1692) / `v2 S1_v2`(0.1732) / `v1 S2a`(0.2035) —— 四個值分得夠開，
  mean 就足以辨識。
- `scared_cam_b2` —— 真的訓練過（到 ep3），但當時 `pose_eval` 沒開，**一個 eval 數字都沒有**。

### 1.5 讀權重身分的可用線索

檔案裡沒有 metadata，但 `state_dict` 的 key 結構能定架構世代（免跑 eval）：

| 線索 | 意思 |
|---|---|
| 有 `motion_head.*` + `flow_head.*`（1671 keys） | **v1/v2** 雙場架構 |
| 有 `aggregator.gate_predictor.*`（1491 keys） | **v3** |
| 只有 `aggregator` + `camera_head`（1423 keys） | v3 的 `oracle_camera_only` 消融 |
| 有 `track_head.*`（1797 keys） | 上游 VGGT-1B 原版 |
| **`aggregator.temporal_blocks.*`（144 keys）** | **不是**線索 —— 所有 v3 匯出檔都有這些權重，temporal 有沒有啟用由 config 的 `aa_order` 決定，不寫在權重裡 |

---

## 2. v1 / v2 方法摘要

> 原本的三份長文件（`dyn_vggt_method_v1.md` 270 行、`dyn_vggt_method_v2.md` 39 行、
> `dyn_vggt_implementation.md` 230 行）於 2026-08-20 刪除，內容壓縮成本節。
> 一頁式沿革見 [../dyn_vggt_history.md](../dyn_vggt_history.md)，現行方法見
> [../dyn_vggt_method_v3.md](../dyn_vggt_method_v3.md)。

### 2.1 v1 —— 運動解耦的雙場表示

**問題診斷**（這部分至今仍成立，v3 用的是同一份診斷）：VGGT 的剛性靜態假設藏在三處 ——
(a) `world_points` 把所有幀統一回歸到第一幀相機座標系，動態表面點在不同時刻的不同 3D 位置無法同時表達；
(b) `aa_order=["frame","global"]` 加上只編碼 `(y,x)` 的 2D RoPE，**完全沒有幀序資訊**；
(c) camera head 把整個場景當剛體，動態物體的光流被錯誤歸因到相機運動。

**做法**（四個貢獻）：

1. **時空聚合器** —— 在 frame / global 之間插入第三種 `temporal` attention（token 重排成
   `(B*P, S, C)`，只沿時間軸做），配一個**獨立的 1D 時間 RoPE**（只在 temporal attention 內作用，
   空間 RoPE 一個字不改，用整數幀索引）。warm-start 靠 LayerScale γ=0 → 第一步嚴格等於原 VGGT。
2. **雙場表示** —— `X = X^can + m·Δ`：靜態正則點 + 動態機率 × 3D 殘差位移。
3. **動態分割頭 + 場景流頭** —— 兩個新的輕量 DPT 頭出 `m` 和 `Δ`；`m` 的自監督信號是
   「GT 光流 − 相機誘導的剛體光流」的殘差。動態屏蔽做兩處：pose 的 `valid_frame` 只數靜態像素，
   `L_point` 逐像素乘 `(1−m)`。
4. **前饋初始化的 4D 全局對齊** —— 用前饋輸出當初值，20–50 迭代收斂（MonST3R 需數百次）。

**Loss**：原有 `L_cam^stat + L_depth + L_point`，加 `L_motion`(BCE) + `L_flow`(dense，
組裝點對 GT `world_points`) + `L_reproj`(跨時刻重投影，把四個頭綁在一起) + `L_tsmooth`。
課程 S0（只訓兩個新頭）→ S1（+temporal + 原三頭，開閉環與屏蔽）→ S2（全網）。

### 2.2 v2 —— 只是 v1 的失敗診斷，沒有新架構

v2 本質是 v1 失敗診斷的整理版（`dyn_vggt_s1_v2.pt` 的權重 key 與 v1 完全相同，1671 keys，
架構沒動）。兩個致命機制：

1. **`m × Δ` 是雙線性項，不可辨識** —— 只監督相加後的結果時，`m·Δ = r` 有無限多組解
   （`(c·m, Δ/c)` 全等價），`Δ` 沒有自己的目標。
2. **動態區 `X^can` 無監督 → 偷懶捷徑** —— `(1−m)` 屏蔽關掉了動態區對 `X^can` 的監督，
   `X^can` 變自由變量。網絡要湊出動態物體位置時，走「強預訓練的 point 頭」比走
   「隨機初始化、梯度還被 `m<1` 縮放的 flow 頭」容易得多 → **運動被塞進 `X^can`，`Δ` 閒置 ≈ 0**。

**決定性證據**：跨 stage、跨域診斷顯示動態/靜態 `Δ` 的中位數比值只有 **0.77–1.16**（理想應 ≫ 1）。
**即使在 PointOdyssey 上 mask 已經很準（AUC ≈ 0.88），`Δ` 仍不分離** —— 證明這不是 mask 不好
（訓練問題），而是目標函數病態（**架構問題**），加資料、調權重都改變不了。

同時量到一個**獨立**的問題：mask 頭在訓練域 PO 上 AUC ≈ 0.88，跨到 Sintel 塌到 ≈ 0.45。
這是 domain gap，與雙場病態無關，後來由 v3 改用幾何殘差監督解決。

### 2.3 v1/v2 留給 v3 的東西

- §2.1 的問題診斷（三處剛性假設）原封不動被 v3 沿用。
- temporal attention 與 LayerScale γ=0 的 warm-start 機制留在程式裡（`enable_temporal`），
  v3 的 `smooth_temporal` 線就是在用它。
- **放棄的**：雙場表示、motion/flow 兩個頭、`L_reproj` 閉環、4D 全局對齊。v3 的做法是
  **不碰幾何表示**，把動態屏蔽從 loss 端搬到 attention 端。
