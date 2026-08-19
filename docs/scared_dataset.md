# SCARED — 資料規格（寫 `data/datasets/scared.py` 用）

內視鏡立體資料集，2026-08-17 由 `training/data/preprocess/scared_convert.py` 從官方原始發佈轉成本 repo 可讀的形式。這份文件寫的是**轉檔後**的規格；原始發佈的種種怪癖都已在轉檔時吸收掉，寫 loader 時不需要知道它們（但第 7 節列出仍會外洩的部分）。

規模：**35 個 keyframe、17,824 幀、48 GB**。

---

## 1. 磁碟佈局

根目錄 `data_path("train", "scared")`（= repo root 的 `data/train/scared/`）。

```
scared/
├── train/  22 keyframes, 15,568 幀
├── val/     6 keyframes,  1,705 幀
├── test/    7 keyframes,    551 幀
│   └── dataset{n}/keyframe{m}/
│       ├── image_left/{fid:06d}.png     RGB, 1280×1024 (W×H), 8-bit
│       ├── depth_left/{fid:06d}.png     深度, 1280×1024, uint16 單通道
│       ├── cam_data/
│       │   ├── intrinsics.txt           1 行: fx fy cx cy
│       │   ├── extrinsics.txt           N 行 × 12: world-to-cam 3×4, row-major
│       │   ├── frames.txt               N 行: 每一列對應的原始 frame index
│       │   ├── valid_frac.txt           N 行: 該幀有效深度像素比例
│       │   └── calibration.json         完整雙目標定 (KL/KR/DL/DR/R/T)
│       ├── rgb.mp4                      完整原始影片（見 §6）
│       └── meta.json                    來源、幀數、連續性、量化參數
├── split/                               官方原始清單（保留供回溯，loader 不需要讀）
└── dataset_1..9/                        原始發佈殘留（**不要讀**，見 §7）
```

`{fid}` 是**原始 frame index**，不是 0..N-1 的序號。`cam_data/*.txt` 的第 k 列 ⟷ `frames.txt` 的第 k 個 fid ⟷ `image_left/{fid:06d}.png`。三者順序一致且遞增。

`dataset{n}/keyframe{m}` 沒有底線，與 `split/*.txt` 的命名一致（原始目錄是有底線的 `dataset_n/keyframe_m`，兩者**不是**同一套編號，見 §7）。

---

## 2. 檔案解碼

### 影像

```python
image = read_image_cv2(f"{seq_dir}/image_left/{fid:06d}.png")   # (1024, 1280, 3) uint8 RGB
```

`read_image_cv2`（`data/dataset_util.py:616`）預設 `rgb=True`，已處理 BGR→RGB。與其他 loader 用法相同。

### 深度

```python
d16 = cv2.imread(f"{seq_dir}/depth_left/{fid:06d}.png", cv2.IMREAD_UNCHANGED)  # uint16
depth = d16.astype(np.float32) / 100.0        # 毫米
```

- **單位是毫米**，量化解析度 0.01 mm（`DEPTH_SCALE = 100`）。
- **0 代表無效**，與本 repo 其他 loader 的慣例一致。轉檔時 NaN、負值、以及 > 655.35 mm 的離群值都已寫成 0。
- 深度是**沿光軸的 Z**，不是到相機中心的距離。可直接餵給 `depth_to_world_coords_points`。
- 有效像素比例只有 **4%–73%**（中位數約 30%），這是資料本身的性質，不是轉檔造成的。

### 內參

```python
fx, fy, cx, cy = np.loadtxt(f"{seq_dir}/cam_data/intrinsics.txt")
intri = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
```

一個 keyframe 內**逐幀完全不變**（已逐幀驗證位元相同），所以只存一行。典型值 fx≈fy≈1035、cx≈597、cy≈520 —— 注意主點明顯偏離影像中心（640, 512），這是真實標定，不是錯誤。

`calibration.json` 裡的畸變係數（量級 1e-3～1e-4）**不需要套用**：影像已是去畸變後的結果，用 `KL` 直接投影 scene points 的重投影誤差中位數為 0.395 px。

### 外參

```python
E = np.loadtxt(f"{seq_dir}/cam_data/extrinsics.txt").reshape(-1, 3, 4)
extri_opencv = E[k].astype(np.float32)     # 直接就是 VGGT 要的格式
```

**已經是 world-to-cam（OpenCV），不需要任何反轉或軸置換。** 這點與 TartanAir / Spring / Waymo 相反（那三個是 c2w，loader 裡都有 `np.linalg.inv`）。35 個 keyframe 全部驗證過：把它當 w2c 解出的世界點雲在相鄰幀間質心差 0.2–6.5 mm，當 c2w 則是 5–40 mm。

---

## 3. 單位

**不要把毫米轉成公尺。** `trainer.py:979` 對每個 sample 呼叫 `normalize_camera_extrinsics_and_points_batch`，會把座標系移到第一台相機並依點距縮放到單位尺度，所以絕對單位在進入模型前就被消掉了。

**但 `depth_max` 是作用在正規化之前的原始單位上**，所以 SCARED 的 `depth_max` 必須用毫米計。建議 `depth_max: float = 655.35`（= 轉檔時的上限，等於不再額外裁切）；若要更保守，各 keyframe 的深度 p99.9 都 ≤ 165 mm，設 300–500 mm 也安全。

對照：Sintel 100、Spring 200、TartanAir/PO 1000 —— 那些是公尺。

---

## 4. `get_data()` 要回傳什麼

照 `data/datasets/tartanair.py:230-243` 的形狀。每幀先走 `BaseDataset.process_one_image()`，它會負責 resize/crop 到 518、同步更新內參、並導出 `world_points`/`cam_points`/`point_masks`。

| 欄位 | 來源 |
|---|---|
| `images` | `image_left/{fid}.png` |
| `depths` | `depth_left/{fid}.png` ÷ 100 |
| `extrinsics` | `extrinsics.txt` 第 k 列 reshape(3,4)，**原樣** |
| `intrinsics` | `intrinsics.txt` 組成 3×3 |
| `motion_mask` | **尚未有標註，見 §8** |
| `seq_name` | 建議 `"scared_" + split_name.replace("/", "_")` |

---

## 5. 取樣規則

### train / val：可以當連續影片

- train 22/22 全部連續（`fid` = 0..N-1 無缺號）。
- val 6/6 連續，但**從 fid=2 起算**，不是從 0。所以 `ids` 必須是「`frames.txt` 的索引位置」而不是 fid 本身，否則會越界。
- 兩者都可以安全使用 `common_conf.get_nearby` / `get_nearby_ids`。

### test：**不可**當影片

7 個 test keyframe 是官方跨全序列的稀疏抽樣，stride 8–38。`get_nearby_ids` 取的是**清單位置**而非時間，對 test 使用會靜默地取到時間上相距數十幀的畫面。test 應該把列出的幀當一組 multi-view 直接餵入。

### 空深度幀必須避開當 anchor

train 有 **449 幀**有效深度 < 1%，集中在：

| keyframe | 空幀 / 總幀 | median valid |
|---|---|---|
| `dataset9/keyframe3` | 169 / 953 | 9.5% |
| `dataset9/keyframe2` | 162 / 590 | 15.6% |
| `dataset9/keyframe4` | 118 / 309 | 4.3% |

這會踩到 `loss.py:181`：

```python
valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
```

它只看**每個 sample 的第一幀**，然後在 `loss.py:212` 用它過濾整個 batch element。**抽到空的 anchor 幀 → 整個 sample 的 camera loss 被丟掉**，其餘 15 幀的位姿監督一併消失，而且沒有任何警告。

轉檔時刻意沒有刪掉這些幀（會在 train 的連續序列上打洞），改為輸出 `cam_data/valid_frac.txt` 讓 loader 自己擋。建議：

```python
vf = np.loadtxt(f"{seq_dir}/cam_data/valid_frac.txt", ndmin=1)
anchor_pool = np.nonzero(vf > 0.05)[0]      # 只從足夠稠密的幀挑 anchor
```

`> 0.05` 是起點，不是定論 —— 門檻要多高取決於 `process_one_image` resize 到 518 之後還剩多少有效點，這需要實測。

---

## 6. `rgb.mp4`

每個 keyframe 都附完整的原始影片（35 支共 3.4 GB）。**loader 不需要讀它**，它是給人看的。

- train：影片長度 = 圖片數（因為存了全幀）。
- val / test：影片是完整序列，比存下來的圖片長（val 407 vs 284、test 348 vs 76）。
- 影片的幀號與圖片檔名是**同一套編號**（已驗證 `mp4 第 N 幀` 與 `{N:06d}.png` 位元相同），所以可以用檔名直接去影片裡查任一幀，包括被 split 跳過的那些。

---

## 7. 陷阱清單

**這些在轉檔後的資料上都已經不成立**，但如果你去讀 `dataset_1..9/` 的原始殘留就會全部踩到：

1. **dataset 8/9 的 keyframe 編號差一**。原始目錄是 0-based（`keyframe_0..3`），而所有 `split/*.txt` 是 1-based，所以 `split` 的 `dataset8/keyframe1` 是目錄 `dataset_8/keyframe_0`。照字面對映會載到同一位病患的**另一個** keyframe，且 frame index 會越界。轉檔輸出一律採 split 命名，此問題已消除。
2. **`scene_points*.tiff` 必須用 `tifffile` 讀**。`cv2.imread` 會做 BGR 交換，把 Z 通道變成 X —— 症狀是深度出現負值、重投影誤差 3000+ px。
3. 原始點圖是 2048×1280×3 的 XYZ，**上半 1024 列才是左視角**，下半是右相機。
4. 原始 `left_depth_map.tiff` 用 **NaN** 標無效，而 `scene_points` 用 **0** —— 兩者混用時遮罩要同時排除。
5. `dataset_2/keyframe_1` 的 `left_depth_map.tiff` 與自己的 scene points 對不上（反向比對最佳落在 ds7/kf3），該檔不可信。轉檔後的 `depth_left/` 來自 scene points，不受影響。

轉檔後仍需注意的只有一項：**`dataset_1..9/` 原始目錄還在碟上（842 GB 的 `scene_points*.tiff`）**，loader 絕對不要碰它。它會在確認訓練跑得起來之後刪除。

---

## 8. 未決：`motion_mask`

SCARED **沒有任何動態標註**，而內視鏡場景確實是動態的（組織形變、器械移動）。三個選項：

- **全零**（比照 TartanAir 當純靜態）—— 最省事，但把器械與形變當成靜態餵給 gate，等於給錯標籤。
- **RAFT flow-residual**（比照 Spring/Waymo 的 `dynamic_source="raft"`，有現成的 `data/preprocess/*_raft_dynmask.py`）—— 但 project memory 記錄 RAFT residual 在低視差場景很脆弱，而內視鏡正是典型的低視差。
- **不接 gate，只當 depth/pose 監督**（config 不給 motion_mask，`ComposedDataset` 會補零，但明確不拿 SCARED 訓 gate）。

在做出決定之前，loader 可以先支援 `dynamic_source: str = "none"` 並回傳全零，把介面留好。**注意全零不是「沒有標籤」而是「宣稱全部靜態」**，兩者對 `L_gate` 的意義完全不同 —— 如果決定不用 SCARED 訓 gate，正確做法是在 config 裡不啟用 gate loss，而不是餵全零標籤。

---

## 9. gate 在 SCARED 的實測（2026-08-18，三支工具的結論存放處）

> 這三筆是**量到的數字**，來源工具（`diag/vis/gate_logit_dist.py`、`diag/gate_offset_sweep.py`、
> `diag/scan_foreground.py`）在 2026-08-19 的整理中刪除，原始 json 留在
> `outputs/gate_logit_dist/`、`outputs/gate_offset_sweep/`、`outputs/scan_foreground/`。
> 記在這裡是為了不用重跑就知道發生過什麼。**解讀（第 4 點）尚未拍板。**

### 9.1 `inst_g` 的 gate logit 在 SCARED 全部落在梯度死區

ckpt `checkpoints/inst_g.pt`，val split，`n_frames=50`：

| 序列組 | patches | mean | 中位數 | max | `g >= 0` 佔比 |
|---|---|---|---|---|---|
| ds2/3/4 kf3（少器械）| 26640 | −6.50 | −6.25 | **−0.98** | **0.0%** |
| ds5/6/7 kf3（有器械）| 39960 | −5.55 | −5.41 | **−0.17** | **0.0%** |

`frac_dead = 1.0`：**沒有任何一個 patch 的 logit 到得了 clamp 折點（g=0）**。attention bias 是
`clamp(log2 − softplus(g), max=0)`，所以在這個 domain **bias 恆等於 0 —— gate 在 SCARED 的前向是
證明性的 no-op**（這是算術，不是推論）。梯度：clamped 版本 `mean|∂bias/∂g| = 0.0`，
拿掉 clamp 也只有 0.008 / 0.015。

> 這就是 `aggregator.gate_leaky` 與 `trainer._reset_gate_head` 存在的理由 —— 兩者都是為了把 logit
> 移出這片死區（commit `e61960a`）。

### 9.2 把 logit 整體加常數 b：b≥5 之後 ATE 單調變差

同一 ckpt，val 六條序列，`n_frames=50`（ATE 單位 mm）：

| b | 0 | 3 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| ATE | 1.7365 | 1.7348 | 1.7589 | 1.7914 | 1.8305 | 1.8853 | 2.2036 |
| `frac_active` | 0.0% | 6.0% | 35.4% | 54.2% | 70.8% | 83.1% | 95.7% |

b=3（只有 6% patch 越過折點）與 b=0 沒有差別（−0.1%，遠在噪聲內）；**b=5 起 ATE 隨 b 單調上升，
b=10 時 +26.9%**。也就是說：把 gate 的排序直接當遮罩用，遮得越多 pose 越差。

### 9.3 SCARED 大部分序列其實**有**器械 —— 與原本的假設相反

`scan_foreground.py` 用「低飽和、非暗」偵測金屬器械，掃過 train/val 全部 35 條序列
（`sat_thr=60`、`val_thr=40`、blob ≥ 2% 畫面算命中）：

- **23/35 條序列有超過一半的幀命中**，其中 11 條是 100%
- 最高：`dataset_6/keyframe_3`（blob 最大 54.8%、平均 23.2%）、`dataset_7/keyframe_4`（43.0% / 15.8%）
- 真正乾淨的只有 `dataset_1/*`、`dataset_3/{1,3,4}`（blob_max ≤ 0.021，命中 0%）

**這推翻了「SCARED 沒有獨立運動前景所以 gate 沒東西可 gate」的原假設。** 注意偵測器是啟發式的，
反光同樣是低飽和高亮度（腳本用形態學開運算濾小塊），所以這是上界而非精確器械面積。

### 9.4 待拍板的解讀

9.1（gate 前向恆為 no-op）與 9.3（場景其實有器械）合起來，排除了「gate 沒作用是因為沒東西可 gate」。
剩下兩個競爭解釋，**都還沒驗證**：

1. gate 的**排序**在內視鏡上就是錯的（在 Sintel 上學到的「動態」特徵遷不過來），9.2 的單調變差是直接證據；
2. 排序是對的但**校準**偏掉，而 9.2 的變差來自「器械其實對 pose 有用」（內視鏡低視差，遮掉任何有紋理
   的區域都會傷 pose，`docs/table.md` 的光流觀測度那條線）。

要分開這兩個，需要器械的 GT 遮罩（或人工標幾十幀）拿去算 gate 的 AUC —— 也就是 §8 仍未決的
`motion_mask` 問題。
