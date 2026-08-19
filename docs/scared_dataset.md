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
