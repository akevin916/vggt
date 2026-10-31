# 方向 2：StereoMIS 上重做步驟 0（遮罩頭空測試），加器械 mask 與 RAFT 遮罩對照（2026-10-14 第 2 版，待 Kevin 審）

目的：SCARED 上遮了沒差（步驟 0），分不出是「資料本來沒東西可遮」還是方法或標籤的問題。
StereoMIS 有器械動作與組織變形，在這裡重做同一套測試：如果這裡遮了有用，問題出在 SCARED 的資料；如果這裡也沒用，就不是資料的問題。

這份只做**評估**（現有 ckpt，不訓練）。訓練要等這份的結果，另寫規格。

背景：[研究背景.md](../研究背景.md)（Q1、Q2、Q5）。SCARED 版：[步驟0_遮罩網格.md](步驟0_遮罩網格.md)。遮罩的程式（`aggregator.py` 的 M1/M2/M3、temporal、`eval_scared.py` 的 `--mask_*` 參數）**沿用步驟 0 已實作的版本，不重寫**。

第 2 版的改動（依 Kevin 10-14 回報的前處理）：資料格式與 split 改成實際轉好的版本；原始資料有器械 mask，加進來當第三種標籤；c5 改成只在 c64 有跡象時才跑。

---

## 1. 資料（Kevin 已在另一台電腦轉好）

### 1.1 已轉好的內容

| 項目 | 內容 |
|---|---|
| 目錄 | `<stereomis>/{train,test}/<seq>/`，底下 `image_left/{fid:06d}.png`、`mask_left/{fid:06d}.png`、`cam_data/{frames.txt, extrinsics.txt, intrinsics.txt}`；原始 `rgb.mp4`、`groundtruth.txt`、`StereoCalibration.ini` 以 hard link 留在同資料夾 |
| 影像 | 左眼（原始 1280×2048 的上半），縮成 640×512，**沒有 rectify、沒有 undistort** |
| fid | 1 起算的影片幀號 |
| pose | `groundtruth.txt` 第 fid−1−4 列（offset −4），c2w 公尺 → world→cam |
| 內參 | 左眼標定縮一半，整段一行（例：fx 519.8、cx 298.0） |
| 器械 mask | 原始資料附的 `masks/*l.png`，640×512；255 = 可用，0 = 器械或整幀不可用 |
| 哪些幀 | 原始有 mask 的幀；test 每幀都有 |
| split | train：P1、P2_0–P2_5；test：P2_6–P2_8 |
| 沒有 | depth、`valid_frac.txt`、右眼 PNG |

offset −4 的依據：Kevin 的極線掃描，−4 的 δ 中位數 1.640 px，比 0 小 21%；最低點是 −3（1.600 px），兩者只差 2.4%。這次照原作者用 −4。

### 1.2 這次用的評估片段（事前定，不挑）

只用 test（P2_6、P2_7、P2_8）。每條序列切成數段連續 512 幀，不抽幀：

- 段數 K = min(4, ⌊N / 512⌋)，N 是該序列 `frames.txt` 的幀數。
- 第 k 段起點 = round(k · (N − 512) / (K − 1))，k = 0 … K−1（K = 1 時取正中間）。N ≥ K × 512，所以各段不重疊。
- 實際幀數 N：P2_6 11855、P2_7 13068、P2_8 19048（2026-10-14 在 75 上核對），三條都切 4 段，共 12 段、6144 幀。

理由：SCARED test/pose 是 411、834 幀的連續片段；512 幀在 c64 下約 10 個 chunk。整條序列直接跑的話 c5 的窗數太多。

### 1.3 片段目錄（scratchpad script `make_clips.py`）

`eval_scared.py` 讀 `<root>/<split>/<seq>/`，需要 `frames.txt`、`extrinsics.txt`、`valid_frac.txt` 和 `image_left/`。不改 repo 程式，改用片段目錄：

（2026-10-14 修訂：`--split` 只接受 `val`／`train`／`test/depth`／`test/pose`，所以片段放在 `test/pose` 底下、eval 用 `--split test/pose`；遮罩檔對應放在 `<mask_dir>/test_pose/<seq>_c<k>.npz`。）

```
<clips>/test/pose/<seq>_c<k>/
    image_left -> <stereomis>/test/<seq>/image_left    （symlink）
    mask_left  -> <stereomis>/test/<seq>/mask_left     （symlink）
    cam_data/frames.txt        該段的 512 個 fid
    cam_data/extrinsics.txt    對應的 512 列
    cam_data/intrinsics.txt    複製
    cam_data/valid_frac.txt    只有 header（同 SCARED test/pose，load_sequence 讀到空檔會填 NaN）
```

`<clips>` = `data_path("train", "stereomis_clips")`。因為只有一層 `<seq>`，`list_sequences` 不適用，eval 時一律用 `--seqs` 列出片段名。

### 1.4 片段檢查（不過就停）

- 每段 512 個 fid 連續、512 列 pose、沒有 NaN。
- 每段相鄰幀平移中位數 > 0。
- 記下每段「整幀不可用」的幀數（mask 中 0 佔 ≥ 90% 的幀），只記錄。

---

## 2. 標籤（四種，都 pool 到 patch 格）

一次性 script，放 scratchpad。輸出格式和步驟 0 §2.3 相同：`<split>/<seq>.npz` 含 `fids [S]`、`mask [S,H_p,W_p] bool`，另外依種類附 `delta`／`visible`；`stats.json` 記門檻與標 1 比例。
patch 格大小、影像縮放與裁切用 `compute_preprocess_meta` 的實際值（640×512 和 SCARED 1280×1024 同比例，預期一樣是 37×30）。內參從 640×512 的 `intrinsics.txt` 依同一組 meta 縮放。

### 2.1 極線標籤（= SCARED 步驟 0 §2.1）

配對 i±k（平移 ≥ 3× 序列中位數）、GT pose → F、RAFT 對應點、δ = 點到極線距離、兩鄰居取最大、pool、τ = 中位數 + 3×MAD。配對只在同一段 512 幀內找。
輸出 `output_dir_for_exp("stereomis_epipolar", "motion_mask")`。

### 2.2 器械 mask（新增）

原始資料附的 mask，0 = 器械或整幀不可用。

- 縮放與裁切和影像相同（nearest）。patch 內 0 的像素 ≥ 50% 標 1（和正控制 GT 遮罩的規則相同）。
- 「整幀不可用」的幀（0 佔 ≥ 90%）：該幀全部標 0，不遮整幀，並記錄幀數。

這是目前最接近「真的在動的東西」的標籤，但只涵蓋器械，不涵蓋組織變形；mask 的來源（人工標還是模型預測）資料集沒寫，視為近似。
輸出 `output_dir_for_exp("stereomis_inst", "motion_mask")`。

### 2.3 RAFT 遮罩（Kevin 10-14 要求的對照）

定義同 repo 的 `m_raft`（`compare_dynmask.raft_gap`）：**殘差 = ‖RAFT 光流 − GT pose 加深度算出的 ego flow‖**。沒有 GT 深度，用立體深度，右眼從 hard link 的 `rgb.mp4` 取：

1. **取右眼**：影片第 fid−1 幀的下半，縮成 640×512。
2. **rectify**：`StereoCalibration.ini` 的內參縮一半，`cv2.stereoRectify(alpha=0)` 得 R1、P1，左右 remap。
3. **視差 → 深度**：RAFT 跑 rectify 後的左→右，d = −flow_x；|flow_y| > 1 px 或 d < 0.5 px 記無效。
4. **搬回原本的左眼影像**（影像與 pose 都沒 rectify）：每個左眼像素用 `cv2.undistortPoints(R=R1, P=P1)` 找到它在 rectify 影像的位置，雙線性取 d，算出 rectify 座標系的 3D 點，乘 R1ᵀ 回到左眼座標系，取 z 當深度。
5. **ego flow 與殘差**：`motion_mask.compute_ego_flow(depth_i, K, ext_i, ext_j)`，配對同 2.1；r = ‖RAFT(i→j) − ego‖，只算可見且深度有效的像素，兩鄰居取最大。
6. **pool 與門檻**：有效像素 < 30% 的 patch 記 0；τ = 中位數 + 3×MAD。

輸出 `output_dir_for_exp("stereomis_raft", "motion_mask")`。

### 2.4 隨機對照（三組）

每一幀隨機選和來源標籤**同樣個數**的 patch，seed 0。`stereomis_rand_epi`、`stereomis_rand_inst`、`stereomis_rand_raft`。

### 2.5 紀錄（只記錄，不停；Kevin 10-14 決定）

`stats.json` 逐段記：

- 極線 δ 中位數；RANSAC 基準（同一組 RAFT 對應點用 RANSAC 估 F，不用 GT pose）的 δ 中位數，以及比值 δ_GT ÷ δ_RANSAC。比值遠大於 1 代表 pose 那側可能有問題。
- RAFT 遮罩的殘差 r 中位數。
- 三種標籤各自的標 1 比例；整幀不可用的幀數。
- **動態量**：δ > τ_SCARED（`outputs/_archive/legacy_2026-10-20/motion_mask/scared_epipolar/test_pose/stats.json` 的 τ，絕對 px）的 patch 比例，和 SCARED ds3 7.3%、ds5 20.5% 並列。
- **標籤對器械的命中**：以器械 mask 為準，極線、RAFT 各自的 recall（器械 patch 被標到的比例）與 precision（標 1 的 patch 落在器械上的比例）。組織變形不在器械 mask 裡，所以 precision 偏低是預期的。

這些數字在判讀時和頭空結果並列，不改變 §6 的判讀規則。

---

## 3. 實驗網格

ckpt：VGGT-1B、`tfs`（`logs/scared_cam_depth_tfs/ckpts/epoch_10.pt`）。兩個在 StereoMIS 上都是 zero-shot。不跑 depth。

| # | 標籤 | 遮法 | 起始層 | temporal | 為什麼 |
|---|---|---|---|---|---|
| 1 | — | off | — | — | 基準 |
| 2 | 極線 | M1 | 8 | — | 和 SCARED 步驟 0 同設定 |
| 3 | 極線 | M2 | 8 | — | 同上 |
| 4 | 極線 | M3 | 8 | — | 同上 |
| 5 | 極線 | M2 | 0 | — | 正控制最好的兩種（救回 0.74 / 0.80） |
| 6 | 極線 | M3 | 0 | — | 同上 |
| 7 | 器械 | M2 | 0 | — | 最接近 GT 的標籤 |
| 8 | 器械 | M3 | 0 | — | 同上 |
| 9 | RAFT | M2 | 0 | — | RAFT 對照 |
| 10 | RAFT | M3 | 0 | — | 同上 |
| 11 | 隨機（對齊極線） | M3 | 0 | — | 排除「遮什麼都有用」 |
| 12 | 隨機（對齊器械） | M3 | 0 | — | 同上 |
| 13 | 隨機（對齊 RAFT） | M3 | 0 | — | 同上 |
| 14 | 極線 | M3 | 0 | on | 只有 `tfs`（VGGT-1B 沒有 temporal block） |
| 15 | 隨機（對齊極線） | M2 | 0 | — | M2 組別自己的隨機對照（2026-10-14 補，Kevin 選 A） |
| 16 | 隨機（對齊器械） | M2 | 0 | — | 同上 |
| 17 | 隨機（對齊 RAFT） | M2 | 0 | — | 同上 |

VGGT-1B 16 組、`tfs` 17 組。

**協定分兩階段**（第 2 版改動，省 c5 的時間）：
1. 全部組別跑 **c64**（chunk 64 / overlap 16）。
2. **c5**（chunk 5 / overlap 4）只跑：#1 off、#11–13 隨機對照，以及 c64 ≤ −2% 的組別；某個 M2 組別進 c5 時，它對應的 #15–17 也跑 c5。

**全 0 檢查**：不跑（Kevin 10-14 決定）。c64 的全 0 檢查在 SCARED 步驟 0 已通過，程式相同；c5 從未做過全 0 檢查，見 §7。

---

## 4. 要改的程式

- repo **不用改**：資料已轉好；`eval_scared.py`、`aggregator.py` 沿用步驟 0 的版本。
- scratchpad：`make_clips.py`（§1.3）、`epipolar_mask.py`（步驟 0 的版本加 `--root`／`--split`）、`inst_mask.py`、`raft_mask.py`、`rand_mask.py`。
- 輸出位置：`eval_scared.py` 預設的輸出路徑只看 ckpt 和協定，會和 SCARED 撞名，所以每次都給 `--out_dir`，用 `output_dir_for_exp("stereomis_step0", "eval")/<組別>`。

---

## 5. 執行順序

```bash
# 0. 片段（§1.3–1.4，不過就停）
python <scratchpad>/make_clips.py --src <stereomis> --out <clips>
SEQS=$(ls <clips>/test/pose)                       # P2_6_c0 P2_6_c1 P2_7_c0 …

# 1. 標籤（§2.5 只記錄，不停）
python <scratchpad>/epipolar_mask.py --root <clips> --split test/pose --out outputs/_archive/legacy_2026-10-20/motion_mask/stereomis_epipolar
python <scratchpad>/inst_mask.py     --root <clips> --split test/pose --out outputs/_archive/legacy_2026-10-20/motion_mask/stereomis_inst
python <scratchpad>/raft_mask.py     --root <clips> --split test/pose --out outputs/_archive/legacy_2026-10-20/motion_mask/stereomis_raft
for L in epipolar inst raft; do
  python <scratchpad>/rand_mask.py --like outputs/_archive/legacy_2026-10-20/motion_mask/stereomis_$L --seed 0 --out outputs/_archive/legacy_2026-10-20/motion_mask/stereomis_rand_$L
done

# 2. 網格：先全部 c64
OUT=outputs/_archive/legacy_2026-10-20/eval/stereomis_step0      # = output_dir_for_exp("stereomis_step0", "eval")
COMMON="--scared_root <clips> --split test/pose --seqs $SEQS --n_frames 0 --no_depth"
run () {  # $1 協定 c64|c5  $2 組名  $3 ckpt  其餘：遮罩參數
  p=$1; name=$2; ck=$3; shift 3
  [[ $p == c64 ]] && CO="--chunk_size 64 --overlap 16" || CO="--chunk_size 5 --overlap 4"
  python pipeline/benchmark/eval_scared.py --ckpts $ck $COMMON $CO --out_dir $OUT/$name/$p "$@"
}
M=outputs/_archive/legacy_2026-10-20/motion_mask
for CK in checkpoints/VGGT-1B.pt logs/scared_cam_depth_tfs/ckpts/epoch_10.pt; do
  t=$(basename $(dirname $(dirname $CK))); [[ $CK == *VGGT-1B* ]] && t=vggt1b
  run c64 $t/01_off         $CK
  run c64 $t/02_epi_M1_s8   $CK --mask_dir $M/stereomis_epipolar --mask_mode camera --mask_start 8
  run c64 $t/03_epi_M2_s8   $CK --mask_dir $M/stereomis_epipolar --mask_mode all    --mask_start 8
  run c64 $t/04_epi_M3_s8   $CK --mask_dir $M/stereomis_epipolar --mask_mode cross  --mask_start 8
  run c64 $t/05_epi_M2_s0   $CK --mask_dir $M/stereomis_epipolar --mask_mode all    --mask_start 0
  run c64 $t/06_epi_M3_s0   $CK --mask_dir $M/stereomis_epipolar --mask_mode cross  --mask_start 0
  run c64 $t/07_inst_M2_s0  $CK --mask_dir $M/stereomis_inst     --mask_mode all    --mask_start 0
  run c64 $t/08_inst_M3_s0  $CK --mask_dir $M/stereomis_inst     --mask_mode cross  --mask_start 0
  run c64 $t/09_raft_M2_s0  $CK --mask_dir $M/stereomis_raft     --mask_mode all    --mask_start 0
  run c64 $t/10_raft_M3_s0  $CK --mask_dir $M/stereomis_raft     --mask_mode cross  --mask_start 0
  run c64 $t/11_rndE_M3_s0  $CK --mask_dir $M/stereomis_rand_epipolar --mask_mode cross --mask_start 0
  run c64 $t/12_rndI_M3_s0  $CK --mask_dir $M/stereomis_rand_inst     --mask_mode cross --mask_start 0
  run c64 $t/13_rndR_M3_s0  $CK --mask_dir $M/stereomis_rand_raft     --mask_mode cross --mask_start 0
  [[ $t != vggt1b ]] && run c64 $t/14_epi_M3T_s0 $CK --mask_dir $M/stereomis_epipolar --mask_mode cross --mask_temporal --mask_start 0
  run c64 $t/15_rndE_M2_s0  $CK --mask_dir $M/stereomis_rand_epipolar --mask_mode all --mask_start 0
  run c64 $t/16_rndI_M2_s0  $CK --mask_dir $M/stereomis_rand_inst     --mask_mode all --mask_start 0
  run c64 $t/17_rndR_M2_s0  $CK --mask_dir $M/stereomis_rand_raft     --mask_mode all --mask_start 0
done

# 3. c5：#1、#11–13，加上 c64 相對 off ≤ −2% 的組別（同樣的 run，協定換 c5）
```

建議機器：76（75 在跑 P3）。

---

## 6. 判讀（事前定，跑之前不再改）

每個 ckpt 各自以自己的 #1 off 為基準，算相對變化。所有片段都算。

1. **頭空**（對 #2–#10、#14 每一組）：
   - c64 > −2% → **沒有頭空**（不跑 c5）。
   - c64 與 c5 都 ≤ −4%，且 ≥ 70% 的片段同方向、3 條序列各自的平均也同方向，且對應的隨機對照（同遮法、同標籤來源：M3 對 #11–13，M2 對 #15–17）c64 > −2% → **有頭空**。
   - 其他 → 回報討論。
   - 隨機對照本身 ≤ −2%：增益來自「遮掉任何東西」，不算標籤的功勞，另外標出。
2. **標籤之間**（同遮法、同起始層比較，例如 #6 vs #8 vs #10）：c64 與 c5 差都 > 2% 且同方向才算分出高下；否則視為相當。
3. **temporal**（#14 vs #6，只有 `tfs`）：差 < 2% 視為 temporal 不是主要來源，和 SCARED 結論一致。
4. **和 SCARED 的對照**（#2–#4 vs SCARED 步驟 0 同格）：只並列，不判讀。

**結論怎麼對應到下一步**（看 M2／M3 s0 的組別）：

| 器械 mask 頭空 | 極線頭空 | 解讀 | 下一步 |
|---|---|---|---|
| 有 | 有 | SCARED 沒效是因為資料沒東西可遮，極線標籤也夠用 | 寫 StereoMIS 訓練規格（train split、只用 camera loss） |
| 有 | 沒有 | 遮器械有用，但極線標不到 | 改標籤（看 RAFT 的結果與 §2.5 的命中率） |
| 沒有 | 有 | 該遮的是器械以外的東西（例如組織變形） | 回報討論 |
| 沒有 | 沒有 | 遮了沒用；若 §2.5 動態量明顯比 SCARED 多，問題就不在資料 | 回報討論是否收斂到 `tfs` |

「動態量明顯比 SCARED 多」：所有片段 δ > τ_SCARED 比例的中位數 > 20.5%（SCARED 兩條中較高的那條）。

---

## 7. 限制

- 沒有 GT 深度：只能判讀 pose。
- pose 是運動學，offset −4 是採用不是確定（−3 只差 2.4%）。§2.5 的 RANSAC 比值偏大的片段，標籤可能大多是雜訊。
- 器械 mask 只涵蓋器械、來源未知；整幀不可用的幀不遮。
- RAFT 遮罩靠立體深度，器械邊緣、反光、煙霧的深度最不準；影像沒 undistort（畸變約 0.004），rectify 與搬回時都忽略畸變。
- test 只有 P2_6–P2_8 三條，片段之間不獨立（同序列的片段相關），所以判讀另外要求 3 條序列各自同方向。
- c5 的全 0 檢查只在步驟 0b 做過一次（M3、起始層 0，0.0936 對不遮 0.0933，過）；M2 在 c5、M3 起始層 8 在 c5 沒做過。
- 兩個 ckpt 都沒看過 StereoMIS；判讀只看同一 ckpt 內開／關的相對差。
- 片段固定位置切，不保證每段都有器械動作；§2.5 的命中率與動態量會說明實際內容。

---

## 8. 成本

- 標籤：每幀 RAFT 2 個鄰居 + 立體 1 次，約 5000 × 3 次 RAFT；器械與隨機標籤不用 GPU。
- eval：c64 33 次；c5 至少 8 次（兩個 ckpt 各 #1、#11–13），其餘看 c64。c5 每次約 5000 個窗，是最慢的部分。
- repo 程式不改；scratchpad 五支 script。
