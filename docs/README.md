# Dyn-VGGT — 主文件

> 2026-10-08 改版：舊的 `README.md`（索引）、`status.md`（結論）、`計劃書_2026-10.md`（工作底稿）合併成這一份。
> 原檔在 [archive/](archive/)。
> **主線：在 `tfs` 上加回期中的 motion gate，並把 gate 修好。** 三個探索分支（attention bias、光照 token、時間 RoPE）已收起。

## 更新規則

- 有新結果時只做兩件事：數字寫進 [results/](results/)，§6 進度紀錄追加一行（日期、run、一句話的觀察）。
- §2「目前的結論」只在使用者確認後才改。進度紀錄寫觀察，不寫結論。
- 新 run 開跑時，先在進度紀錄寫「開跑：<run>，在哪台、預計何時完成、要回答哪條假設」。
- 待辦放 [待處理.md](待處理.md)。

---

## 1. 索引

| 文件 | 內容 |
|---|---|
| 本文件 | 結論、訓練設定、評估標準、gate 的改進目標、進度紀錄 |
| [method.md](method.md) | 機制與公式：gate、temporal、camera_smooth、`flow_geom`。不寫數字 |
| [results/scared.md](results/scared.md) | SCARED 數字：協定、對外對照、自家 arm 主表 |
| [results/c3vd.md](results/c3vd.md) | C3VD 數字 |
| [results/private.md](results/private.md) | 私人資料集（lesion / gastric） |
| [內視鏡整理.md](內視鏡整理.md) | 期中前後的內視鏡實驗整理，含 gate 在 SCARED 的實測（測試 §2） |
| [experiments.md](experiments.md) | 每個 run 的身分與下場 |
| [checkpoints.md](checkpoints.md) | 每支權重的身分 |
| [topics/scared_dataset.md](topics/scared_dataset.md) | SCARED 資料規格：轉檔佈局、單位、取樣規則、陷阱 |
| [ops/execution.md](ops/execution.md) | 訓練／恢復／評測的完整指令 |
| [report/](report/) | 期中報告（2026-08-26 凍結，不改） |
| [archive/](archive/) | 封存：舊版 status / 計劃書 / method、自然場景數字、三個分支的數字、planning、reading |

---

## 2. 目前的結論

只寫已確認的；未驗證的假設在 §5。

### 2.1 `tfs` 是現在最好的組合

| 對照 | pose c64 | pose c5 |
|---|---|---|
| 底座 `vanilla`（解凍） | 0.0711 | 0.0851 |
| `tfs` | **0.0609（−14.3%）** | **0.0709（−16.7%）** |

- 三者合起來在 pose c64、c5、多張 depth 都是解凍組最好（[results/scared.md](results/scared.md) §3）。單獨看，c64 上 t、f、s 都比底座好；c5 上 `smooth` 單獨是 0.0859，比底座差 0.9%。
- **單張 depth 分不出勝負**：同底座全部落在 0.0495–0.0502。真實的落差在兩個底座之間（凍結 0.0561 對解凍 0.0499），那是解凍 depth head 帶來的，不能歸功給任何成分。

### 2.2 對外：depth 單張追平，pose c5 還輸

- depth 單張：`tfs` 0.0498，published 最好是 EndoSfM3D 0.050。
- pose c5 逐序列：`tfs` ds5 0.0848 / ds3 0.0571，Endo-FASt3r 0.0702 / 0.0438，分別差 +20.8% / +30.4%。
- c64（0.0746 / 0.0472）沒有文獻對應，只能當補充。

### 2.3 gate 在期中的醫學線上邊際效果很小

- `b16` → `b16_gg`（唯一差別是 gate 可訓練）：c64 0.0706 → 0.0699（−1.0%），c5 0.0824 → 0.0810（−1.7%），都在 ckpt 漂移內。
- `inst_g` 的 gate 在 SCARED 上前向是 no-op：所有 patch 的 logit < 0，bias 恆為 0（[內視鏡整理.md](內視鏡整理.md) 測試 §2）。

### 2.4 速度

64 幀一條序列：Ours 7.2 s、MonST3R 450.7 s（63 倍）。gate 對推論時間量不到成本。

---

## 3. 訓練設定（`tfs`）

2026-10-08 依 `pipeline/training/config/scared_cam_depth_tfs.yaml` 的繼承鏈（`scared_cam_vanilla` → `scared_cam_depth_flowgeom_smooth` → `tfs`）
與 `logs/scared_cam_depth_tfs/{log.txt,trainable.txt}` 核對。

| 項目 | 內容 |
|---|---|
| run | `scared_cam_depth_tfs`，seed 42 |
| 起點 | `checkpoints/VGGT-1B.pt`（`strict: false`，temporal block 為新建） |
| 凍結 | `patch_embed`、`frame_blocks`、`global_blocks` 0–7 |
| 可訓練 | `global_blocks` 8–23、camera head、camera / register token、depth head、8 個 temporal block |
| 模型 | `enable_temporal: True`、`enable_gate: False`、point / track head 關閉 |
| 資料 | SCARED train 22 條序列；clip 4–12 張（`img_nums [4, 12]`）、`max_img_per_gpu 12`；影像 518 |
| 抽樣 | 錨點只從深度夠密的幀抽，在錨點附近 ±2×clip 長度內抽其餘幀，依時間排序 |
| 增強 | color jitter（亮度 / 對比 / 飽和 0.5、hue 0.1、p 0.9）、灰階、scale 0.8–1.2、aspect 0.33–1.0 |
| 優化 | AdamW lr 3e-5、weight decay 0.05；5% linear warmup（1e-8 起）+ cosine 到 1e-8 |
| 長度 | 10 epoch × 1500 步，`accum_steps 1` |
| 其他 | bf16 autocast、gradient checkpointing、每組 grad clip 1.0（global+camera head / token / depth head / temporal 四組）；flash SDP 關、TF32 開 |

loss：

| 項 | weight | 設定 |
|---|---|---|
| camera | 5.0 | L1，γ 0.6 |
| depth | 1.0 | gradient loss `grad`、`valid_range 0.98`、**α 0.02** |
| camera_smooth | 3.0 | 平移 1.0、旋轉 1.0、γ 0.6，2 階 |
| `flow_geom` | 1.0 | 見 §3.1 |

### 3.1 `flow_geom` 的設定

機制與公式見 [method.md](method.md) §5。

| 參數 | 值 | 意思 |
|---|---|---|
| `src_offsets` | [−1, 1] | 每張和 clip 裡前後相鄰的那張配對 |
| `flow_proc` | 256 | RAFT 在 256 解析度上算光流 |
| `occ_thresh` | 0.95 | 反向光流覆蓋率 > 0.95 的像素才算可見 |
| `min_valid` | 100 | 一對裡可見像素 < 100 就跳過 |

### 3.2 加 gate 時要決定的設定

`tfs` 沒有 gate。期中 `b16_gg` / `smooth_temporal` 的 gate 設定是：從 `inst_g` 接續、`reset_gate_head: True`、`gate_pose_grad: True`、`gate_leaky: 0.1`、`gate_block_iter 7`，沒有 `L_gate`。
`tfs` 從 VGGT-1B 起訓，沒有 `inst_g` 的 gate predictor 可以接，這是加回 gate 時第一個要決定的事。

---

## 4. 評估標準

### 4.1 協定與指令

| 角色 | 協定 |
|---|---|
| 主指標 | pose c64、depth 多張 |
| 參考（對外比較用） | pose c5、depth 單張 |

- 探索期用 **test 當 validation**（2026-10-05 決定）。
- **訓練中**：trainer 每個 epoch 評主指標（`pose_eval.dataset: scared_primary`），結果在 `logs/<exp>/pose_eval/epoch_N/`，`best_ate.pt` 依 test c64 挑。
- **訓練後**：另外跑參考指標。

```bash
# <ck> = logs/<exp>/ckpts/last.pt 或 best_ate.pt；舊 run 用 epoch_10.pt
# pose c64（主指標）→ pose_c64_*
python pipeline/benchmark/eval_scared.py --ckpts <ck> --split test/pose --n_frames 0 --chunk_size 64 --overlap 16 --no_depth
# pose c5（參考）→ pose_*
python pipeline/benchmark/eval_scared.py --ckpts <ck> --split test/pose --n_frames 0 --chunk_size 5 --overlap 4 --no_depth
# depth 單張（參考）→ depth_*
python pipeline/benchmark/eval_scared.py --ckpts <ck> --split test/depth --n_frames 0 --single_view --depth_protocol afsfm
# depth 多張（主指標）→ depth_multi_*
python pipeline/benchmark/eval_scared.py --ckpts <ck> --split test/depth --n_frames 0 --chunk_size 64 --overlap 16 --depth_protocol afsfm
```

- 結果寫到 `outputs/eval/<exp>/<協定>_<last|best>/results.json`；其他 ckpt 或非標準參數要自己帶 `--out_dir`。
- 用 `n_chunks` 確認協定：c64 是 18 / 9，c5 是 830 / 407。
- 用 `failed_seqs` 確認沒有序列失敗；失敗時平均值是 NaN，不要自己重算後當正常結果用。

### 4.2 比較與報告

- **對照組是 `tfs`**，報相對 `tfs` 的百分比。順序：c64 → 多張 depth → c5 → 單張 depth。
- **results 表逐格取 `epoch_10` 與 `best_ate` 較好的值**（2026-10-08 決定）。
  ⚠️ 判斷 2% 門檻時，**兩邊要用同一種 ckpt**：`tfs` 的 `epoch_10` 是 0.0609 / 0.0413 / 0.0709 / 0.0498，逐格最好是 0.0609 / 0.0409 / 0.0709 / 0.0498。
- pose 一律附逐序列值（ds3 / ds5），看兩條是否同方向。
- 擺動參考量：四個指標統一 **2%**。< 2% 不算贏；≥ 2% 只算「勉強能用」。
- 全部單一 seed；`tfs` 的多張 depth seed 擺動有 6.5%（[results/scared.md](results/scared.md) §5）。
- 新 run 命名：`scared_tfs_<分支>_<變體>`；gate 用 `g`，例如 `scared_tfs_g_leaky`。

---

## 5. gate 的改進目標

**目標**：讓 gate 在 `tfs` 上真的改善 pose。回頭條件使用者沒有設（2026-10-08）。

### 5.1 已知的毛病

| # | 毛病 | 證據 | 出處 |
|---|---|---|---|
| P1 | **bias 死區**：預設 bias 的 kink 在 σ(g)=0.5，g<0 的 patch bias 和梯度都是 0 | `inst_g` 在 SCARED 所有 patch 的 logit < 0（max −0.17） | [內視鏡整理.md](內視鏡整理.md) 測試 §2 |
| P2 | **soft bias 沒兌現**（自然場景）：同一份 g，只換「怎麼變成 bias」，效果差很多 | Sintel ATE：`predicted` −0.15%、二值化 −4.26%、每幀前 20% −4.76%、oracle −5.85% | [archive/status.md](archive/status.md) A.2 |
| P3 | **沒有動態標籤**：SCARED 只能靠 camera loss 訓練 gate | 全零標籤等於宣稱全靜態；RAFT 在低視差不可靠 | [內視鏡整理.md](內視鏡整理.md) 測試 §2 |
| P4 | **遮得越多 pose 越差**：把 logit 整體加常數 b，b ≥ 5 後 ATE 單調變差 | b=10 時 +26.9%（舊協定 val 全序列 ATE） | [內視鏡整理.md](內視鏡整理.md) 測試 §2 |

### 5.2 假設

| # | 假設 | 現有證據 | 狀態 |
|---|---|---|---|
| G1 | gate 在 SCARED 沒作用，是因為場景裡沒有東西可遮 | 35 條序列裡有 23 條、過半的幀偵測到器械 | **已否定** |
| G2 | gate 的排序在內視鏡上是錯的（Sintel 學到的「動態」遷不過來） | P4 的單調變差 | 未驗證 |
| G3 | 排序是對的，但器械對 pose 其實有用（低視差，遮掉有紋理的區域就傷 pose） | P4 的單調變差也符合 | 未驗證 |
| G4 | 修掉死區（P1）就能讓 gate 兌現 | 自然場景 `gate_leaky=1.0` 只回收 −1.79%、`zero_ref` −1.10%（P2 的 oracle 是 −5.85%） | 有線索 |

- 要分開 G2 和 G3，需要器械的 GT 遮罩（或人工標幾十幀）來算 gate 的 AUC，也就是 P3。

### 5.3 下一步

待使用者決定第一步（2026-10-08 提出、未回答）：

- A 先診斷現有 gate：用期中 ckpt 只跑推論，量關掉 gate 時 pose 變多少、σ(g) 落在死區的比例
- B 直接改 bias 函數（二值化、前 k%、`gate_bias_scale` 參數化）
- C 先替 SCARED 造動態標籤

---

## 6. 進度紀錄

- 2026-10-08 主線改為「在 `tfs` 上修好 gate」；三個分支收起，數字封存在 [archive/results/](archive/results/)。
- 2026-10-08 文件整理：README / status / 計劃書合併成本文件；SCARED 數字合併成 [results/scared.md](results/scared.md)，C3VD、私人資料集各自獨立。
- 2026-10-08 讀期中訓練 log 的 gate 統計（train 的 running 平均，最後一步）：`b16_gg` `gate_g_median` −0.0688（平均 −0.0764）、`gate_frac_active` 0.0005（平均 0.0034）；`smooth_temporal` `gate_g_median` 0.0139（平均 0.0254）、`gate_frac_active` 0.6068（平均 0.6894）。`gate_frac_active` 是 g > 0 的 patch 比例。來源 `logs/scared_cam_b16_gg{,_smooth_temporal}/log.txt`。
