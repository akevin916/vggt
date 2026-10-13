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
- 2026-10-10 config 與 diag 整理：`pipeline/training/config/` 留 20 支（`tfs` 繼承鏈、期中線、解凍底座成分消融、其他機器用的、C3VD 兩支），其餘 107 支搬到 `archive/2026-10-10/config/`；diag 搬走 6 支反光／照明線的工具，根目錄兩支一次性 `eval_scared_*.py` 也搬走。留下的 20 支 config 全部能 compose。
- 2026-10-10 log 整理：46 個 smoke／`aborted` 項目、`scared_cam_depth_primary_check`（27 GB）、`inst_gtsp_buginit`（8 GB）搬到待刪區 `logs/_trash_2026-10-10/` 與資料碟 `vggt_logs/_trash_2026-10-10/`；使用者已於 2026-10-12 前刪除。其餘權重都保留。
- 2026-10-12 RAFT mask 探測（SCARED test/pose 兩條完整序列 + Hamlyn 22 條各取中間 64 幀，間隔 5 幀、門檻 0.3）：用 `tfs` 預測幾何扣相機運動，動態比例 SCARED ds3 10.0%、ds5 10.0%，Hamlyn 22 條中 11 條在 2–40%、4 條在 40–60%、7 條 > 60%；用 VGGT-1B 幾何則 Hamlyn 16 條 > 60%。純 RAFT 版（扣一個 homography）：SCARED 1.6% / 1.7%，Hamlyn `05` 68.0%、`04` 58.3%。結果在 `outputs/gate/raft_mask/`、`outputs/gate/raft_only/`。使用者看 gif 後判斷：`rectified05` 確實整個畫面都在動。
- 2026-10-12 把 RAFT mask 當 `gate_logits_override` 放回同一個模型（SCARED test pose c64，動態 patch 給 +30、靜態給 −30；mask 來自各模型自己的預測幾何，間隔 5 幀、門檻 0.3）：VGGT-1B 不遮 0.1209、照 mask 遮 0.1213（+0.3%）、位置打亂 0.1207（−0.1%）；`tfs` 不遮 0.0610、照 mask 遮 0.0610（+0.05%）、位置打亂 0.0611（+0.2%）。被遮的 patch 比例：VGGT-1B ds3 32.8%、ds5 9.5%；`tfs` 11.2%、11.3%。結果在 `outputs/gate/mask_override/summary.json`。
- 2026-10-12 PnP 假標準答案（GT 深度 + RAFT 相鄰幀 + PnP-RANSAC 串接）在 SCARED val 6 條的驗證，snippet ATE 平均：假答案對 GT 0.0620；`tfs` 對 GT 0.0582、對假答案 0.0517；VGGT-1B 對 GT 0.1088、對假答案 0.0975。6 條序列上兩個模型的排序在兩種答案下都相同。Hamlyn 內參核對（6 條抽查，用左右眼視差反推焦距）：`01`、`08` 與內參檔相符（差 < 0.5%），`14`、`18`、`22` 偏低 10–12%，`26` 偏低 22%。結果在 `outputs/gate/pnp_pseudo_gt/validation.json`。
- 2026-10-12 隨機遮 patch（只遮 camera／register 在 global attention 看 patch 這條路，24 個 global block 全套用；SCARED test pose c64 平均）：VGGT-1B 遮 0/10/30/50/70/90/100% → 0.1209 / 0.1206 / 0.1199 / 0.1194 / 0.1192 / 0.1190 / 0.1297；`tfs` → 0.0610 / 0.0610 / 0.0609 / 0.0611 / 0.0612 / 0.0618 / 0.0682。結果在 `outputs/gate/random_mask_sweep/summary.json`。
- 2026-10-12 RAFT mask（預測幾何，間隔 5 幀、門檻 0.3）落在有／沒有 GT 深度區域的比例，SCARED val 6 條：`tfs` 在有 GT 區域被標成動態的比例平均 9.7%、沒有 GT 區域 9.7%，逐序列的比值 0.82–1.32；VGGT-1B 逐序列比值 0.64–1.17。沒有 GT 深度的區域佔可用像素的 64–80%。結果在 `outputs/gate/mask_vs_gtvalid/summary.json`。
- 2026-10-12 分層把 patch 從 attention 的 key 藏起來（推論時攔截，repo 程式未改；SCARED test pose c64 平均；mask = RAFT 門檻 0.3、間隔 5 幀，shuf = 同數量位置打亂）。VGGT-1B：不藏 0.1209；只藏 global（對所有 token）mask 0.1227 / shuf 0.1211；再加 frame + temporal mask 0.1660 / shuf 0.1392；再加 DINOv2 mask 0.1631 / shuf 0.1465。`tfs`：不藏 0.0609；global mask 0.0617 / shuf 0.0617；加 frame + temporal mask 0.1273 / shuf 0.0880；加 DINOv2 mask 0.1263 / shuf 0.0905。結果在 `outputs/gate/hide_routes/summary.json`。
- 2026-10-12 形狀相同的對照（藏 global + frame + temporal，把 RAFT mask 翻轉後再藏；SCARED test pose c64 平均）：VGGT-1B 原 mask 0.1660、左右翻轉 0.1589、旋轉 180° 0.1609；`tfs` 原 mask 0.1273、左右翻轉 0.1251、旋轉 180° 0.1271。結果在 `outputs/gate/hide_routes/summary.json`。
- 2026-10-12 期中 gate 的排序放大（`smooth_temporal` best_ate，SCARED test pose c64 平均；每幀 logit 扣中位數後乘倍數，套用位置與訓練時相同）：自己的 gate 0.0597、關掉 0.0596、×10 0.0597（打亂 0.0598）、×30 0.0598（打亂 0.0597）、×100 0.0597（打亂 0.0598）。gate 原始輸出每幀內 logit 標準差 0.043、中位數 0.025。結果在 `outputs/gate/gate_amplify/summary.json`。
- 2026-10-13 極線 oracle 遮罩網格（步驟 0，pose 部分；SCARED test pose，snippet ATE 平均，c64 / c5；遮罩只在推論時套用、起始 aa-block 8）。標籤：GT pose 的極線 + RAFT 對應點的距離 δ，test/pose 的 δ 中位數 1.40 px、門檻 τ 3.52 px，標 1 比例 ds3 7.3%、ds5 20.5%。VGGT-1B：不遮 0.1209 / 0.1421；M1（只有 camera／register 不看）0.1208 / 0.1419；M2（所有 token 不看，實作為拿掉 key）0.1222 / 0.1468；M3（所有 token 不看其他幀的）0.1213 / 0.1528。`tfs`：不遮 0.0609 / 0.0709；只遮 temporal 0.0621 / 0.0715；M1 0.0611 / 0.0711；M1+temporal 0.0622 / 0.0716；M2 0.0606 / 0.0744；M2+temporal 0.0618 / 0.0747；M3 0.0608 / 0.0910；M3+temporal 0.0621 / 0.0914。遮罩全 0 時 c64 與不遮的差：`tfs` 0.0608–0.0609、VGGT-1B 0.1209–0.1212。未做：M3 在 c5 的全 0 檢查（M3 是唯一走 flex kernel 的遮法）；depth 多張（test/depth 的標籤 δ 中位數 4.12 px，未過 < 2 px 的檢查，沒有產出遮罩）。結果在 `outputs/mask_grid/summary.json`，標籤與圖在 `outputs/motion_mask/scared_epipolar/`。
- 2026-10-13 人造干擾物的正控制（步驟 0b；`tfs`，SCARED test pose，snippet ATE 平均；貼片 156×126 px 取自 train `dataset9/keyframe1`，佔 patch 的 8.9–9.0%）。c64：原始 0.0609；貼不動的干擾物 0.0698；貼斜向每幀 3 px 移動的干擾物 0.0686。在不動的干擾物上用 GT 遮罩（起始 aa-block 0 / 8 / 16）：M1 0.0700 / 0.0702 / 0.0699；M2 0.0632 / 0.0633 / 0.0706；M3 0.0627 / 0.0628 / 0.0701；M3+temporal 0.0629 / 0.0632 / 0.0703。移動的干擾物、M3 起始 0：GT 遮罩 0.0617、極線標籤 0.0631；極線標籤對貼片的 recall 不動 0.209、移動 0.783（ds3 0.741、ds5 0.867）。原始影像、M3 起始 0：隨機遮罩 3 個 seed 0.0606 / 0.0612 / 0.0609，極線標籤 0.0611。c5（不動的干擾物、M3 起始 0）：原始 0.0709、貼了不遮 0.0933、遮罩全 0 0.0936、GT 遮罩 0.0825。結果在 `outputs/pos_control/summary.json`，計畫在 `docs/plan/步驟0b_正控制.md`。
