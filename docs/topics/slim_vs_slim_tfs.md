# slim 對 slim-tfs 的逐段對比

**一句話**：`slim` = `slim-tfs` 再加一個「高斯時空遮罩」（gaussmask，3-10）—— 在 global attention 裡，query 依它到 key 的位移遠近丟掉一部分 key；遮罩關掉時，兩者算出的結果逐位元相同。

- 對比對象：`slim`（2026-10-29 起只留 gaussmask；colmask、relmask 已移除）對 `slim-tfs`（只留 `tfs_fpos_seed43` 會走到的程式，見 [slim_tfs_vs_vggt.md](slim_tfs_vs_vggt.md)）。
- 範圍：`vggt/models/aggregator.py`（slim 523 行 / slim-tfs 414 行）、`vggt/models/vggt.py`（119 / 110 行）、`vggt/models/gaussmask.py`（slim 新增 228 行）。
- 行號都是 `slim` 的行號。
- 自己重看：`diff vggt_tfs/vggt/models/aggregator.py vggt_slim/vggt/models/aggregator.py`
- colmask（3-2 / 3-4）、relmask（3-3 / 3-6 / 3-7）與它們的 ckpt / config，一律在完整版跑：branch `illu-vggt` 或 tag `archive-pre-slim`。

---

## 1. gaussmask 在做什麼

**做什麼**：query 看別幀的 patch key 時，用位移 (Δs 幀, Δy 格, Δx 格) 算一個橢球距離 Q；Q 太大的 key 丟掉，其餘依 Q 打折。同幀、第 0 幀（參考相機）、camera / register token 一律保留。

- Q = Δs²/σs² + Δy²/σy(Δs)² + Δx²/σx(Δs)²
  - 球版：σy、σx 固定（v = 0）。
  - 錐版：σy(Δs) = max(σy0 + v·|Δs|, 0.5)，x 同理；v 是學的，讓越遠的幀空間範圍變寬或變窄。
- Q ≤ c² 的 key 保留，bias = −Q/2；其他加 −1e4（等於丟掉）。
- **例子**：σs = 8、σy = σx = 6。query 在第 10 幀 (5, 5)，key 在第 14 幀 (7, 9) → 位移 (4, 2, 4) → Q = 16/64 + 4/36 + 16/36 ≈ 0.81 → 若 c² ≥ 0.81 就保留，bias ≈ −0.40。
- **c² 不學**：每步依預算 ρ（預設 0.25）用二分法解出來，權重是各位移在 64 幀 chunk 裡出現的次數。所以 σ 只決定形狀（時間 vs 空間的比例），保留多少由 ρ 決定。
- **作用範圍**：global 第 8–23 層，每層一組參數（球版 3 個、錐版 4 個）。
- 訓練：soft 邊界 σ((c² − Q)/T)，溫度 T 從 2 退火到 0.1，之後改 straight-through 硬邊界。eval：硬邊界。

---

## 2. `vggt/models/vggt.py`

| slim 行 | 內容 | slim-tfs |
|---|---|---|
| 21 | `head_grad_checkpoint`：DPT head 分段做 gradient checkpoint，只省記憶體 | 沒有 |
| 22–23 | `gaussmask=None`：一個 dict（§5） | 沒有 |
| 33 | 把 `gaussmask` 轉交給 `Aggregator` | 沒有 |
| 39–41 | 把 `head_grad_checkpoint` 設到 point / depth head 上 | 沒有 |

`forward` 兩邊相同。

---

## 3. `vggt/models/aggregator.py`

| slim 行 | 內容 |
|---|---|
| 21 | `_FLEX_FN`：編譯過的 flex attention，第一次用時才建 |
| 76–78 | 建構參數 `gaussmask: Optional[dict]` |
| 167–170 | 有 `enable` 或 `fixed_path` 才建 `self.gaussmask`（`_build_gaussmask`） |
| 307–311 | forward 先建好每層一張「位移 → bias」表 `gm_tab`；S 超過 `S_max`（64）直接報錯 |
| 368 | `_process_global_attention` 多一個參數 `gm_tab` |
| 385–397 | 第 8 層以上且有 `gm_tab` → `_gaussmask_global_block_forward`；否則走原版 global block（和 slim-tfs 同一段程式） |
| 409–425 | `_flex_attention`：讓每個 (query, key) 分數在 softmax 前加 bias，不必把 N×N 分數矩陣放進記憶體 |
| 427–439 | `_build_gaussmask`：把 config dict 轉成 `GaussMask` |
| 441–444 | `set_gaussmask_fixed`：eval 時從 json 掛上固定遮罩（`eval_scared --gaussmask`） |
| 446–457 | `_gaussmask_score_mod`：flex 裡每個分數查表 `tab[位移]`，該保留的直接回 0 |
| 459–475 | `_gaussmask_global_block_forward`：手動算 q / k / v（含 2D RoPE），走 flex，再接回 LayerScale 與 MLP |

其他部分（patch embed、2D RoPE、frame / global / temporal block、γ = 0、`frame_pos`）兩邊相同。

**為什麼要查表**：Q 只取決於位移，所以每層建一張 127×59×73（= 2·64−1 × 2·30−1 × 2·37−1，約 55 萬格）的表，flex 只查表。直接在 flex 裡對 σ 求梯度時，每個分數的梯度都搶同一個純量，S = 48 時每層 82 s；查表版 4.3 s。

---

## 4. `vggt/models/gaussmask.py`（`GaussMask`）

| 函式 | 做什麼 |
|---|---|
| `__init__` | 建參數 `log_s`、`log_y`、`log_x`（錐版加 `v`），和各位移的出現次數 `n` |
| `Q` | 每層的 Q 表 `[16, 127, 59, 73]` |
| `solve_c2_hard` / `solve_c2_soft` | 依預算解 c²（硬邊界 / soft 邊界），不傳梯度 |
| `tables` | 依 train / eval 與退火進度回傳 bias 表 |
| `index` | 把 (query 幀, key 幀, 格子座標) 換成表的位置 |
| `post_step_` | 每個 optimizer step 後步數 +1（trainer 呼叫） |
| `stats` / `export` | 讀數、把學好的遮罩存成 json 給 eval 用 |

---

## 5. config（`gaussmask` dict 的內容）

以 `c3_gball_A.yaml` 為例：

| key | 例子 | 意思 |
|---|---|---|
| `enable` | true | 開啟學習版 |
| `cone` | false | false = 球版，true = 錐版（`c3_gcone_A` 只改這個） |
| `rho` | 0.25 | 每層保留 25% |
| `S_max` | 64 | 位移表涵蓋最多 64 幀 |
| `start` | 8 | 從 global 第 8 層起作用 |
| `grid` | [30, 37] | patch 格數 |
| `init` | σs 8、σy 6、σx 6、v 0 | 初始寬度 |
| `T_start` / `T_end` / `T_steps` | 2.0 → 0.1，第 0–367 步 | soft 邊界溫度的退火 |
| `hard_from` | 367 | 之後改硬邊界 |
| `fixed_path` | （eval 用） | 從 json 讀固定遮罩 |

`c3_gball_A` 原本繼承 `c3_pos_b25_A` → `c3_reltile_b25_A` → `scared_cam_depth_tfs_fpos_seed43_colmask_r25`，2026-10-29 改成直接繼承 `scared_cam_depth_tfs_fpos_seed43`，把中間三層的設定寫進檔案本身。

---

## 6. slim-tfs 有、slim 沒有的

只剩一項：`temporal_every` 預設，slim-tfs 工作區是 **1**，slim 是 3（見 §8 警告）。其他 slim-tfs 的改動（`RotaryTimeEmbedding1D`、`frame_pos` 註解、`eval_scared --frame_pos`、`aa_order` 寫法）2026-10-29 已搬到 slim。

---

## 7. 證據（2026-10-29，RTX 5090）

- 預設 `VGGT()` 的 state_dict：1797 個 key，與 `main` 相同。
- 遮罩關：`tfs_abs/epoch_10.pt` 12 幀 dump，對 `archive-pre-slim` 的 `pose_enc` / `depth` 逐位元相同。
- gaussmask 球 / 錐：eval 的 `pose_enc` / `depth` 與移除 colmask / relmask 之前的 slim 逐位元相同；那份 slim 又與 `illu-vggt` 逐位元相同（2026-10-28）。
- `c3_gball_A` / `c3_gcone_A`：Hydra 展開後，與攤平前相比只少了 26 個 colmask / relmask 的 model key，其他設定完全相同；兩者都能 instantiate 出模型。
- 帶 `aggregator.relmask_*` 或 `aggregator.colmask_*` key 的 ckpt，`load_vggt_for_eval` 會直接拒絕並指向完整版。

依據：scratchpad 的 `verify_slim.py`、`gm/probe.py`、`slim/compose.py`。

---

## 8. 限制與警告

- **⚠️ gaussmask 在 RTX 5090 跑不起來**：它呼叫 flex 時沒指定 tile 大小，kernel 要 208 KB shared memory，上限 99 KB（`illu-vggt` 原版也一樣）。§7 的比對是在驗證 script 裡把 tile 設成 64×64、1 stage 跑的，repo 程式沒改。要在這台訓練 `c3_gball_A`，得先讓 `_gaussmask_global_block_forward` 帶上 `kernel_options`。
- **⚠️ 學過的 gaussmask ckpt**：`load_vggt_for_eval` 不重建 gaussmask，會印 WARNING 並丟掉 `aggregator.gaussmask.*`。eval 必須帶 `--gaussmask <export 出來的 json>`，否則模型等於沒有遮罩。
- **⚠️ 固定遮罩的舊 ckpt 不會被擋**：例如 `c3_full_reluni_b25`（3-6，用 `.npy` 固定 relmask 訓練），ckpt 裡沒有任何 relmask key，在 slim 會被當成一般 tfs 模型載入、不報錯。這類 ckpt 必須在完整版用 `--relmask` 跑。
- **⚠️ slim-tfs 工作區的 `temporal_every=1`**：現有 tfs ckpt 載進去會缺 288 個 key 且不報錯。slim 仍是 3。

---

## 9. 試過：用 flex BlockMask 跳過整塊被丟掉的 key（已放棄，2026-10-29）

**問題**：gaussmask 的 eval forward 用 BlockMask 跳過「整塊都被丟掉」的 (query 區塊, key 區塊)，會不會比沒有遮罩的原版快？事前門檻：≥ 1.2× → 採用；比原版慢 → 不加；中間 → 討論。

| 設定（tfs_abs，64 幀，bf16，20 次中位數） | 一次 forward |
|---|---|
| 沒有遮罩的原版（flash SDPA） | 4.140 s |
| flex gaussmask，初始 σ（8 / 6 / 6） | 6.505 s |
| flex + BlockMask，初始 σ（跳過 35% 區塊） | 6.305 s |
| flex gaussmask，σs 2 / σy 15 / σx 15 | 6.498 s |
| flex + BlockMask，σs 2 / σy 15 / σx 15（跳過 55% 區塊） | 4.852 s |

- BlockMask 對原版：0.66×（初始 σ）、0.85×（σs 2 / σy 15）→ 兩者都比原版慢 → 依門檻不加。
- 正確性：BlockMask 對一般 flex 的 pose_enc 最大差 4.5e-4 / 5.0e-4（< 1e-3）。
- 慢在哪：一般 flex 本身就比原版慢 2.4 s（64 幀、16 層），跳過區塊只省回 0.2–1.6 s。保留 25% 的 key 時，c² 很大，遮罩接近「只看前後幾幀、空間幾乎全留」，按 128 token 切的區塊只有 35–55% 能整塊跳過。
- 量測時 flex tile 設為 64×64、1 stage（RTX 5090 的 shared memory 限制）；換更大的 tile 或別張卡，數字可能不同。
- 訓練沒量：硬邊界段（`hard_from` 之後）跳過在數值上等價；soft 段沒有真正為 −1e4 的格子，只能以門檻近似。

依據：scratchpad 的 `gm/blockspeed.py`、`gm/blocksparsity.py`，結果 `gm/blockspeed.json`。
