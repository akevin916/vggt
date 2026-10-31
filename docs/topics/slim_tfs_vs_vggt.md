# slim-tfs 對原版 VGGT 的逐行對比

**一句話**：`slim-tfs` 只比原版 VGGT 多一件事 —— 每 3 層插 1 個「沿時間軸做 attention」的 block，共 8 個；flag 關掉時和原版一模一樣。

- 對比對象：原版 = `main`（`a288dd0`，上游 VGGT）；新版 = branch `slim-tfs`。
- 範圍：`vggt/models/aggregator.py`（331 → 425 行，+97 / −3）、`vggt/models/vggt.py`（97 → 109 行，+17 / −5）。另有一個相依檔 `vggt/layers/rope.py`（+28 行），放在 §3。
- 這版只留 `tfs_fpos_seed43`（= `checkpoints/scared_tfs_abs/epoch_10.pt`）會走到的程式。colmask、relmask、gate、dual_stream 等都不在這裡；要載那些 ckpt 請用 branch `slim` 或 tag `archive-pre-slim`。
- 自己重看 diff：`git diff main slim-tfs -- vggt/models/ vggt/layers/rope.py`

---

## 1. 先看整體：多了什麼

拿一個具體輸入走一遍：SCARED 一段 S = 64 幀、每幀 518×420。

| 項目 | 原版 VGGT | slim-tfs（`enable_temporal=True`） |
|---|---|---|
| 每幀 token 數 P | 1 camera + 4 register + 30×37 patch = 1115 | 相同 |
| 一輪 aa-block 的順序 | frame → global | frame → temporal → global |
| aa-block 輪數 | 24 | 24 |
| 有 temporal block 的輪 | — | 第 2, 5, 8, 11, 14, 17, 20, 23 輪（0 起算），共 8 個 |
| temporal attention 一次看幾個 token | — | 每個位置看自己的 64 幀：1115 條序列、每條長 64 |
| 給 head 的輸出 | 24 個 `[B,S,P,2C]` | 相同，temporal block 不額外輸出 |
| 參數量 | — | 多 100,788,224（8 個 block，每個 18 個 state_dict key） |
| 位置編碼 | 2D RoPE（patch 的 y, x） | frame / global 不變；temporal block 用 1D RoPE（幀號） |

三種 attention 的差別，用一句對比說：

- **frame**：同一幀內的 1115 個 token 互相看。
- **global**：64 幀 × 1115 = 71,360 個 token 全部互相看。
- **temporal**（新增）：同一個位置（例如第 500 號 patch）的 64 個 token 互相看，只沿時間軸。

---

## 2. 逐行對比

行號：「原版」是 `main` 的行號，「新版」是 `slim-tfs` 的行號。

### 2.1 `vggt/models/vggt.py`

| 原版行 | 新版行 | 改了什麼 | 作用 |
|---|---|---|---|
| 18–19 | 18–22 | `__init__` 多一個參數 `enable_temporal=False` | 預設關，原版的呼叫方式不受影響 |
| 22 | 25–28 | `Aggregator(...)` 多傳 `aa_order` | 開 temporal 時傳 `["frame","temporal","global"]`，否則傳原版預設 `["frame","global"]` |
| 29 | 35–36 | `forward` 多一個參數 `frame_pos=None` | 每幀在序列裡的位置 `[B,S]`，只給 temporal block 用 |
| — | 46–50 | docstring 補 `frame_pos` 說明 | 說明文字 |
| 61 | 73 | `self.aggregator(images)` → `self.aggregator(images, frame_pos=frame_pos)` | 把幀號往下傳 |
| 65 | 77 | `torch.cuda.amp.autocast(enabled=False)` → `torch.amp.autocast('cuda', enabled=False)` | 換成新 API。行為相同；舊寫法在 torch 2.x 會印 deprecation warning |

head（camera / point / depth / track）的程式完全沒改。`tfs_fpos_seed43` 用的是 `enable_point=False, enable_track=False`，只建 camera 和 depth head。

### 2.2 `vggt/models/aggregator.py` —— `__init__`

| 原版行 | 新版行 | 改了什麼 | 作用 |
|---|---|---|---|
| 16 | 16 | import 多 `RotaryTimeEmbedding1D` | temporal block 的位置編碼（§3） |
| 70 後 | 71 | 參數多 `temporal_every=3` | 每 3 輪 aa-block 插 1 個 temporal block |
| 78 後 | 81–83 | 新增 `self.temporal_rope = RotaryTimeEmbedding1D(frequency=rope_freq)` | 頻率沿用 2D RoPE 的 100。空間 RoPE 不動，預訓練權重才能直接載入 |
| 117 後 | 123 | 新增 `self.temporal_every` | 存起來給 `forward` 用 |
| 123 後 | 130–162 | 新增 temporal block 的建構 | 見下 |

**新版 130–162 行在做什麼**

1. 只有 `"temporal" in aa_order` 時才建，否則 `n_temporal = 0`、`temporal_blocks = None`，state_dict 一個 key 都不多。
2. 個數 `n_temporal = aa_block_num // temporal_every = 24 // 3 = 8`。
3. 每個 block 和 frame / global block 用同一個 `Block` 類別、同樣的寬度（1024 維、16 個 head、每 head 64 維），只有 `rope=self.temporal_rope` 不同。
4. **γ = 0 初始化**：把每個 block 的兩個 LayerScale 係數 `ls1.gamma`、`ls2.gamma` 設成 0。`Block` 的輸出是 `x + γ₁·attn(x)` 再 `+ γ₂·mlp(x)`，γ = 0 時輸出 = 輸入，所以剛插進去的 temporal block 什麼都不做。設計上，把 VGGT-1B 權重載進開了 temporal 的模型，一開始的輸出應該和 VGGT-1B 相同；這點沒有另外測過。
5. 註解說「必須清零既有的 LayerScale」：`init_values` 是 0 或 None 時 `Block` 會用 `nn.Identity()`，沒有 `gamma` 可以清，所以不能靠傳 `init_values=0` 達成。

### 2.3 `vggt/models/aggregator.py` —— `forward`

| 原版行 | 新版行 | 改了什麼 | 作用 |
|---|---|---|---|
| 184 | 223–227 | 簽名多 `frame_pos: Optional[Tensor] = None` | 接幀號 |
| — | 232–235 | docstring 補 `frame_pos` 說明 | 說明文字 |
| 231 後 | 280–294 | 新增 `temporal_pos` 的計算 | 見下 |
| — | 298 | 新增計數器 `temporal_idx = 0` | 數到第幾個 temporal block |
| 237 | 301 | `for _ in range(...)` → `for block_iter in range(...)` | 需要知道現在是第幾輪 |
| 248 前 | 311–318 | 新增 `elif attn_type == "temporal":` 分支 | 見下 |

**`temporal_pos`（新版 280–294 行）**：每個 token 在時間軸上的位置，shape `(B·P, S)`。

- 沒給 `frame_pos` 時，位置是 `0, 1, …, S−1`。
- 有給 `frame_pos` 時，位置是「這幀的幀號 − 第一幀的幀號」。例如取到第 100、103、110 幀，位置就是 0、3、10，temporal block 因此知道兩幀實際隔了多遠。幀連續時兩種寫法結果相同。
- camera token 和 patch token 用上面的位置。4 個 register token 一律設 0，1D RoPE 在位置 0 是「不旋轉」，等於 register token 沒有時間位置。

**temporal 分支（新版 311–318 行）**：

- 只在 `block_iter % 3 == 2` 時執行，也就是第 2, 5, …, 23 輪；其他輪直接跳過，那一輪就和原版一樣是 frame → global。
- 它只更新往下流的 `tokens`，**不**往 `output_list` 加東西。所以 `frame_intermediates` 記的是 temporal **之前**的 token，`global_intermediates` 記的是 temporal 加 global **之後**的 token。head 拿到的仍是 24 個 `[B,S,P,2C]`，head 不用改。

### 2.4 `vggt/models/aggregator.py` —— 新函式 `_process_temporal_attention`（新版 379–399 行）

原版沒有這個函式，整段是新增的。

1. `(B·S, P, C)` → `(B·P, S, C)`：把時間軸換到「序列」那一維。以 S = 64 為例，變成 1115 條、每條 64 個 token 的序列。
2. 跑一個 temporal block。訓練時用 gradient checkpoint（和 frame / global block 一樣），推論時直接跑。
3. 換回 `(B·S, P, C)`，交給同一輪接下來的 global attention。

`_process_frame_attention`、`_process_global_attention`、`slice_expand_and_flatten`、`__build_patch_embed__` 都和原版逐字相同。

---

## 3. 相依檔：`vggt/layers/rope.py` 的 `RotaryTimeEmbedding1D`（+28 行）

- **做什麼**：依幀號旋轉 q 和 k，讓 attention 分數只跟「兩幀差幾幀」有關。
- **和 2D 版的差別**：2D 版把 64 維拆成兩半，一半依 y、一半依 x 旋轉；1D 版整個 64 維都依幀號旋轉。頻率表和旋轉運算直接繼承 2D 版（`_compute_frequency_components`、`_apply_1d_rope`）。
- 原版 `RotaryPositionEmbedding2D` 沒有改。

---

## 4. flag 關掉時和原版相同的證據

- 預設 `VGGT()` 的 state_dict 有 1797 個 key，和 `main` 相同。
- 開 temporal、關 point / track 的 state_dict 有 1485 個 key，和 `scared_tfs_abs/epoch_10.pt` 逐一對上，沒有缺 key 也沒有多 key。
- 12 幀 dump 比對（輸入：SCARED test/pose dataset3/keyframe4）：`pose_enc` 和 `depth` 與 `archive-pre-slim` 逐位元相同。

| ckpt | 結果 |
|---|---|
| `logs/scared_cam_depth_tfs/ckpts/epoch_10.pt` | 逐位元相同 |
| `checkpoints/scared_tfs_abs/epoch_10.pt` | 逐位元相同 |
| `checkpoints/VGGT-1B.pt` | 逐位元相同 |

依據：scratchpad 的 `verify_slim.py dump/compare`，2026-10-28。

---

## 5. 限制

- **eval 沒有傳 `frame_pos`**：訓練時 trainer 傳 `batch["ids"]`（`temporal_frame_pos: true`），這是幀在序列清單裡的位置。`pipeline/eval/vggt_infer.py` 呼叫模型時沒傳，所以推論一律用 `0…S−1`。模型只看「和第一幀差幾幀」，連續 chunk 的真實差就是 `0…S−1`（例：第 48–111 幀 → 減 48 → `0…63`），所以 test / val 數字不受影響。只有跳幀、挑關鍵幀或重複幀的輸入才會靜默算錯，那時要補傳。
- **這版載不了 colmask / relmask ckpt**：VGGT 不再接受 `enable_colmask` 等參數，載入時會直接報錯，不會靜默跑錯。
- **只驗了 eval forward**：training forward 和 `slim` 同一段程式，這次沒有另外測。
