# D1 欄遮罩：凍結 `tfs`、只學遮罩 —— 第一輪（2026-10-20 ～ 10-22）

> **狀態：第一輪跑完，結論未確認。** 學到的遮罩在 c64 上**沒有勝過同比例的隨機遮罩**：保留 50% 時打平（差 0.5%），保留 25% 時反而差 5.4%。
> 下一步（找原因／改做微調／收線）尚未決定，回頭條件也尚未訂。結論定案後寫進 [status.md](../status.md)，本檔只交代這一輪做了什麼、量到什麼。
>
> 名詞（欄遮罩、ρ\*、ρ_t、對角、平均項、c64）見 [glossary.md](../glossary.md)。

---

## 1. 這一輪要回答什麼

global attention 裡，每個 query 會看所有幀的所有 patch key。**這一輪問：能不能只留一部分 key（例如 25%），而且「學出來留哪些」比「隨機留」好？**

做法是**欄遮罩**：每個 global 層 ℓ 對每個 patch 位置 p′ 學一個保留權重 g_ℓ(p′) = σ(θ_ℓ(p′))，加在 attention 分數上：

```
score_ℓ(q, k_p′) = q·k/√d + log g_ℓ(p′)
```

同一個 p′ 在所有非參考幀共用同一個 g（所以遮罩和幀數 S 無關，S=32 學到的可直接拿去 c64 用）。

### 和原始設計（D1 兩階段）的關係

原始設計是「第一階段凍結模型學遮罩 → 第二階段二值化後微調模型」。**這一輪把順序反過來**（Kevin 10-20 決定）：直接拿已訓練完的 `scared_cam_depth_tfs_fpos_seed43` 當凍結的模型，只做「學遮罩」這一步，微調先不做。

- 好處：模型權重不動，不需要「多訓練同樣步數」那組對照。
- 代價：沒有微調，精度幾乎不可能超過不遮的 `tfs`。這一輪能主張的是**效率**（用 X% 的 key 維持精度），不是精度提升。

---

## 2. 決定清單

### Kevin 明確決定的

| 日期 | 決定 |
|---|---|
| 10-20 | 各層保留比例由學習決定，只約束全部層的平均 |
| 10-20 | 加 AVGGT 的對角與平均項 |
| 10-20 | 順序反過來：凍結 `tfs`、只學遮罩；S 盡量大、取樣改連續幀 |
| 10-20 | 不必和 AVGGT 網格同預算比較；最低對照是同比例隨機遮罩 |
| 10-21 | 稀疏度用**預算排程**（ρ_t 從 1 降到 ρ\*），掃 ρ\* = 50／25／10% |
| 10-21 | 每個 run 訓練 1000 步 |

### 執行時決定（已回報、未單獨確認）

- **預算用投影控制，不用罰項**：每步後把所有 θ 平移同一常數，讓平均 g 恰好等於 ρ_t。省掉罰項權重 μ 的調參。
- **平均項權重 = log Σ(1−g)**：代表被丟掉那批 key 的總量。AVGGT 可能是「一個 key 的權重」，該寫法也實作了（`colmask_mean: one`），本輪未用。
- **對角 = query 自己那個 token**（不是自己那一整幀）。
- **永遠不遮**：第 0 幀、special token（camera、register）。
- **二值化 = 全部層一起取前 k 名**；最後 20% 步數用 straight-through（forward 用硬遮罩、backward 用軟 g 的梯度）。
- **eval 一律用硬遮罩**（取前 k 名、比例 ρ\*）。
- **起點用 `tfs` 的 `last.pt`**，不用 `best_ate.pt`（後者是在 test 上挑的）。
- **影像固定 518×420**（30×37 patch），因為 θ 以 patch 位置索引，必須和 test 一致；其餘增強照 `tfs`。
- **S = 32**（不是想要的 64）：事前規則「30 GB 內最大的 S」，見第 4 節。
- **θ 初始 5（g≈0.993）、lr 0.02 固定、weight decay 0、Adam eps 1e-12**。

---

## 3. 設定

| 項目 | 值 |
|---|---|
| 凍結的模型 | `logs/scared_cam_depth_tfs_fpos_seed43/ckpts/last.pt`（只載權重） |
| 可訓練參數 | `aggregator.colmask_theta` [24, 1110]，共 26,640 個 |
| loss | `tfs` 原樣：camera + depth + flow_geom + camera_smooth |
| 資料 | SCARED train，連續 stride-1、每 clip 32 幀、518×420 |
| 步數 | 1000（2 epoch × 500）；ρ_t 在前 600 步從 1 線性降到 ρ\*；800–1000 步 straight-through |
| 每步時間 | 約 14 秒（RTX 5090，功耗上限 400 W） |
| eval | SCARED test，c64（64 幀一次、重疊 16），pose = snippet ATE、depth = depth multi abs_rel（afsfm）；flash 關、TF32 開 |

config：[scared_cam_depth_tfs_fpos_seed43_colmask_r25.yaml](../../pipeline/training/config/scared_cam_depth_tfs_fpos_seed43_colmask_r25.yaml)（`_r50`、`_r10` 只改 `colmask_rho`）。

---

## 4. 開跑前的檢查（smoke）

| 檢查 | 結果 |
|---|---|
| 遮罩全開時重現 `tfs` | ATE 差 0.19%，**未過**事前門檻 0.1% → 診斷：把 bias 全設 0 只換 kernel（SDPA → flex），差距完全相同（每一位一致）→ 差距來自 kernel，不是遮罩邏輯。依事前規則改用「flex、bias 0」當基準，相對差 0 |
| 顯存 | S=32：峰值 25.3 GB、16.2 s/step；S=48：flow_geom 的 RAFT OOM；S=64：global attention OOM → 取 S=32 |
| 梯度 | 可訓練參數只有 θ；θ 的梯度非零有限；θ 有更新 |
| eval sanity | r50 把遮罩關掉，c64 ATE = 0.05364648，與 `tfs` 完全相同 |

---

## 5. 結果

基準：`tfs` 不遮（flex kernel）ATE 0.05375、abs_rel 0.04394。隨機 = 每層保留個數與學到的相同、位置隨機，3 個 seed 平均。

| run | 保留 | 學到 ATE | 隨機 ATE | 學到 vs 隨機 | 學到 vs `tfs`（ATE） | 學到 abs_rel | 學到 vs `tfs`（abs_rel） |
|---|---|---|---|---|---|---|---|
| r50 | 50% | 0.05558 | 0.05586 | 好 0.5% | +3.4% | 0.04362 | −0.7% |
| r25 | 25% | 0.08032 | 0.07624 | **差 5.4%** | +49.4% | 0.04717 | +7.4% |
| r10 | 10% | 0.10594 | 未跑 | — | +97.1% | 0.04973 | +13.2% |

- 隨機 3 個 seed 之間很穩：r50 0.0554–0.0563、r25 0.0758–0.0767。
- 隨機的 abs_rel 平均：r50 0.04321、r25 0.04655。
- r10 的數字取自 trainer 第 2 epoch 的 eval（同一 `last.pt`、同協定）。

事前判讀：學到 ≤ 隨機 × 0.95 → 學到的位置有貢獻；差 <3% → 分不出；其他 → 討論。
**r50 = 分不出；r25 = 學到的更差（事前規則沒寫到這個方向，歸為討論）。**

### 待確認的詮釋

- 精度損失主要來自「留多少」，不是「留哪些」：同比例隨機遮罩在 25% 時也掉了 42%。
- 學到的位置選擇在 c64 test 上沒有用處，甚至有害（25%）。原因未查：可能是方法本身學不到，也可能是 S=32 train → c64 test 的落差。

---

## 6. 遮罩長什麼樣（數字）

| 指標 | r50 學到 | r50 隨機 | r25 學到 | r25 隨機 | r10 學到 | r10 隨機 |
|---|---|---|---|---|---|---|
| 鄰格一致率 | 0.824 | 0.624 | 0.941 | 0.761 | 0.984 | 0.905 |
| 相鄰層重疊（Jaccard） | 0.361 | 0.343 | 0.214 | 0.134 | 0.088 | 0.033 |
| 外圈 2 格保留率 | 0.465 | — | 0.274 | — | 0.113 | — |
| 內部保留率 | 0.510 | — | 0.243 | — | 0.096 | — |
| 訓練結束時 g 介於 0.1–0.9 的比例 | 0.763 | — | 0.249 | — | 0.051 | — |

- **鄰格一致率**：上下左右相鄰的 patch 中，留／丟狀態相同的比例（1 = 成大塊，越低越零碎）。
- **巢狀關係**：r25 保留的位置有 96.8% 也被 r50 保留（隨機預期 50%）；r10 的 100% 被 r25 保留（隨機預期 25%）。
- **各層**：3–16 層保留最少（r25 多數 <0.06、r10 幾乎全 0）；17、18、21–23 層保留最多（r25 0.52–0.91）。逐層數字在 `keep_per_layer.csv`。

圖（`outputs/colmask_vis/scared_cam_depth_tfs_fpos_seed43/`）：

| 檔名 | 內容 |
|---|---|
| `per_layer_keep.png`／`per_layer_drop.png` | 每層保留／丟掉的比例 |
| `masks_learned_r*.png`／`drop_learned_r*.png` | 學到的遮罩，24 層 × 30×37 格（前者彩色 = 留，後者彩色 = 丟） |
| `masks_random_r*_seed0.png`／`drop_random_r*_seed0.png` | 隨機對照，同排版 |
| `keep_frequency_vs_frame.png`／`drop_frequency_vs_frame.png` | 每個 patch 位置在 24 層中被留／被丟的比例，旁附一張 test 影像 |
| `layer_jaccard_r25.png` | r25 層與層之間保留位置的重疊度，學到 vs 隨機 |
| `soft_g_hist.png` | 訓練結束時 g 的分布 |

---

## 7. 限制

- **沒有省算力**：遮罩是把分數加 −10000，計算量不變。要報 FLOPs／速度，需另外實作只計算保留 key 的版本。
- **flex kernel 讓 eval 慢約 21%**（316 → 383 秒），且本身讓 ATE 差 0.19%；所有比較都已用 flex 版 `tfs` 當基準。
- **S=32 ≠ c64**：訓練看 32 幀、測試看 64 幀。要上 S=48 需把 RAFT 拆小批；S=64 還需再壓 global attention 的記憶體。
- **軟遮罩沒測**：只測了硬遮罩。不過最後 200 步的訓練已經是硬遮罩，測試與訓練一致。
- **straight-through 的限制**：被丟掉的 key 幾乎拿不到梯度，只能靠保留中的 key 掉到門檻以下才換進來。
- **r10 沒有隨機對照**。

---

## 8. 待決

- 下一步：(A) 在 train/val 用 S=32 比學到 vs 隨機的 loss，分出「沒學到」與「學到但到 c64 test 不成立」；(B) 二值化後微調模型；(C) 收掉這條線。
- 回頭條件（何時改回「先學遮罩、再微調」）：未訂。曾提議「ρ\*=25% 時 ATE 比 `tfs` 差 >5%」，若採用則已觸發（+49%）。

---

## 9. 執行時遇到的主機問題

10-21 10:17 第一次開跑時 GPU0 掉卡（`Xid 79, GPU has fallen off the bus`），同一秒兩張卡都被標成需要重開機，三個 run 全部中斷。這是 10-11、10-13 之後第三次同樣的事件。

功耗上限降到 400 W 後重跑：兩卡同時滿載 5.5 小時（未達事前定的 6 小時）＋單卡 3 小時，沒有再掉卡。原因仍未確定。主機問題的總整理在 [ops/machine_error.md](../ops/machine_error.md)（該檔記的是另一種簽名的內核崩潰）。

---

## 10. 程式與產物

### 程式（皆在 flag 後，關閉時與原模型逐位元相同）

| 檔案 | 改了什麼 |
|---|---|
| [aggregator.py](../../vggt/models/aggregator.py) | `enable_colmask`、θ 與 buffer、`_colmask_global_block_forward`（flex_attention + 對角 + 平均項）、`colmask_post_step_`（投影）、`colmask_hard_keep`／`colmask_random_keep` |
| [vggt.py](../../vggt/models/vggt.py) | 傳遞 `colmask_*`；`head_grad_checkpoint`（DPT head 逐 chunk 重算） |
| [dpt_head.py](../../vggt/heads/dpt_head.py) | `grad_checkpoint` 屬性（預設關） |
| [trainer.py](../../pipeline/training/trainer.py) | `warm_start_weights_only`；每步後呼叫投影並記 log；依 `max_epochs × limit_train_batches` 設定排程總步數 |
| [vggt_infer.py](../../pipeline/eval/vggt_infer.py) | 自動辨識 colmask ckpt |
| [eval_scared.py](../../pipeline/benchmark/eval_scared.py) | `--colmask_mode hard\|soft\|off\|random`、`--colmask_seed` |

### 產物

| 內容 | 位置 |
|---|---|
| 訓練 | `logs/scared_cam_depth_tfs_fpos_seed43_colmask_r{50,25,10}/`（第一次中斷的在 `*_crashed_1021`） |
| smoke／sanity | `outputs/colmask_smoke/scared_cam_depth_tfs_fpos_seed43/` |
| 學到 vs 隨機 | `outputs/colmask_eval/scared_cam_depth_tfs_fpos_seed43/summary_r50_r25.json` |
| 遮罩圖與數字 | `outputs/colmask_vis/scared_cam_depth_tfs_fpos_seed43/` |

### 重現

```bash
# 訓練（每個約 4.5 小時；中斷後加 --resume 接續）
torchrun --nproc_per_node=1 pipeline/training/launch.py --config scared_cam_depth_tfs_fpos_seed43_colmask_r25

# 學到的遮罩 / 隨機對照
python pipeline/benchmark/eval_scared.py --ckpts logs/scared_cam_depth_tfs_fpos_seed43_colmask_r25/ckpts/last.pt \
    --split test/pose --n_frames 0 --chunk_size 64 --overlap 16 --no_depth --colmask_mode hard
python pipeline/benchmark/eval_scared.py ... --colmask_mode random --colmask_seed 0
```

注意：第 5 節的數字是在 flash 關閉下量的（與 trainer 相同）；CLI 預設 flash 開，數字可能差 ≤0.1%。
