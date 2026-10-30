# 方向 3：StereoMIS 上用器械遮罩訓練（2026-10-16，gate 討論串；計畫書，Kevin 10-16：先不實作）

承接 [gate_下一步_想法整理](../方法/gate_下一步_想法整理_2026-10-16.md) 線 B。

**要回答**：用「幾乎是 GT」的遮罩（器械 mask）、正控制證明有效的遮法（M3），在訓練時就遮，pose 會不會比同配方不遮好？以及 pose loss 的梯度到底想不想壓低器械的注意力。

為什麼要重新設計：現有三個訓練版都分不出原因。期中 b16_gg／smooth_temporal 的 bias 有死區（g < 0 時梯度為 0，gate 沒學）；P2 只遮 camera／register（M1），正控制已顯示 M1 連 GT 都救不回；三者都沒有同配方的不遮／隨機遮對照，也沒有記梯度。

遮罩只用器械 mask。極線與 RAFT 遮罩在醫學影像上精確度太低，不用（Kevin 10-16）。

---

## 1. 三組（同配方，只差遮罩）

| 組 | 訓練時遮罩 | 回答 |
|---|---|---|
| R0 | 不遮 | 對照：在 StereoMIS 上微調本身帶來的變化 |
| R1 | 器械 mask，M3，global block 8–23，固定（不學） | 主組 |
| R2（條件式） | 同比例隨機遮罩（每幀被遮 patch 數和 R1 相同，位置隨機，每幀固定 seed），M3，block 8–23 | 排除「遮什麼都有正則化效果」。**只在 R1 比 R0 好 ≥ 2% 時才跑**（Kevin 10-16 問為什麼要跑；R1 ≈ R0 時 R2 回答不了任何事） |

可選 R3：器械 mask，M1（只限 camera／register），block 8–23。回答「PAGE-4D 的遮法在訓練時配上 GT 遮罩有沒有用」。R0–R2 跑完、R1 有效才排。

遮罩定義：`mask_left` 為 0（器械）的像素佔 patch ≥ 50% 標 1，和方向 2 的 `stereomis_inst` 標籤相同。整幀不可用（0 佔 ≥ 90%）的幀不進訓練取樣。

## 2. 配方（事前定）

- **起點**：VGGT-1B（和 P2 一樣，可比）。
- **架構與資料管線**：`tfs_rel` 的配方（`scared_cam_depth_tfs.yaml`）：temporal block 8 個、相對幀序、±2S 隨機視窗、凍結 patch_embed／frame_blocks／global_blocks 0–7。只換資料集。
- **loss**：camera + camera_smooth。depth loss 不計算，flow_geom 關掉（Kevin 10-16 確認）。理由：StereoMIS 沒有 GT depth；flow_geom 要用沒有監督的預測 depth，會多一個變因。
- **步數**：10 epoch × 1500 步，和 P2、`tfs` 相同，組間可比。
- **挑 ckpt**：train 留出 P2_5 當 val（不進訓練），每個 epoch 評 val c64（GT pose），取 val 最好的 ckpt；test 只在最後評一次。三組用同一規則。不用 `pose_eval.dataset: scared_primary`（那個模式的 `best_ate.pt` 是用 test 挑的）。
- **seed**：每組一個（42）。判讀門檻 2% 大於已知的 seed 擺動（`tfs` seed 42 vs 43：c64 0.0536 vs 0.0541，約 1%）。

### 2.1 資料：train split

- 序列：train 的 P1、P2_0–P2_4 訓練，P2_5 當 val（方向 2 §1.1，已轉好；只含原始有 mask 的幀）。
- **沒有 depth 時的三個阻礙**（`程式導讀/B_data.md`、`D_loss.md`）：
  1. anchor 池是 `valid_frac > 0.05` 的幀（`scared.py:136`）；StereoMIS 的 `valid_frac` 空 → 整條序列被略過。
  2. camera loss 與 camera_smooth 只在第 0 幀 `point_masks` 有效像素 > 100 時才算（`loss.py:207`、`loss.py:1103`）；沒有 depth → 永遠不算，也不報錯。
  3. **GT 尺度正規化**：GT 平移除以 s_gt = 所有有效像素的 3D 點到第一相機中心的平均距離（`normalization.py:100-103`）。沒有 depth → s_gt 沒有定義 → loss 非有限值，trainer 直接跳過 backward（`trainer.py:1096-1099`），一樣不報錯。
- **解法（Kevin 10-16 定案：A，改程式，StereoMIS 不靠 depth；尺度用 scale-invariant loss）**
  1. anchor：不改程式。`valid_frac.txt` 寫成「可用 = 1、整幀不可用 = 0」，anchor 池就是可用幀。
  2. camera loss／camera_smooth：新 flag `data.no_depth: true`，`valid_frame_mask` 一律為 True。
  3. **GT 正規化**（只決定 loss 的單位）：同一個 flag 下，s_gt = clip 內各相機中心到第一相機中心的平均距離，下限 = 該序列相鄰幀平移中位數 × S。
  4. **scale-invariant 平移**：每個 sample、每個 iterative stage，求 s* = ⟨T̂, T⟩ / ⟨T̂, T̂⟩（T̂ 預測平移、T 正規化後 GT 平移，對 S 幀攤平），**detach**，clamp 到 [0.01, 100]；camera loss 的平移項改比 s*·T̂ 與 T。
     - 方向：一定是把預測對到 GT，不能把 GT 對到預測（後者讓 T̂ → 0 時 loss → 0，會塌縮）。
     - **camera_smooth 也要用同一個 s*·T̂**：它直接看預測平移，不乘 s* 的話，縮小 T̂ 就能降低平滑懲罰，模型會往 T̂ → 0 漂。
     - ‖T̂‖ < 1e-6 的 sample 平移項記 0，並計數。
     - 旋轉、FoV 項不變。
  5. depth loss 必須是「不計算」，不是權重 0：0 × NaN 仍是 NaN。
  6. 理由：評估（snippet ATE）會先對齊尺度，訓練也對齊，模型只因軌跡的方向、形狀、相對長短被懲罰，不必從預訓練的「場景平均距離 = 1」重學單位。
- 放棄的解法 B（立體參考深度）：不改程式，但要先算 train split 的 RAFT 立體深度。

### 2.2 遮罩在訓練時怎麼加

遮罩是固定的、不需要對 bias 反傳，但 attention 本身要反傳：

- **建議**：global block 8–23 用 SDPA 的 boolean `attn_mask`（`True` = 可看），條件 `not (flag[k] and frame(q) ≠ frame(k))`，special token 的 key 一律可看。訓練時 S ≤ 12、N ≈ 13,400，mask 約 180 MB，可以接受；不需要 flex，也避開 P3 的崩潰問題。
- flag 預設關閉時，輸出要和原本逐位元相同。
- 新參數例如 `train_mask: {dir, mode: cross, start: 8}`；dataloader 依 `ids` 讀對應幀的 patch 標籤，放進 batch。
- 推論時遮罩開關沿用 `eval_scared.py` 的 `--mask_dir --mask_mode cross --mask_start 8`。

## 3. 梯度記錄（三組都做，每 250 步一次）

| 量 | 怎麼算 | 回答 |
|---|---|---|
| g_cam(l) | pose loss 對第 l 層 global block 輸出的 camera token 的梯度範數 | 梯度在哪一層開始變小 |
| g_D(l)、g_C(l) | 同上，器械 patch 與非器械 patch token 的平均梯度範數 | 梯度有沒有流到器械 patch |
| **∂L/∂b(l)** | 在 global block l，對「其他幀的器械 key」加一個值為 0、`requires_grad` 的純量 bias b_l，做一次 forward／backward，取 b_l 的梯度 | **pose loss 想不想壓低器械的注意力**：負值 = 想壓（gate 學得起來），≈ 0 = 不在乎 |


**值域監測（Kevin 10-16 加）**：同一個 probe batch、同樣每 250 步，對 global block 0–23 記錄（算法同 [步驟A](步驟A_注意力分析.md) §3.2，分塊算、不改訓練的 forward）：

| 量 | 內容 | 回答 |
|---|---|---|
| logit 值域 | 每列 logit 的「最大值 − 中位數」p50／p99，D× 最大 logit − 非 D× 最大 logit 的 p99 | 訓練中 logit 會不會長大到讓 −30 擋不住（R1、R2），或 R0 的值域怎麼變 |
| 剩餘質量 | R1、R2：遮罩後被遮 key 實際拿到的 attention 質量（max、p99.9） | 遮罩是否一直有效；> 1e-3 就是遮罩失效 |
| ρ×（器械） | 不遮時（R1、R2 另跑一次不遮的 probe forward）其他幀器械 key 拿到的注意力相對份量，camera query 與 patch query 各一 | 訓練有沒有讓模型自己更看或更不看器械 |

判讀（只記錄，用來解釋 5.2）：剩餘質量在任一記錄點 ≥ 1e-3 → 該組的遮罩在訓練中失效，5.2 的結論對該組不成立，回報。

∂L/∂b 需要 bias 可反傳：只在記錄的那一步用 SDPA 的 float `attn_mask`（math kernel）另外跑一個 batch，不影響訓練本身。R0 上的 ∂L/∂b 最關鍵：它回答「沒遮的模型，pose loss 會不會推它去遮器械」。

輸出 `logs/<run>/grad_probe.jsonl`，每行一步，每層一個值。

## 4. 評估

test 片段同方向 2（P2_6–P2_8 約 10 段，各 512 幀），c64 snippet ATE（µm）。

| eval | 回答 |
|---|---|
| R0、R1、R2 c64，推論時遮罩開（R0 開 = 用器械 mask M3 s8） | 主結果 |
| R0、R1、R2 c64，推論時遮罩關 | 增益來自推論時遮罩，還是訓練時的正則化 |
| VGGT-1B 零樣本、`tfs_rel` 零樣本 | 參考（方向 2 已有：178.8、157.1 µm） |

c5 只在主結果有一組 ≤ −2% 時才跑。

## 5. 判讀（事前定，跑之前不再改）

### 5.1 sanity（第一步，不過就停）

- train 資料：每條序列 anchor 池非空，camera loss 在 ≥ 90% 的 batch 有算（log 一個計數）。
- 遮罩全 0 時，R1 的 forward 和 R0 逐位元相同（同 seed、同 batch）。
- 第 0 步 objective 是有限值，camera loss 在 batch 裡有算（計數 > 0）。
- s* 的分布（每 250 步記 p5／p50／p95）：p50 長期偏離 1 超過 10 倍，或 ‖T̂‖ < 1e-6 的計數持續增加 → 平移尺度在漂，停下回報。
- R1 第 0 步：log 的被遮 key 比例和 `stereomis_inst` 的標 1 比例一致（±1%）。

### 5.2 主結果（R1 遮罩開 vs R0 遮罩關）

- R1 ≤ R0 −2%，≥ 70% 片段同方向，P2_6／P2_7／P2_8 三條各自平均同方向，且 R1 比 R2（遮罩開）也好 ≥ 2% → **訓練時遮器械有用**。
- R1 與 R0 相差在 ±2% 內 → **訓練時遮 GT 等級的遮罩也沒用**。
- R1 比 R0 差 ≥ 2% → 變差。
- R2 本身比 R0 好 ≥ 2%：改善來自遮罩的正則化，不算器械的功勞，另外標出。
- 其他 → 回報討論。

### 5.3 遮罩開關（只看 R1）

R1 遮罩關與遮罩開相差 < 2% → 增益（若有）來自訓練時的正則化，推論可以不遮（同 PAGE-4D 附錄 §8.1）。

### 5.4 梯度（只記錄，用來解釋 5.2）

預先寫好的解讀：
- R0 的 ∂L/∂b 在所有層都 ≈ 0（|∂L/∂b| < camera token 梯度範數的 1%）→ pose loss 不在乎器械的注意力，**gate 就算沒有死區也學不到東西**；和 5.2「沒用」一致。
- R0 的 ∂L/∂b 明顯為負、集中在某幾層 → pose loss 想壓器械，那幾層是 gate 該作用的位置；若 5.2 仍沒用，問題在遮法或層的選擇。
- g_cam(l) 在某層之前降到最後一層的 1% 以下 → 那一層之前的遮罩訊號傳不到 pose（梯度消失的位置）。

## 6. 要改的程式（Kevin 在 75／76 實作）

| 檔案 | 內容 | 估計 |
|---|---|---|
| `aggregator.py` | 訓練用的 boolean mask M3 路徑（§2.2），flag 預設關閉 | 約 30 行 |
| `normalization.py`、`loss.py` | `data.no_depth`：s_gt 改用軌跡、`valid_frame_mask` 一律 True、depth loss 不計算、平移 scale-invariant（camera 與 camera_smooth 共用 s*），§2.1 | 約 40 行，flag 關閉時逐位元不變 |
| dataloader（`scared.py` 或新 `stereomis.py`） | 讀 patch 標籤、依 `ids` 放進 batch；排除整幀不可用的幀 | 約 30 行 |
| trainer | `grad_probe`（§3），每 250 步 | 約 50 行 |
| config | `stereomis_cam_tfs_{nomask,inst_m3,rand_m3}.yaml`，以 `scared_cam_depth_tfs.yaml` 為 parent，只寫 delta | 3 份 |
| scratchpad | train split 的立體深度與 `valid_frac.txt`（方向 2b 的 `raft_mask.py --dump_depth` 加 `--split train`）；train split 的器械與隨機 patch 標籤 | 沿用 |

## 7. 成本

- 訓練 3 次，每次約 `tfs` 量級（10–12 小時）；R1、R2 的 boolean mask 約慢 10–20%。75、76 各跑一組，第三組排隊。
- train split 的立體深度：每幀 RAFT 2 次，幀數待查。
- eval：3 組 × 開／關 × c64，加 c5（條件式）。

## 8. 限制

- pose GT 是運動學，offset −4 是採用不是確定。
- 單一 seed。
- 器械 mask 標的是器械，不是「所有對 pose 不可靠的區域」（反光、變形沒標）。
- test 只有 3 條序列，片段之間不獨立。
- 訓練從 VGGT-1B 起，結果不能直接和 SCARED 上的 `tfs_rel` 比，只看組間差。

## 9. 待 Kevin 決定

1. 起點 VGGT-1B？（建議是；未回覆前以此為預設）
2. ~~loss~~：已定，camera + camera_smooth，flow_geom 關（10-16）。
3. ~~depth 阻礙~~：已定，解法 A + scale-invariant 平移（10-16）。
4. R3（M1 + 器械）先不排？（建議是；未回覆前以此為預設）
