# 方向 4：用 frame／temporal attention 預測 global attention（2026-10-20，規格書；未執行）

承接 10-19 發想串「用 frame／temporal attention 導出 global attention 的稀疏模式」。本檔只規定**第一個檢查**（只做 eval、不訓練），判讀門檻在這裡事前寫死。

**要回答**：在同樣的 key 數下，由 `tfs` 自己的 frame／temporal attention 導出的 key 子集（讀法 A、C），涵蓋真實 global attention 的權重，是否明顯多於 AVGGT 式網格與隨機子集？

只有答「是」，才值得做第二步：同 key 預算下的稀疏 global，比 c64 pose。

---

## 1. 量什麼

### 1.1 主指標：權重涵蓋率

權重涵蓋率：對一個 query，被選中的 key 拿到的真實 global attention 權重總和。1 = 選中的 key 包含了全部權重，0.1 = 只拿到 10%。

- 真實權重 = softmax(q·kᵀ/√d) 的一列，用該 global block 的 q、k（RoPE 之後）重算。
- 每個 global head 各算一次，再對 16 個 head 取平均。
- 選中的 key 一律包含**所有 special token**（camera＋register，每幀 5 個，各方法都送）。預算只算 patch key。
- 改用涵蓋率而不用發想串寫的 recall@k（top-k 重疊比例），理由：稀疏化損失的是被丟掉的權重，涵蓋率直接量這件事；top-k 重疊對「權重很平」的列不敏感。recall@k 仍一併輸出，不進判讀。

### 1.2 預算

（10-20 實作前改：對照 AVGGT 原文後，預算改由 AVGGT 網格決定。）

預算 K：每個 query 可以看的 patch key 數，各方法相同。K = AVGGT 網格在步長 σ 下保留的 patch 數：第一幀整幀 + 其他幀每個 s_h×s_w 窗格的第一個 patch。

- 主預算 **σ = 16（4×4）**：K = 6,150，佔全部 patch key 的 8.7%（S = 64、每幀 37×30 = 1,110 個 patch）。
- 另報 σ = 64（K = 2,370，3.3%）、σ = 4（K = 19,065，26.8%），不進判讀。
- 每個 query 自己的 key 和 special token 一樣免費（AVGGT 的 diagonal preservation），不算進 K。
- AVGGT 的 mean-fill（被丟掉的 key 合成一個平均 key）不是 key 子集，這個檢查不模擬它。

### 1.3 query

- **camera query**：每幀的 camera token，共 S 個。pose 由它讀出，**判讀用這組**。
- **patch query**：每個 clip、每層隨機抽 512 個 patch token（固定 seed）。稀疏化會套在全部 query 上，所以這組當副條件。

---

## 2. 比較哪些 key 子集

所有方法的 key 數相同（= ρ·S·P）。frame／temporal 的權重一律**先對 head 取平均**再用，因為不同類型 block 的 head 編號互不對應。

| 方法 | 怎麼選 key | 依賴 query？ |
|---|---|---|
| A（兩跳） | 取 temporal 權重 A_t^{(p)}(s→·) 最高的 k_s 幀；在每個被選的幀 s′，取 frame 權重 A_f^{(s′)}(p→·) 最高的 k_p 個 patch。k_s = min(S, 4ρS)，k_p = ρS·P / k_s | 是 |
| A-3×3 | 同 A，但第一跳的 temporal 權重改用 p 周圍 3×3 槽位的平均（發想串盲點 3） | 是 |
| B（frame 熱門錨點） | 每幀取「被 frame attention 收到最多權重」的 ρ·P 個 patch，所有 query 共用 | 否 |
| C（temporal 選幀） | 用 camera 槽位的 temporal 權重 A_t^{(cam)}(s→·)，取最高的 ρS 幀，這幾幀的 patch 全收 | 只依賴 query 所在幀 |
| 網格 | 每幀以固定步長取 patch，使每幀 ρ·P 個（AVGGT 式；實作前對照 AVGGT 原文核對取點方式，若不同以原文為準並記錄差異） | 否 |
| 隨機 | 從全部 patch 均勻抽 ρ·S·P 個，每個 query 獨立抽，固定 seed | — |
| oracle | 真實權重最高的 ρ·S·P 個 patch。只當上限用 | — |

- patch query 的 p 就是它自己的槽位。camera query 的「槽位」是 camera token 本身：A 的第一跳用 camera 槽位的 temporal 權重，第二跳用 camera token 在 frame attention 裡的那一列。
- k_s、k_p、ρ 都事前固定，不在結果出來後挑最好的組合。

---

## 3. 模型、資料、層

- **ckpt**：`logs/scared_tfs_seed43/ckpts/epoch_10.pt`。原本的 `tfs`（seed 42）權重在另一台機器；seed 43 是同配方，c64 0.0541 vs 0.0536。
- **資料**：SCARED **val**（`data/train/scared/val/`，dataset2–7 的 keyframe3）。不用 test，因為這個檢查會決定方法設計。
- **clip**：每個 keyframe 序列取 2 個不重疊的連續 64 幀片段（序列開頭與中段），共約 12 個 clip。一次 forward 整段（`chunk_size = 0` 的意思：64 幀一起進）。
- **層**：
  - 主層 **8、11、14、17、20**：temporal block 在同一輪、global 之前跑（每輪順序是 frame → temporal → global，temporal 只在 block_iter 2, 5, …, 23 跑），所以 A_t 是新鮮的。
  - 副層 9、12、15、18：沿用上一輪的 A_t（發想串盲點 4）。只報告，不進判讀。

---

## 4. 判讀（事前寫死）

先定兩個數：

- **基線 G** = max(網格, 隨機) 的權重涵蓋率。
- **差距 Δ_X** = 方法 X 的涵蓋率 − G。全部指 ρ = 1/16、camera query、主層與 clip 的平均。

依序判斷：

1. **oracle − G < 0.10** → 本方向關閉。真實權重本身就很分散，任何選法都贏不了網格多少，稀疏化沒有可利用的結構。
2. **A 或 C 有 Δ ≥ 0.10**，而且這個方法在 5 個主層中至少 4 層贏 G，而且它在 patch query 上的 Δ ≥ 0 → 進第二步（同預算稀疏 global，比 c64 pose；另寫規格）。
3. **A 和 C 的 Δ 都 < 0.03** → 本方向關閉，記成負結果。
4. 其他情況 → 回報討論。

B、A-3×3、副層的數字只回報，不改變上面的判斷。B 和 A-3×3 若出現 Δ ≥ 0.10，在第 4 種情況的討論裡提出。

門檻 0.10 的意思：同預算下多拿到 10% 的權重。這是事前的先驗，沒有既有數據支撐；選它是因為小於這個差距，第二步的 pose 差異很可能落在 seed 擺動（約 1%）以內。

---

## 5. 批次的第一步：sanity check（不過就中止並回報）

1. **重算一致**：用擷取到的 q、k、v 重算 global attention 輸出，和該 block 實際的 attention 輸出比，相對誤差 < 1e-2（bf16）。檢查擷取點對不對。
2. **ρ = 1 時**，所有方法的涵蓋率 = 1（誤差 < 1e-4）。
3. **隨機的涵蓋率** ≈ ρ ×（該 query 落在 patch key 上的權重比例），相對誤差 < 20%。檢查指標程式本身。
4. **oracle ≥ 其他所有方法**（逐 query 成立）。
5. **掛 hook 不改輸出**：同一個 clip，有掛與沒掛 hook 的 pose 編碼 max abs diff < 1e-5。

---

## 6. 實作與成本

- **擷取方式**：在 frame／temporal／global block 的 attention 上掛 forward hook，存 RoPE 之後的 q、k（v 只在 sanity check 1 用）。不用現有的 `probe` 路徑：它會建 N×N 的 bias，c64 時 N ≈ 71k，放不下。
- 真實 global 權重只算被選的 query 列（S + 512 列 × N 個 key × 16 head），逐層、逐 clip 在 GPU 上算完就丟，不存完整矩陣。
- **程式**：預估約 200 行，超過 50 行的門檻。需要它的原因是現有 diag 都不擷取 frame／temporal 的 attention 權重。不需要新 dependency 或新的資料前處理。依規則放 scratchpad，不進 repo。
- **輸出**：`outputs/global_sparsity_probe/scared_tfs_seed43/`（經 `default_output_dir`）：每層×方法×預算×query 類型的涵蓋率與 recall@k json，加一張「涵蓋率 vs 層」的圖。
- **成本**：單張 5090，約 12 次 64 幀 forward 加上逐層的權重計算，預估 20–40 分鐘 GPU。不訓練。

---

## 7. 這個檢查回答不了的事

- 涵蓋率高不等於 pose 不變：丟掉的權重即使少，也可能剛好是關鍵的 key。這要第二步才知道。
- 只在 `tfs` 類的權重上看。零樣本 VGGT-1B 沒有 temporal block，A、C 無法定義。
- 只在 SCARED 上看。器械多的 StereoMIS 上模式可能不同。
- novelty（和 TimeSformer／ViViT 的 factorized attention、FasterVGGT 的 block-sparse 差在哪）不在這個檢查的範圍，第二步之前要另外查證。
