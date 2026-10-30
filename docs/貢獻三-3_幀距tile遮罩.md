> **貢獻三-3 幀距 tile 遮罩（兩階段）**（2026-10-22，規格串，第 2 版，待 Kevin 同意）
> 第 1 版（依幀相位取模）見 [貢獻三-3_依幀tile遮罩.md](貢獻三-3_依幀tile遮罩.md)，已被這份取代。

# 貢獻三-3：遮罩由「query 幀和 key 幀的距離」決定，空間上用 8×8 tile 週期鋪

Kevin 10-22 的決定：
- 幀內可以週期（tile 鋪滿），幀與幀之間不能週期（第 1 幀和第 25 幀不能用同一張）。
- 用「幀距」決定遮罩，幀距用**對數分桶**。
- tile 用 **8×8**。
- 做兩階段：A 凍結 `tfs_abs` 只學遮罩，B 固定遮罩後微調。同時跑 [貢獻三-4](貢獻三-4_D1遮罩微調.md)（D1 遮罩的微調）。

程式以三-2 的 `colgate`（flex、對 bias 反傳、預算寫進閘、溫度、straight-through、凍結）為底。下面只寫不同的地方。

---

## 1. 遮罩的定義

### 1.1 幀距分桶

query 在第 s 幀、key 在第 s′ 幀，Δ = |s − s′|：

| 桶 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Δ | 1 | 2 | 3–4 | 5–8 | 9–16 | ≥ 17 |

- 訓練 clip 是 S=32，看得到 Δ ≤ 31；c64 會出現 Δ ≤ 63。最後一桶包含 17 以上，所以 Δ ≥ 32 用的是訓練學過的同一個 tile，不會有沒訓練到的參數。
- 永遠保留（bias 0）：Δ = 0（同一幀）、key 在第 0 幀（參考幀）、special token 的 key、對角（query 看自己）。

### 1.2 參數與閘

$$\theta \in \mathbb{R}^{6\times 8\times 8}\ (384\ \text{個},\ 16\ \text{層共用}),\qquad g(b, y', x') = \sigma\!\Big(\frac{\theta[b,\ y' \bmod 8,\ x' \bmod 8] - \tau}{T}\Big)$$

- key 位置 (y′, x′)，y′ = 0..29、x′ = 0..36；邊緣的 tile 不完整。
- 初始化 θ ~ N(0, 0.01²)，seed 固定。
- 每層各一組（16 × 384 = 6,144 個）列為之後的 run。

### 1.3 預算：以 c64 的實際 key 數加權

不同桶在一個 c64 chunk 裡出現的「(query 幀, key 幀) 配對數」差很多，遠的桶配對多。為了讓 ρ 對應到測試時真正的 key 數，τ 用配對數加權：

$$w_b = \sum_{\Delta \in b} 2\,(63 - \Delta + 1)\ \ (\text{key 幀} \ge 1,\ 64\ \text{幀}),\qquad \tau\ \text{使}\ \frac{\sum_b w_b \sum_{y,x} g(b,y,x)}{64\sum_b w_b} = \rho_t$$

- 程式裡直接用 S_eval = 64 數一次每桶的配對數（含「key 不是第 0 幀」的條件），不要手算。
- 硬遮罩：θ 由大到小排序，依權重累加到 ρ 為止。
- 每桶實際保留多少由學習決定（可能學成「近的多留、遠的少留」），結束時報告每桶保留比例。
- ρ 預設 0.25。

### 1.4 平均項：每個 query 幀各一個

遮罩依 query 幀而不同，被拿掉的 key 也依 query 幀不同，所以平均 token 每個 query 幀各一個：

$$\bar k_s = \frac{\sum_{s',p'} (1-g_{s,s',p'})\,k_{s',p'}}{\sum (1-g_{s,s',p'})},\quad \bar v_s\ \text{同理}$$

kv 序列後面接 S 個平均 token，第 s 個只給第 s 幀的 query 看（bias 0）。

### 1.5 flex 的 `score_mod`

```python
# fq, fk: token → 幀號；pk: token → patch 編號（special 為 -1）；B: [6, 64] 的 bias 表（由 θ 算）
# kv 長度 N + S：最後 S 個是平均 token
def _score_mod(score, b, h, q_idx, kv_idx):
    is_mean = kv_idx >= N
    mean_ok = fq[q_idx] == (kv_idx - N)
    kv = torch.minimum(kv_idx, N - 1)
    d = (fq[q_idx] - fk[kv]).abs()
    keep = (q_idx == kv) | (d == 0) | (fk[kv] == 0) | (pk[kv] < 0)
    tile = (pk[kv] // 37 % 8) * 8 + (pk[kv] % 37 % 8)
    bias = torch.where(keep, 0.0, B[bucket[d], tile])
    return torch.where(is_mean, torch.where(mean_ok, score, -1e4), score + bias)
```

- `bucket` 是長度 64 的查表（Δ → 桶）。B 是 fp32，對它反傳（三-2 已驗證 flex 能對捕捉的張量求梯度）。
- 平均 token 不給其他幀的 query 看，用 −1e4（不用 −inf，避免 NaN）。

---

## 2. 第一階段（A）：凍結 `tfs_abs`，只學 θ

和三-2 §3 相同：只有權重的 ckpt、全部凍結、AdamW lr 5e-3 常數、wd 0、連續幀、S 用三-2 實測可行的值、2000 步（ρ 從 1 cosine 降到 0.25 於前 1000 步，T 1.0→0.05 到 1400 步，1400 步後 straight-through）。可訓練參數 = 384。

```yaml
defaults:
- <三-2 的 config 名>
- _self_
exp_name: c3_reltile_b25_A
colgate_schedule:
  rho_target: 0.25
model:
  colgate:
    param: rel_tile
    buckets: [1, 2, 4, 8, 16]      # 上界：1 | 2 | 3–4 | 5–8 | 9–16 | ≥17
    tile: [8, 8]
    share_layers: true
    budget_weight_S: 64
```

起手檢查（加在三-2 §4 之後）：
1. ρ = 1 時輸出和 `tfs_abs` 相同。
2. 抽一個 clip，手算幾個 (q, k) 配對的 bias，和 `score_mod` 的結果相同（含 Δ=0、第 0 幀、special、對角、平均 token 只給同幀 query）。
3. 每步加權保留比例 = ρ_t（誤差 < 1e-3）。

匯出：`relmask.npy`，bool [6, 8, 8]（True = 保留），加一份每桶保留比例。

---

## 3. 第二階段（B）：固定遮罩，微調

和三-4 用同一套設定（從 `tfs_abs` 權重出發、4500 步、lr 峰值 1e-5、可訓練部分與取樣同 `tfs_abs`），只把遮罩換成第一階段的 `relmask.npy`：

```yaml
defaults:
- <tfs_abs 的 config 名>
- _self_
exp_name: c3_ft_reltile_b25
max_epochs: 3
checkpoint:
  resume_checkpoint_path: checkpoints/tfs_fpos_seed43_weights.pt
  strict: false
model:
  relmask_path: outputs/colmask/relmask_b25.npy    # 新 kwarg；硬遮罩，走 §1.5 的 score_mod（B 換成 0 / −30 的表）
  relmask_buckets: [1, 2, 4, 8, 16]
  relmask_tile: [8, 8]
optim:   # lr 與三-4 相同
```

- **對照組和三-4 共用**：三-4 的 `c3_ft_ctrl`（不遮、同樣 4500 步、同 lr）。不用另外跑。
- ckpt 裡沒有遮罩，eval 要帶 `--relmask`。

---

## 4. eval

`eval_scared.py` 新增 `--relmask <npy> --relmask_buckets 1 2 4 8 16`，走 §1.5 的 score_mod（硬遮罩）。

第一階段只做 eval 的對照（都用 `tfs_abs`、25%、加權預算相同）：

| 組 | 遮罩 | 回答 |
|---|---|---|
| 不遮 | — | 基準 0.05375 |
| 幀距隨機 | 每桶保留數和學到的相同，位置隨機，3 seed | 學到的位置有沒有用 |
| 幀距均勻 | 每桶保留比例都是 25%（不讓近的多留），位置照學到的 tile 排序 | 「每桶留多少」這個分配有沒有用 |
| 三-2 欄遮罩 | 已有：學到 0.08032、隨機 0.07624 | 幀距版有沒有比「所有幀丟同一批」好 |

```bash
CK=<tfs_abs 的 ckpt>; M=outputs/colmask; R="--relmask_buckets 1 2 4 8 16"
C64="--split test/pose --n_frames 0 --chunk_size 64 --overlap 16 --no_depth"
C5="--split test/pose --n_frames 0 --chunk_size 5 --overlap 4 --no_depth"
DEP="--split test/depth --n_frames 0 --chunk_size 64 --overlap 16 --depth_protocol afsfm"
# 第一階段
python pipeline/benchmark/eval_scared.py --ckpts $CK $C64 --relmask $M/relmask_b25.npy $R
for f in relrand_b25_s0 relrand_b25_s1 relrand_b25_s2 reluniform_b25; do
  python pipeline/benchmark/eval_scared.py --ckpts $CK $C64 --relmask $M/$f.npy $R
done
python pipeline/benchmark/eval_scared.py --ckpts $CK $DEP --relmask $M/relmask_b25.npy $R
# 第二階段
FT=logs/c3_ft_reltile_b25/ckpts/last.pt
for P in "$C64" "$C5" "$DEP"; do python pipeline/benchmark/eval_scared.py --ckpts $FT $P --relmask $M/relmask_b25.npy $R; done
```

對照遮罩的產生（一次性 script，不進 repo）：讀 `relmask_b25.npy` 數每桶保留數，隨機版在每桶的 64 格裡隨機挑同樣多格；均勻版每桶取 θ 最大的 16 格（64 × 25%）。

---

## 5. 判讀（事前定，跑之前不再改）

**第一階段**：Δ = (該組 − 不遮) / 不遮。

- **A1 收斂**：同三-2 J1。
- **A2 學到的 vs 幀距隨機**：低 ≥ 2 個百分點且 ds3、ds5 都 ≤ 0 → 學到的位置有用；±2 內 → 位置不重要；高 ≥ 2 → 回報討論。
- **A3 學到的 vs 三-2 欄遮罩**：學到的 Δ 比三-2 隨機欄遮罩（+42%）低 ≥ 10 個百分點 → 「依幀距錯開」比「所有幀丟同一批」好。
- **A4 分配**：學到的 vs 幀距均勻差 ≥ 2 個百分點 → 「近的多留、遠的少留」這個分配有貢獻。
- **A5 描述**：每桶保留比例、8×8 tile 長什麼樣、相鄰桶的 tile 是否錯開。

**第二階段**：Δ = (該組 − 三-4 對照組) / 三-4 對照組。

- **B1 精度回不回得來**：c64 Δ ≤ +2% 且 depth 多張 Δ ≤ +2% → 「global 只看 25% 的 key，精度不掉」成立；+2% ～ +5% 回報討論；> +5% 不成立。
- **B2 和三-4 學到組比**（兩邊都微調過、同 25%）：c64 低 ≥ 2 個百分點 → 幀距 tile 遮罩比 D1 欄遮罩好。
- 門檻用 2%：已知 `tfs` 兩個 seed 的 c64 差約 1%。

---

## 6. 順序與成本

1. 程式：`ColumnGate` 加 `rel_tile` 參數化與加權 τ（約 40 行）、§1.5 的 score_mod 與每幀平均 token（約 40 行）、`eval_scared.py` 的 `--relmask`（約 15 行）、`vggt.py` 的 `relmask_path`（約 10 行）。
2. 第一階段 2000 步（約同三-2 一個 run）→ eval 與對照組。
3. 第二階段 4500 步（走 flex，估比三-4 對照組慢 1.5–2 倍）→ eval。
4. 三-4 可以在另一台同時跑；它的對照組 `c3_ft_ctrl` 也是這份第二階段的對照。

## 7. 限制

- 遮罩依幀距與位置，仍然和影像內容無關。
- global 的 RoPE 沒有幀號，這個遮罩第一次把「幀距」放進 global。連續幀訓練下訓練與測試的幀距定義一致；c64 的 Δ ≥ 32 只在測試出現，靠最後一桶涵蓋。
- 對角與平均項只能走 flex，所以這份只回答精度。真正的加速可以按 query 幀分組收集 key 再各做一次 SDPA（同一個 query 幀的 query 看同一組 key），留給三-5 的速度量測。
- 單一 seed。
