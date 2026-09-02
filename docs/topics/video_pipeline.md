# 最終呈現影片 —— 配方、耗時、進度

2×2 grid：左上 input 影片、右上 Ours、左下 VGGT-1B、右下 MonST3R。逐幀累積重建（polish
渲染），長完停頓，再依序往右／左／上／下擺動，每次回到中心停頓。

## 耗時（實測，非估計）

渲染成本只跟 **cloud panel 數 × 輸出幀數** 有關，input 影片那格不算。以下都在 24 核 CPU 上跑，
`--blend` 渲染、每個 cloud 1.2M 點、每格 540×720：

| 型態 | 幀數 | 實測 | 換算 |
|---|---|---|---|
| 單格 check（1 個 cloud，540×720） | 412 | 7m13s / 7m15s / 6m57s | **1.03 s / panel-frame** |
| grid4（3 個 cloud，1080×1440） | 400 | 21m26s / 21m26s | **1.07 s / panel-frame** |
| grid4 | 412 | 22m13s / 22m03s | **1.07 s / panel-frame** |

四支 grid4 連續跑的實測值：22m13s、21m26s、22m03s（第一支無前值可差）。四次量測的 panel-frame
單價都落在 1.03–1.08 s，跨解析度、跨格數一致，所以下面那條公式可以直接拿來估。

→ **預測公式：渲染秒數 ≈ 輸出幀數 × cloud 格數 × 1.05**

分鏡長度 = `intro + n_frames×build_hold + pause + swing_frames×4 + pause×4 + hold`，
目前參數（intro 12、build_hold 3、pause 20、swing_frames 24、hold 12）下：
- 60 幀素材 → 400 輸出幀 → grid4 約 **21 分鐘**
- 64 幀素材 → 412 輸出幀 → grid4 約 **22 分鐘**

GPU 端（RTX 5090 32 GB，**一次只能跑一個**，見下方陷阱）：

| 工作 | 幀數 | 實測 |
|---|---|---|
| MonST3R，舊設定（DUSt3R 模式、swin-5） | 64 | 1m54s |
| MonST3R，MonST3R 影片設定（flow loss + self-mask + batchify=False） | 60 | **6m15s** |
| 同上，實測平均（5 條，60–64 幀） | 60–64 | **約 7.5 分鐘／條** |
| eval_lesion 單 ckpt 單 folder | 60–64 | 約 1 分鐘 |

MonST3R 的影片設定比舊的 DUSt3R 模式慢 3.3 倍：多了 RAFT 光流預算（46 chunk × 0.35 s）和
`batchify=False` 的逐 pair 迴圈。修好後 flow loss 落在 3.1–3.3 並隨迭代下降，遠低於
`flow_loss_thre=25`，也就是它真的有參與最佳化（修好前貼著門檻 24.7，隨時會被整個丟棄）。

修好後量到的「軌跡長度／深度中位數」（無量綱，可跨 arm 比較）：MonST3R 從病灶1 的 **11.26**
降到 **0.585**、ds2 從 1.37 降到 0.390，跟 Ours（0.11–1.00）和 VGGT-1B（0.20–1.17）同一量級。
先前 MonST3R 的軌跡漂移確實來自缺少 flow loss 與 temporal smoothing，不是模型本身的問題。

## 三個模型的時間對比

全部在同一台機器上連續量測，一次只有一個進程在跑。推論用 `torch.cuda.synchronize()` 夾住計時
（CUDA 是非同步的，不同步的話量到的是送出 kernel 的時間而非跑完的時間），並先跑一次 4 幀 warm-up
排除 CUDA context 初始化。5 條序列（病灶1/2/3 各 60/64/60 幀、ds3/ds2 各 64 幀）× 3 個模型。

### 推論（秒，一條序列一次）

| | 病灶1 | 病灶2 | 病灶3 | ds3 | ds2 | lesion 均 | SCARED 均 | 總平均 | 每幀 ms | 權重載入 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours | 6.9 | 7.7 | 7.0 | 7.1 | 7.3 | 7.2 | 7.2 | **7.2** | 115 | 9.5 s |
| VGGT-1B | 6.9 | 7.8 | 7.0 | 7.0 | 7.2 | 7.2 | 7.1 | **7.2** | 115 | 6.7 s |
| MonST3R | 370.5 | 454.5 | 447.6 | 512.4 | 468.3 | 424.2 | 490.3 | **450.7** | 7212 | 2.3 s |

**MonST3R 比我們慢 63 倍**，而且這個差距是結構性的，不是實作差異：Ours 和 VGGT-1B 是單次
feed-forward，MonST3R 每條序列都要跑 550 對的 pairwise inference 加 300 次全域對齊迭代，外加
RAFT 光流預算。`gate` 對推論時間沒有可測量的成本 —— Ours 與 VGGT-1B 在五條序列上都是 7.2 s，
差異落在量測雜訊內（每次 ±0.1 s），與 GatePredictor 只是 aggregator 中段一個 MLP、且 attention
bias 走的是不影響 flash path 的小 op 這件事一致。

### 渲染（單格 540×720、412 幀）

| | ds2 | 病灶3 | 平均 |
|---|---|---|---|
| Ours | 6m55s | 7m06s | **7m01s** |
| VGGT-1B | 6m47s | 6m34s | **6m41s** |
| MonST3R | 6m12s | 6m25s | **6m19s** |

三者相差 10% 以內。`--blend` 會把每個 cloud 都降到 `--max_points 1.2M`，所以點數不是變因；
MonST3R 略快是因為它的 depth 有效像素比較少（512×512 對 518×518，且邊界的無效區較大）。
**渲染成本由分鏡長度決定，不由模型決定。**

## 兩個踩過的陷阱

1. **GPU 一次只能一個進程。** gate arm 在 992×992 的 60 幀就吃掉約 28/32 GB。並行跑第二個
   GPU 工作必然 OOM，而 `eval_lesion.py` 會把例外吞掉只記 `error: true`，**失敗時舊檔案還在**，
   看起來像成功。每次跑完都要驗 `n_frames`。
2. **MonST3R 的 flow loss 會除以零。** 見 `reference/monst3r/dust3r/cloud_opt/optimizer.py:24`
   的 LOCAL FIX 註解。另外 batchify 的 ego-flow warp 在 60 幀會 OOM，必須 `batchify=False`。

## 素材

| 序列 | 來源 | 時間窗 | 幀 | 初始視角 |
|---|---|---|---|---|
| 病灶1 | `lesion_seq/dense` | 3.00–4.97s | 60 | azim 0 / elev 0 |
| 病灶2 | `lesion_seq/dense` | 0.00–2.10s | 64 | azim 0 / elev −10 |
| 病灶3 | `lesion_seq/dense` | 3.00–4.97s | 60 | azim 0 / elev 0 |
| SCARED ds3 | `val/dataset3/keyframe3` | 開頭 64 幀 | 64 | azim 0 / elev 12 |
| SCARED ds2 | `val/dataset2/keyframe3` | 開頭 64 幀 | 64 | azim 0 / elev 12 |

擺動一律 `azim ±60 / elev ±40`，`--dist 0.9`，`--scale_norm --no_progress --frame_pct 2`。
SCARED 用的是 **val**（模型選擇集，`pose_eval.split: val` 挑出 best_ate.pt），不是 test；
真正 held-out 且對應論文 pose 表的是 `pose_seq/dataset{3,5}/keyframe4`。
