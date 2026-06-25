# Dyn-VGGT：實作對應、開放決策、與實驗執行

> 本文件是 [dyn_vggt_method.md](dyn_vggt_method.md)（方法/理論）的**工程對應版**：程式落點、關鍵實作細節、開放決策、訓練/評測 SOP，以及**目前實作狀態與驗證結果**。章節編號 §M-x 指向方法文件。

---

## 1. 落地文件清單（對應本倉庫）

| 模塊 | 文件 | 改動 | 狀態 |
|---|---|---|---|
| 時序注意力 + 1D 時間 RoPE | [aggregator.py](../vggt/models/aggregator.py)、[rope.py](../vggt/layers/rope.py) | `RotaryPositionEmbedding1D`、`temporal_blocks`、`_process_temporal_attention`、`aa_order`、`temporal_every` | ✅ 已實作 |
| 雙場輸出組裝 | [vggt.py](../vggt/models/vggt.py) | `motion_head`/`flow_head`、`enable_temporal/motion/flow` 旗標、forward 組裝 `X_{t,p}` | ✅ 已實作 |
| 新 loss | [loss.py](../training/loss.py) | `compute_motion/flow/reproj/tsmooth_loss`、point `(1−m)` 屏蔽、四分支 gating | ✅ 已實作 |
| 尺度正規化（B6） | [normalization.py](../training/train_utils/normalization.py) | `normalize_camera_extrinsics_and_points_batch` 增 `scene_flow` 同步 `/avg_scale` | ✅ 已實作 |
| 數據（PointOdyssey） | [pointodyssey.py](../training/data/datasets/pointodyssey.py) | 返回 `motion_mask`（RNG-replay 對齊） | ✅ 已實作 |
| 數據 collation | [composed_dataset.py](../training/data/composed_dataset.py) | 轉發 `motion_mask`/`scene_flow_gt` | ✅ 已實作 |
| 訓練 config | [config/dyn_vggt_po_s0.yaml](../training/config/dyn_vggt_po_s0.yaml)、[dyn_vggt_po.yaml](../training/config/dyn_vggt_po.yaml) | S0 / S1 | ✅ 已實作 |
| warm-start 驗證 | [verify_warmstart.py](../training/verify_warmstart.py) | 載入 VGGT-1B 檢查 missing/unexpected + γ=0 | ✅ PASS |
| 全局對齊（④） | 新建 `dyn_alignment.py` | 前饋初始化的 test-time 優化 | ⏳ 未做（P4） |
| 評測 pipeline | 新建 eval 腳本 | §5 的 ATE/AbsRel/IoU/Chamfer | ⏳ 未做（P5） |
| Demo | [demo_viser.py](../demo_viser.py) | 動靜分色 + 軌跡可視化 | ⏳ 未做 |

---

## 2. 關鍵實作細節

### 2.1 1D 時間 RoPE（§M-3.1）
與 `RotaryPositionEmbedding2D` 同構，但整個 head_dim 用時間座標旋轉、不切 y/x：
```python
# vggt/layers/rope.py
class RotaryPositionEmbedding1D(RotaryPositionEmbedding2D):
    def forward(self, tokens, positions):     # tokens:(B*P,n_heads,S,head_dim); positions:(B*P,S) 整數幀索引
        feature_dim = tokens.size(-1)
        cos, sin = self._compute_frequency_components(feature_dim, int(positions.max())+1, ...)
        return self._apply_1d_rope(tokens, positions, cos, sin)
```
register token 的時間位置設 0 → RoPE 退化為 identity（即「不給時間 RoPE」，對應決策 B4）。

### 2.2 temporal 只更新串流 token、不進 `output_list`（決策 A2，關鍵）
aggregator 裡 token 有兩條去向：
1. **串流隱狀態 `tokens`**：frame→temporal→global 一路被更新（網絡內部計算流）。
2. **回傳給 head 的 `output_list`**：每層把 `frame_intermediates` 與 `global_intermediates` **沿通道維拼接成 `[B,S,P,2C]`**；所有 head 的 `dim_in=2*embed_dim` 正是吃這「frame+global 兩份」。

temporal **必須**更新 `tokens`（讓 global 與後續層看到時序融合），但**絕不可**把中間輸出拼進 `output_list` → 否則通道維變 `3C`、head `dim_in` 被迫改 `3*embed_dim`、**預訓練 head 載入失敗、warm-start 全毀**。故：
```python
def _process_temporal_attention(self, tokens, B, S, P, C, idx, pos=None):
    t = tokens.view(B,S,P,C).permute(0,2,1,3).reshape(B*P,S,C)   # 沿時間軸
    t = self.temporal_blocks[idx](t, pos=pos)
    out = t.view(B,P,S,C).permute(0,2,1,3).reshape(B*S,P,C)
    return out, idx+1        # 不回傳進 output_list 的 intermediate
```

### 2.3 兩個新頭（§M-5）
復用 `DPTHead`。**注意**：`activate_head` 會把最後一個 channel 切成 confidence，故實際 `output_dim` 比語意維度多 1：
```python
# vggt.py __init__（實際採用值）
self.motion_head = DPTHead(dim_in=2*embed_dim, output_dim=2, activation="sigmoid",  conf_activation="sigmoid") if enable_motion else None  # 1 motion + 1 conf
self.flow_head   = DPTHead(dim_in=2*embed_dim, output_dim=4, activation="linear",   conf_activation="expp1")  if enable_flow   else None  # 3 disp  + 1 conf
```
forward 在 point/depth 之後組裝 `predictions["world_points_dyn"] = world_points + motion_prob * scene_flow`。

### 2.4 warm-start γ=0（§M-3.2）
`Block` 在 `init_values` 為 falsy 時用 `nn.Identity()`（不是 LayerScale），故**不能**靠 `init_values=0` 取得 γ=0。做法：用正常 `init_values` 建塊後，把 `ls1.gamma`/`ls2.gamma` 歸零 → temporal 塊初期為 identity。

### 2.5 motion BCE 手寫（AMP 相容）
`F.binary_cross_entropy` 在 AMP autocast 下被禁用；motion 頭輸出已過 sigmoid，故手寫 BCE：`-(gt·log p + (1-gt)·log(1-p))`，autocast 安全。

### 2.6 loss 分支 gating（向後相容）
`MultitaskLoss.forward` 的 **每個**分支（含原 camera/depth/point）都加 `self.X is not None` 守衛 → config 設 `null` 即跳過（S0 用此關掉 camera/depth/point/reproj/tsmooth）。原 VGGT loss 行為不變。

### 2.7 PointOdyssey motion_mask 對齊（RNG-replay）
mask 需與 image/depth 經**相同**的隨機幾何增強。做法：跑 `process_one_image` 前存 `np.random` state，跑完還原 state、把二值 mask 當 pseudo-depth（nearest）再跑一次 → 取其空間輸出。零改動共用 transform helper，不影響既有 co3d/vkitti。

### 2.8 B6 尺度自洽
`scene_flow` 是 world point 的位移 → 與 `world_points` 同 `/avg_scale`（無偏移）。`L_reproj`/`L_flow` 因相機平移也已 `/avg_scale` 而天然尺度不變。不歸一 → `X^can+m·Δ` 量綱錯、**靜默不收斂**。

---

## 3. 開放決策與預設值（Open Design Decisions）

**A 類**影響架構正確性、**必須先拍板**；**B/C 類**可先用預設、列消融；**D 類**為超參。★=建議進 §5 消融表。

### A 類：必須先定（架構正確性）
| # | 決策 | **採用（預設）** | 消融 |
|---|---|---|---|
| A1 | `X^can` 語義 | **per-(t,p) 剛性分支 + 殘差位移**；VGGT 無跨幀對應，「跨幀共享真 canonical」落不了地 | ★（退回單一 world point） |
| A2 | temporal 是否進 `output_list` | **不進**；head 介面維持 `2C`、預訓練權重不動（§2.2） | 否（正確性） |
| A3 | `aa_order` + temporal 數量 | **`["frame","temporal","global"]`，均勻 k=3，共 8 塊** | ★（k / n_temporal） |
| B6 | `Δ`/`X^can` 尺度正規化 | **與 point map 同一 per-scene scale**（§2.8） | 否（正確性） |

### B/C 類：先給預設、列消融
| # | 決策 | **採用（預設）** | 消融 |
|---|---|---|---|
| B4 | special token 進 temporal? | **camera+patch 進且帶時間 RoPE；register 進但時間維補 0**（順序無關摘要槽）；γ=0 暖啟保證最壞無損 | ★（三檔 × pose ATE/depth；可視化 register attn） |
| B5 | 偽標籤 `m*` 自舉穩定 | **detach** depth/pose 算 `m*`（防作弊塌縮，S2 硬性必要）；有 GT 的合成集優先用 GT 算 `f^cam`；`α_m,β_m` 前寬後緊 anneal。curriculum 緩解：S0 的 depth/pose 凍結 → `f^cam` 一開始即可靠 | 可選 |
| B7 | temporal RoPE 長序列外插 | **訓練隨機 frame stride 增廣**；頻率 base 沿用 100，必要時 NTK-aware 插值 | 可選 |
| B8 | causal vs bidirectional | **雙向**（離線重建最準）；保留 causal flag 作串流 future work | 可選 |
| C9 | 全局對齊優化細節 | **先最小版**（只優化 pose、固定 depth）再加碼；長序列滑窗 + 窗間 pose 拼接 | ★（0 迭代 vs 對齊） |

### D 類：超參
| 超參 | 預設 | 備註 |
|---|---|---|
| 動態閾值 `τ` | 0.5 | 僅用於 `valid_frame` 計數；loss 主用連續 `m` |
| 靜態數據混入比例 | 30% | 防遺忘，★ 消融 |
| EMA-teacher 蒸餾 | 開、權重待定 | 只在 `m≈0` 區；是否必要待 ablate |
| `L_flow` 的 SEA-RAFT 配對 | 相鄰 `t→t+1` | 是否加長程影響快取大小 |
| 損失權重 `λ_m,λ_f,λ_rp,λ_ts,λ_tv` | 1.0/0.5/1.0/0.1/0.1 | 必調；reproj 已對角線正規化（`huber_delta≈0.01`） |

**拍板優先序**：實作前必須定 **A1、A2、A3、B6**；其餘可先取預設值開跑。

---

## 4. 開發與訓練流程（耦合分析 + 執行 SOP）

### 4.1 模塊耦合度（決定能否平行開發）
判準：① 是否共享參數；② 一塊輸出是否進另一塊的 forward/loss。皆否才是真 decoupled。

| 配對 | 耦合度 | 開發策略 |
|---|---|---|
| ② 雙場 vs ③ motion/flow 頭 | **強耦合**（② 組裝 ③ 的輸出） | 從頭綁一組 |
| ① temporal vs (②+③) | **弱耦合**（共享 backbone） | 可平行開發，**合併須重對齊** |
| ④ 全局對齊 vs 其餘 | **近乎完全 decoupled**（test-time 只讀凍結輸出） | 隨時並行、零成本合併 |

> 弱耦合陷阱：②③ 的頭若在原始（無 temporal）backbone 訓出，換 temporal backbone 後 feature 分布改變，「各自都 work」不保證直接拼起來 work，須靠一次聯合 fine-tune（P3）縫合。

### 4.2 執行 SOP（含 gate）
| Phase | 做什麼 | 凍結/解凍 | 啟用 loss | 數據 | gate |
|---|---|---|---|---|---|
| **P0 準備** | 擴 dataloader、smoke test | — | 各跑一步驗可降 | 小子集 | forward+backward 不 NaN、shape 對、loss 降 |
| **P1-A 時序探針**（可平行） | 原 VGGT 加 temporal(γ=0)，凍其餘 | 僅解凍 temporal | `L_depth`+`L_tsmooth` | 動態合成集 | video depth 時序一致性優於 VGGT；單幀不退 |
| **P1-B 雙場探針**（可平行） | 原 backbone 加兩頭+組裝 | 凍 backbone+原三頭 | `L_motion`+`L_flow` | 動態合成集 | `m` IoU 可用；`Δ` 方向正確 |
| **P2 合併** | 兩條線裝進同一模型 | — | — | — | 載入無誤、前向不崩 |
| **P3a 對齊新頭**（≈S0） | 兩頭適配新 backbone | 凍 backbone+原三頭 | `L_motion`+`L_flow` | 動態為主 | `m`/`Δ` 恢復 ≥ P1-B |
| **P3b 運動解耦**（≈S1） | 接閉環與動態屏蔽 | 凍 DINOv2 | + `L_reproj`+`L_tsmooth`；`L_cam` 靜態 `valid_frame`、`L_point` `(1−m)` | 動態+少量靜態 | 動態區深度/點雲、ATE 優於「VGGT 直接微調」 |
| **P3c 全網精修**（≈S2） | 全網小 lr + 防遺忘 | 全解凍 | 全 7 項 | 動靜混合(靜~30%) | 達/超 MonST3R；靜態無退化 |
| **P4 全局對齊**（decoupled） | test-time 4D 對齊 | 全凍 | — | eval 序列 | 精修模式指標再升、20–50 迭代收斂 |
| **P5 評測+消融** | 跑 §5 全 benchmark+ablation | — | — | eval 集 | 完整表 + ablation |

**執行要點**：P1-A/P1-B 可平行；P2 之後串行。P1 是降風險探針（小數據短步數早止損），SOTA 數字在 P3c 產生。所有 ablation 與 full model **共享 P3a 後 checkpoint** 起跑省算力。P3c 結束必須在純靜態 benchmark 驗不退化（硬性防遺忘 gate）。

**最小可跑路徑（code 視角）**：① aggregator+rope temporal → ② vggt 接兩頭+組裝 → ③ loss 四項+屏蔽 → ④ dataloader 返回 GT → ⑤ dyn_alignment → ⑥ 評測。

---

## 5. 評測協議（CVPR 標準）

在 MonST3R 同套 benchmark 同台競技。

| 任務 | 數據集 | 指標 |
|---|---|---|
| 相機 pose | Sintel / TUM-dynamics / ScanNet | ATE、RPE-trans、RPE-rot |
| Video depth | Sintel / Bonn / KITTI | Abs-Rel、δ<1.25、時序一致性（OPW / temporal-AbsRel） |
| 動態分割 | DAVIS / Sintel | IoU、F-measure |
| 點雲/重建 | Dynamic Replica / TUM | Chamfer、Accuracy、Completeness |
| 效率 | 各序列 | 秒/序列 vs MonST3R |

**消融**：①去 temporal-attn；②去雙場分解（退回單一 world point，A1）；③去動態屏蔽（pose `valid_frame`+point `(1−m)`）；④0 迭代 vs 全局對齊（C9）；⑤對比「VGGT 直接微調」（證架構創新非數據紅利）；⑥temporal 數量/間隔 `n_temporal×k`（A3）；⑦special token 進 temporal 三檔 + register attn 可視化（B4）。

**SOTA 對比**：MonST3R、CUT3R、DUSt3R、Robust-CVD、CasualSAM、原始 VGGT。
**賣點**：精度 ≥ MonST3R、速度快一個量級、可擴展到數百幀。

> ⚠️ **trainer 的 validation 只算 val-set 上的 loss，不算上表指標**。ATE/AbsRel/IoU/Chamfer 需獨立 eval pipeline（P5，尚未實作）。

---

## 6. 目前實作狀態與驗證

### 6.1 已驗證（smoke / P0）
- **架構 smoke**：RoPE1D pos=0=identity；temporal γ=0=identity；加 temporal 後 `output_list` 仍 `[B,S,P,2C]` 且輸出與原版逐位元相同（warm-start 嚴格等價）；雙場組裝廣播正確。
- **loss smoke**：7 項 forward+backward、objective 有限、梯度流向 motion/flow/world/depth。
- **warm-start 驗證**（`verify_warmstart.py`）：載入 `VGGT-1B.pt` → missing = `temporal_blocks+motion_head+flow_head`、unexpected = `track_head`（track 關閉）、temporal γ 全 0 → **PASS**。
- **P0 真資料 overfit**：full model（1.35B）在真實 PointOdyssey batch 上 objective 下降、無 NaN。
- **S0 launch smoke**（完整 `launch.py`→DDP→trainer）：warm-start 載入、PointOdyssey dataloader、trainable 65.3M（只 motion/flow）、訓練步 loss 計算、checkpoint 存檔、validation 跑通 → **exit 0**。

### 6.2 尚未做
- `dyn_alignment.py`（P4 全局對齊）。
- eval pipeline（P5 標準指標）。
- 靜態集 loader（TartanAir 等，S1/S2 防遺忘混入用；目前只有 PointOdyssey 可訓）。
- SEA-RAFT 光流快取（僅 mask-less 集的 `m*` 需要；PointOdyssey 有 GT mask 不需）。
- PointOdyssey `trajs_3d` → 稀疏 `scene_flow_gt`（可選加強）。

### 6.3 環境與啟動
- **主 repo = `/home/cvml-75/Desktop/vggt`**；env `vggt-dyn` 的 editable `vggt` 已 `pip install -e .` 指向此 repo（不再需 PYTHONPATH）。
- 訓練 harness 套件已裝：`hydra-core / omegaconf / iopath / fvcore / wcmatch`。
- warm-start 權重：`training/checkpoints/VGGT-1B.pt`（另有 `raft-things.pth` 供未來光流）。
- 數據：PointOdyssey 在 `/media/cvml-75/ssd2t1/data/point_odyssey`（train 264 seq / test 28 seq）。
- **啟動指令**（trainer 需 `LOCAL_RANK`/`RANK`，必用 torchrun）：
  ```bash
  cd /home/cvml-75/Desktop/vggt/training
  torchrun --nproc_per_node=1 launch.py --config dyn_vggt_po_s0      # S0 正式
  torchrun --nproc_per_node=1 launch.py --config dyn_vggt_po_s0_smoke # 極短 smoke
  ```

### 6.4 config 對應
| config | stage | 凍結 | 啟用 loss | 數據 |
|---|---|---|---|---|
| `dyn_vggt_po_s0.yaml` | S0 | backbone+原三頭 | motion+flow | PointOdyssey |
| `dyn_vggt_po.yaml` | S1 | DINOv2 patch embed | 7 項；point `mask_dynamic` 開 | PointOdyssey |
| `dyn_vggt_po_s0_smoke.yaml` | S0(smoke) | 同 S0 | 同 S0；極小 limit/img_nums | PointOdyssey |
