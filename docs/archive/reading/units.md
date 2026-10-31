# Dyn-VGGT 程式重讀 —— 單元與自我驗證題

> **這份文件的用途**：這個 repo 的實作大部分由 AI 代寫且未經審查。本文件把整個專案切成 13 個閱讀單元，
> 依**資料流順序**（data → model → loss → train → eval）排列。每個單元讀完後：
> 1. 做該單元的自我驗證題（答案在 [answers.md](answers.md)，**先做完再看**）；
> 2. 填 [../spec_impl_ledger.md](spec_impl_ledger.md) 中該單元對應的條目。
>
> **規則**：題目答不出來 → 回去讀，不要翻答案。答案翻了才懂 → 在 ledger 備註欄記一筆，那是你原本的盲區。
>
> 行號以 2026-08-03 的 `dyn-vggt-v2` branch 為準。若對不上，先 `git log -p` 看那段改了什麼 —— 那本身就是有用的資訊。

---

## 進度表

| 單元 | 主題 | 主要檔案 | 完成 |
|---|---|---|---|
| U0 | 契約與 batch dict | `eval_utils/__init__.py`、`composed_dataset.py` | ☐ |
| U1 | 資料層骨架 | `base_dataset.py`、`dynamic_dataloader.py` | ☐ |
| U2 | 動態標籤 `m*` | `pointodyssey.py`、`motion_mask.py`、`preprocess/` | ☐ |
| U3 | Aggregator 骨架 | `aggregator.py`（非 gate 部分） | ☐ |
| U4 | Gate 機制核心 | `aggregator.py`（gate 部分） | ☐ |
| U5 | VGGT 外層與 predictions | `vggt/models/vggt.py` | ☐ |
| U6 | Loss dispatcher 與 `L_gate` | `loss.py:17-147, 702-780` | ☐ |
| U7 | 其餘 loss 項 | `loss.py:150-700` | ☐ |
| U8 | Trainer 主迴圈 | `trainer.py:483-556, 732-1026` | ☐ |
| U9 | 驗證雙通道與 checkpoint | `trainer.py:251-299, 397-462, 558-730` | ☐ |
| U10 | Config 層與凍結 | `config/*.yaml`、`train_utils/freeze.py` | ☐ |
| U11 | Eval 管線 | `eval_utils/`、`benchmark/eval_sintel.py` | ☐ |
| U12 | 診斷工具與 docs 對讀 | `diag/`、`docs/` | ☐ |

---

## U0 — 契約與 batch dict（暖身）

**目標**：知道「模型/loss 到底收到什麼」與「產物該寫到哪」，之後每個單元都靠這兩件事定位。

**讀**
- [training/eval_utils/\_\_init\_\_.py](../../training/eval_utils/__init__.py)（9 行）—— eval 層三目錄的分工契約
- [training/data/composed_dataset.py:142-161](../../training/data/composed_dataset.py#L142-L161) —— **batch dict 的權威定義**
- [docs/table.md](../table.md) 開頭「世代地圖」與「Δ% 的基準」兩節
- [docs/method.md:27-51](../method.md#L27-L51) —— 狀態 banner

**盯**
- 哪些 key 無條件存在、哪些條件存在
- `motion_mask` 在 `composed_dataset.py:156-158` 被零填充的理由與副作用

**題目**
- **Q0.1** batch dict 裡哪一個 key 是**無論資料集是否有動態標註都一定存在**的？它在缺標註時的值是什麼？這對 `L_gate` 的統計（例如 `dyn_frac`）有什麼影響？
- **Q0.2** `scene_flow_gt` 與 `tracks` 兩個 key 的出現條件分別是什麼？
- **Q0.3** 依 `eval_utils/__init__.py` 的契約，一支「要寫進論文的 Sintel pose 數字」的 script 該放哪個目錄？一支「這禮拜想確認 gate 是不是在閃爍」的 script 呢？兩者輸出各該落在哪個路徑？
- **Q0.4** `docs/table.md` 說 Δ% 的分母是「該列自己的 `off`」。為什麼不是拿 base model 當共同分母？
- **Q0.5** 狀態 banner 說「§3.6/§4.3/§5.4/§8 只放診斷、不得重下結論」。這條規則想防的是哪一種具體錯誤？

**ledger**：E-01

---

## U1 — 資料層骨架

**目標**：搞懂一個 batch 是怎麼從磁碟長出來的，以及 augmentation 對 mask 做了什麼。

**讀**
- [training/data/base_dataset.py:60](../../training/data/base_dataset.py#L60)（`__getitem__`）、`:95`（`get_target_shape`）、`:115`（`process_one_image`）、`:237`（`get_nearby_ids`）
- [training/data/composed_dataset.py:85-197](../../training/data/composed_dataset.py#L85-L197)
- [training/data/dynamic_dataloader.py:17,67,94,199](../../training/data/dynamic_dataloader.py#L17)

**盯**
- mask 是靠什麼機制跟 depth 走同一組隨機 crop/resize？若不同步會怎樣？
- clip 長度、長寬比每個 batch 都在變 —— 誰決定、以什麼為預算上限？

**題目**
- **Q1.1** `process_one_image` 對 image / depth / mask 用的 interpolation 分別是什麼？為什麼不能全用同一種？
- **Q1.2** 同一個 sample 內的多幀，色彩 jitter 是「每幀獨立」還是「整段一致」？由哪個參數控制？這對 `L_static_photo`（跨幀比 RGB）意味著什麼？
- **Q1.3** `DynamicBatchSampler` 在什麼預算下決定一個 batch 放幾幀？如果 `img_nums=[4,16]`，某個 batch 抽到 4 幀、另一個抽到 16 幀，兩者的 batch size 會一樣嗎？
- **Q1.4** `camera_smooth` 需要 `S≥3`。資料端有任何機制保證這件事嗎？在哪裡？
- **Q1.5** `get_nearby_ids` 抽幀時允許重複 index 嗎？這件事在下游哪個 loss 會爆炸，該 loss 怎麼防？

**ledger**：T-05

---

## U2 — 動態標籤 `m*` 怎麼來的

**目標**：這是「AI 自作主張」最容易藏身的地方 —— 你以為在訓 A 標籤，其實在訓 B。

**讀**
- [training/data/datasets/pointodyssey.py:108-146](../../training/data/datasets/pointodyssey.py#L108-L146) —— `_motion_mask_path` / `_read_binary_mask` / `_binary_dynamic_mask`
- [training/data/motion_mask.py:49](../../training/data/motion_mask.py#L49)（`compute_ego_flow`）、`:81`（`derive_motion_mask`）、`:107-133`（cache）
- [training/data/preprocess/po_instance_dynmask.py](../../training/data/preprocess/po_instance_dynmask.py)（144 行，全讀）
- 快掃 `tartanair.py:225,240`、`spring.py:132-134`、`waymo.py:150-154,177-179`、`sintel.py:135,233`
- 對照 [docs/method.md §3.5.2](../method.md#L254)

**盯**
- `dynamic_source="instance"` 的三層 fallback：`dynmask_inst` → `dynmask_raft` → 全零
- Sintel 的 `m_geo` 用的是 GT `.flo`，不是 RAFT

**題目**
- **Q2.1** `dynamic_source="instance"` 時，一個缺 `dynmask_inst/` 的 PO 序列會拿到什麼標籤？你要怎麼從**訓練 log** 察覺這件事正在發生？（提示：你察覺不到 —— 說明為什麼，以及該加什麼才察覺得到）
- **Q2.2** `po_instance_dynmask.py` 判斷一個 instance 連通塊「在動」的準則是什麼？門檻是絕對值還是相對值？靜態物體被誤標的風險來自哪裡？
- **Q2.3** `derive_motion_mask` 的門檻是 `max(abs_thr, rel_thr·‖f_gt‖)`。`rel_thr` 這一項是為了解決哪個具體序列的什麼問題？如果只留 `abs_thr` 會怎樣？
- **Q2.4** `compute_ego_flow` 需要哪些輸入？其中哪些是 GT、哪些是預測？這決定了 `m_geo` 能不能用在訓練時，為什麼？
- **Q2.5** PO 原生 `masks/`（`dynamic_source="native"`）為什麼被否決？它標的是什麼、不是什麼？
- **Q2.6** TartanAir 的 `motion_mask` 恆為全零。它進到 `L_gate` 的 BCE 裡是什麼角色？這算 bug 還是設計？
- **Q2.7** Sintel 的 mask 有落盤 cache。cache 的 key 包含哪些參數？如果你改了 `rel_thr` 但沒清 cache，會發生什麼？

**ledger**：G-10、G-11、G-12

---

## U3 — Aggregator 骨架（先不看 gate）

**目標**：建立 token layout 的正確心智模型。**這是 U4 的前置**，跳過會讀不懂 gate。

**讀**
- [vggt/models/aggregator.py:97-236](../../vggt/models/aggregator.py#L97-L236)（`__init__`）
- `:279-393`（`forward` 主迴圈）、`:395-417`（frame）、`:575-595`（temporal）、`:598-621`（`slice_expand_and_flatten`）
- [vggt/layers/block.py](../../vggt/layers/block.py)（整支，短）—— U4 要拿來逐行比對

**盯**
- 三種 attention 各自把 tokens reshape 成什麼形狀
- `patch_start_idx = 1 + num_register_tokens`（`:226`）
- temporal block 的插入條件（`:371`）與 γ=0 warm-start（`:203-207`）

**題目**
- **Q3.1** frame / global / temporal 三種 attention，tokens 分別被 reshape 成哪三種形狀？各自「誰跟誰互看」？
- **Q3.2** 在 global attention 的 `[B, S*P, C]` 佈局下，寫出「第 s 幀的 camera token」在序列軸上的 index。`tokens[:, :patch_start_idx]` 取到的是什麼？
- **Q3.3** `slice_expand_and_flatten` 為什麼要把 camera token 分成「第 0 幀」與「其餘幀」兩種？
- **Q3.4** `enable_temporal=True` 時 temporal block 共有幾個？插在哪幾個 `block_iter` 之後？（寫出算式與實際數列）
- **Q3.5** temporal block 為什麼不 emit intermediate？如果 emit 了，下游哪裡會壞？
- **Q3.6** temporal RoPE 給 register token 的位置索引是 0。這代表什麼效果？為什麼 camera token 不比照辦理？
- **Q3.7** 為什麼 temporal block 用 γ=0 而 gate predictor 用 zero-init last linear？兩者達成的是同一件事嗎？

**ledger**：T-01、T-02、T-03

---

## U4 — Gate 機制核心 ★

**目標**：這是整個 v3 的心臟。讀完你要能在紙上重畫 bias 怎麼進到 attention。

**讀**
- [vggt/models/aggregator.py:27-64](../../vggt/models/aggregator.py#L27-L64)（`GatePredictor`）
- `:354`、`:363`、`:380-383`、`:393`（觸發與回傳）
- `:419-462`（`_process_global_attention`，含 `:441` detach）
- `:464-573`（`_gated_global_block_forward`）—— **與 `vggt/layers/block.py` 逐行對照**
- 對照 [docs/method.md §3.1-3.4](../method.md#L161)

**盯**
- bias 公式 `clamp(log2 − softplus(g), max=0)`（`:541-543`）的三個性質
- frame-interleaved 切分（`:521-523`）與重組（`:557-560`）
- gate 在哪一個 block 之後才生效
- `gate_logits_override` 與模型自身 `gate_logits` 的**不對稱**

**題目**
- **Q4.1** 用 `predicted` gate（不給 override）時，24 個 global block 中**實際被加 bias 的是哪幾個**？寫出區間與個數。
- **Q4.2** 給了 `gate_logits_override`（oracle 消融）時，答案還一樣嗎？看 `:363` 的取值時機再答。這對「oracle Δ 是 predicted 的效能上界」這個論述有什麼影響？
- **Q4.3** 為什麼 bias 是 `clamp(log2 − softplus(g), max=0)` 而不是 `−softplus(g)`？各自在 `g=0` 時給出什麼值、對 warm-start 有什麼差別？`clamp` 的上界又擋掉什麼？
- **Q4.4** `clamp` 在 `g=0` 有 kink（不可微）。為什麼這不影響訓練？講出具體的那一行。
- **Q4.5** 寫出 `q_special` 與 `q_patch` 的 shape。為什麼不能直接寫 `q[:, :, :patch_start_idx]`？若這樣寫，實際上 gate 到了什麼？（這正是 2026-07-09 修掉的 bug）
- **Q4.6** `attn_bias` 的 shape 是 `[B, 1, 1, N]`。三個維度各自 broadcast 到什麼？為什麼 bias 只需要一個 key 維度、不需要 query 維度？
- **Q4.7** `_gated_global_block_forward` 從 `block` 物件上讀了哪些屬性？若有人改了 `vggt/layers/block.py` 的 `Block.forward`（例如加一個 norm），這裡會報錯還是靜默行為不同？
- **Q4.8** 兩條 path 的 dropout 與原本融合成一條的 path 相比，有什麼統計上的差異？在 `attn_drop.p=0` 的預設下重要嗎？
- **Q4.9** `enable_gate=False` 但呼叫端傳了 `gate_logits_override`，會發生什麼？`predictions` 裡會有 `gate_logits` 嗎？
- **Q4.10** 若 `gate_block_iter=30`（大於 `aa_block_num=24`），會發生什麼？有沒有任何錯誤訊息？

**ledger**：G-01 ~ G-09

---

## U5 — VGGT 外層與 predictions

**目標**：知道 8 個 `enable_*` flag 各自打開什麼、`predictions` 的每個 key 從哪來。

**讀**
- [vggt/models/vggt.py](../../vggt/models/vggt.py)（147 行，全讀）

**題目**
- **Q5.1** `enable_temporal` 這個 flag 實際上只改了一件事，是哪一行、改了什麼？
- **Q5.2** `predictions["gate_logits"]` 被寫在 `autocast(enabled=False)` 區塊之**前**（`:92-95`）。這代表它的 dtype 跟 `depth`/`pose_enc` 一樣嗎？對 `compute_gate_loss` 有什麼影響？
- **Q5.3** 哪個 prediction key 只在 `model.eval()` 時存在？誰在用它？
- **Q5.4** v3 的 scope 說「point head 關掉」。若某個 checkpoint 存了 `point_head` 權重、但建模型時 `enable_point=False`，載入時會怎樣？（提示：看 `eval_utils/vggt_infer.py` 的 `load_vggt_for_eval` / `graft_point_head`）
- **Q5.5** `motion_head` 與 gate 的 `σ(g)` 都能產出動態 mask。v3 選了哪個、為什麼另一個被關掉？

**ledger**：G-13

---

## U6 — Loss dispatcher 與 `L_gate` ★

**目標**：搞懂「哪個 loss 在什麼條件下才會被算」，以及 `L_gate` 的每一個細節。

**讀**
- [training/loss.py:29-147](../../training/loss.py#L29-L147)（`MultitaskLoss.__init__` + `forward`）
- `:702-757`（`compute_gate_loss`）、`:760-780`（`oracle_gate_logits_from_mask`）
- 對照 [docs/method.md §3.5.1、§6](../method.md#L244)

**盯**
- 每個分支的三重 guard：config 存在 + prediction key 存在 + batch key 存在
- pooling 沒有 assert；`hard_lo`/`hard_hi` 必須成對

**題目**
- **Q6.1** `enable_gate=True` 但 yaml 的 `loss.gate` 是 `null`，會發生什麼？反過來（`loss.gate` 有設定但 `enable_gate=False`）呢？兩種情況各自會不會報錯？
- **Q6.2** `compute_gate_loss` 的簽章裡有 `alpha_m` 與 `beta_m` 兩個參數。它們在函式體裡被用到了嗎？對應文件的哪一節？
- **Q6.3** `m_star_patch` 的 `P_h*P_w` 與 `gate_logits` 的 `P_patch` 若不相等（例如 H 不是 14 的倍數），會在哪一行、以什麼形式失敗？
- **Q6.4** 只設 `hard_lo=0.3`、不設 `hard_hi`，行為是什麼？程式會提醒你嗎？
- **Q6.5** hard-label 分支的 `denom` 為什麼要 `clamp_min(1.0)`？什麼情況下 `keep.sum()` 會是 0？
- **Q6.6** `oracle_gate_logits_from_mask` 對一個 pool 後值為 0.5 的邊界 patch，輸出什麼 logit？經過 bias 公式後得到多少？這算「靜態」還是「動態」？
- **Q6.7** `MultitaskLoss.forward` 對 `track` 這個分支做了什麼特別的事？為什麼是這樣寫？
- **Q6.8** 文件 §6 寫 `L = L_cam + L_depth + λ_g·L_gate`，但 run 1 的 config 裡 `depth: null`。誰對？這對「S1 會不會遺忘深度」的論述有什麼影響？

**ledger**：L-01 ~ L-05

---

## U7 — 其餘 loss 項

**目標**：`L_cam` 是 pose 的主要監督、`L_static_photo` 與 `L_camera_smooth` 是貢獻②③ 的全部。

**讀**
- [training/loss.py:150-266](../../training/loss.py#L150-L266)（camera）
- `:268-363`（point / depth）
- `:488-592`（`compute_static_photo_loss`）
- `:594-682`（`compute_camera_smooth_loss`）
- 快掃 `:365-486`、`:684-700`（v1/v2 遺留，v3 未開）
- 對照 [§4.2](../method.md#L338)、[§5.3](../method.md#L397)

**題目**
- **Q7.1** `compute_camera_loss` 的 `valid_frame_mask` 判準是什麼？文件 §6 特別說**不要**改成「靜態點數 > 100」，理由是什麼？程式現在是哪一種？
- **Q7.2** `gamma=0.6` 的跨 refine stage 加權，權重是遞增還是遞減到最後一個 stage？寫出指數。
- **Q7.3** `compute_static_photo_loss` 的梯度會流進哪些模組？GT depth、GT 內參、影像三者各自是常數還是可微？
- **Q7.4** static-photo 的遮擋防護具體怎麼做的？它擋得住「被靜態物體遮擋」嗎？
- **Q7.5** `compute_camera_smooth_loss` 用 `batch["ids"]` 做 Δt 正規化。若某個 clip 的 ids 有重複（Δt=0），程式怎麼處理？
- **Q7.6** 四元數的半球 sign-fix 在哪一行？不做會怎樣？
- **Q7.7** `camera_smooth` 罰的是幾階差分？它需要 `enable_temporal` 才能算嗎？（這關係到 run 2 的因子混淆能不能拆）

**ledger**：P-01 ~ P-03、T-04

---

## U8 — Trainer 主迴圈

**目標**：一個 step 到底發生什麼；梯度累積、scheduler、clip 的順序。

**讀**
- [training/trainer.py:483-556](../../training/trainer.py#L483-L556)（`run_train`）
- `:732-893`（`train_epoch`）、`:895-947`（`_run_steps_on_batch_chunks`）
- `:973-997`（`_process_batch`）、`:999-1026`（`_step`）

**盯**
- `where` 的計算（`:820-821`）與它餵給 scheduler 的意義
- `_process_batch` 就地覆寫了 batch 的哪些 key
- 模型 forward 只吃 `images`

**題目**
- **Q8.1** `model(...)` 的呼叫在 `:1014`。它傳了幾個參數？GT depth / extrinsics 有沒有進到模型？那它們是誰在用？
- **Q8.2** `where` 的公式是什麼？如果 `limit_train_batches` 設得比實際 dataloader 短，`where` 會不會到 1.0？scheduler 會怎樣？
- **Q8.3** 一個 epoch 實際看過多少個 sample？用 config 的哪幾個欄位算出來？（提示：不是 `len(dataset)`）
- **Q8.4** `_process_batch` 做了 normalization，改寫了哪些 key？為什麼要在 `autocast(enabled=False)` 下做？
- **Q8.5** oracle-gate 訓練模式（`:1008-1013`）啟動時，`L_gate` 監督的是模型自己的 gate 還是 oracle？兩者會不會打架？
- **Q8.6** NaN 防護在 `:943` 附近。它是丟掉那個 step 還是丟掉整個 batch？丟掉時 `steps` 計數會不會前進？
- **Q8.7** `scaler.unscale_` 與 `gradient_clipper` 的呼叫順序是什麼？順序反了會怎樣？

**ledger**：TR-04

---

## U9 — 驗證雙通道與 checkpoint

**目標**：搞懂 `best_ate.pt` / `best_loss.pt` / `last.pt` 三者的選點訊號各是什麼 —— 這直接決定你報的數字可不可信。

**讀**
- [training/trainer.py:558-577](../../training/trainer.py#L558-L577) + `:643-730`（channel A）
- `:579-641`（channel B）
- `:397-462`（save）、`:251-299`（resume）、`:133-150`（四種情況 guard）
- [training/config/default.yaml:15-28](../../training/config/default.yaml#L15-L28)

**題目**
- **Q9.1** channel A 與 channel B 各自跑什麼資料、什麼序列長度、產出什麼訊號？兩者有沒有任何一方會產出對方的指標？
- **Q9.2** `best_ate.pt` 的選點訊號是 Sintel ATE，而 Sintel 也是論文的報告資料集。這構成什麼問題？`best_loss.pt` 是怎麼迴避它的？
- **Q9.3** project memory 說「windowed `best.pt` 不可信、一律用 `last.pt`」。對照 `:534-552`，那個不可信的 `best.pt` 是被哪一條分支存下來的？現在的程式還會產生同樣的問題嗎？
- **Q9.4** channel B 用 `chunk_size=0`。若改成 32，ATE 會偏高還是偏低？機制上為什麼？
- **Q9.5** 沒帶 `--resume` 但 `logs/<exp>/ckpts/last.pt` 已存在，會發生什麼？這個設計擋的是哪一種事故？
- **Q9.6** checkpoint 存了 `epoch_completed` 與 `resume_iter`。從 epoch 中間崩潰後 resume，程式怎麼回到原位？dataloader 的隨機狀態有被還原嗎？
- **Q9.7** `run_pose_eval` 在多卡下只有 rank 0 真的算 ATE。其餘 rank 怎麼拿到同一個值？如果不做這件事會怎樣？

**ledger**：E-02 ~ E-04

---

## U10 — Config 層與凍結 ★

**目標**：把 yaml 的每一塊對回程式。**這一單元包含一個已確認的實作落差，自己讀出來。**

**讀**
- [training/config/default.yaml](../../training/config/default.yaml)（224 行）
- [training/config/inst_g.yaml](../../training/config/inst_g.yaml)（182 行，全讀，run 1 base）
- [training/train_utils/freeze.py](../../training/train_utils/freeze.py)（`freeze_modules` + `GLOB_FLAGS`）
- 對照 [docs/method.md §7.1、§7.3](../method.md#L458)

**盯**
- `frozen_module_names` 的 pattern 是對 **module name** 做 glob（`freeze.py:48-52`），不是對 param name
- `GLOB_FLAGS` 含 `DOTMATCH`，`*` 會吃掉 `.`

**題目**
- **Q10.1** pattern `"*aggregator.global_blocks.[0-7]*"` 會凍住哪幾個 global block？**逐一列出 0–23 的結果**。不要用直覺，用 `fnmatch` 的規則推。（推完可以用一行 python 驗證）
- **Q10.2** 承上，config 註解說「8..23 trainable」。實際 trainable 的是哪幾個？這對所有 S1 run 的結論有什麼影響？
- **Q10.3** 要正確凍住 0–7、放開 8–23，pattern 該怎麼寫？（至少給兩種寫法）
- **Q10.4** `camera_token` / `register_token` 是 `nn.Parameter` 而非 module。`frozen_module_names` 管得到它們嗎？程式怎麼處理這一類？
- **Q10.5** `_check_every_pattern_used` 在做什麼？如果你打錯一個 pattern（例如 `*depth_heads*`），會靜默略過還是報錯？
- **Q10.6** `inst_g_hard.yaml` 只有 29 行。它靠什麼機制拿到其餘設定？改一個 `resume_checkpoint_path` 就換一條 lineage，這在 ledger 上要記什麼？
- **Q10.7** 磁碟上有 S0 或 S2 的 config 嗎？`dyn_vggt_v3_s1_oracle_camera_only.yaml` 呢？誰在引用它？
- **Q10.8** `optim.gradient_clip.configs` 依 module group 分組。`aggregator.gate_predictor` 與 `camera_head` 用同一個 `max_norm=1.0`。gate 的梯度尺度跟 camera head 差幾個量級？（可從 TB 的 grad-norm meter 查）

**ledger**：TR-01 ~ TR-05

---

## U11 — Eval 管線

**目標**：論文數字是怎麼算出來的；哪些地方會靜默改變數字。

**讀**
- [training/eval_utils/paths.py](../../training/eval_utils/paths.py)（69 行）
- [training/eval_utils/vggt_infer.py](../../training/eval_utils/vggt_infer.py)（217 行，全讀）
- [training/eval_utils/metrics_pose.py](../../training/eval_utils/metrics_pose.py)、[metrics_depth.py](../../training/eval_utils/metrics_depth.py)
- [training/benchmark/eval_sintel.py](../../training/benchmark/eval_sintel.py)（170 行）

**題目**
- **Q11.1** `infer_sequence_chunked` 在 `len > chunk_size` 時，回傳的 dict 少了哪些 key？呼叫端會報錯還是靜默拿不到？
- **Q11.2** 同上，跨 chunk 的 global attention 被切斷。為什麼這會讓 ATE 變大而不是變小？
- **Q11.3** `load_vggt_for_eval` 怎麼決定要不要建 gate predictor / temporal block？如果 checkpoint 是 S0（只有 gate、沒 temporal），它建出來的模型長什麼樣？
- **Q11.4** `metrics_pose.eval_pose_metrics` 做的是哪一種對齊？`correct_scale=True` 代表報的 ATE 對什麼不敏感？
- **Q11.5** `metrics_depth` 有 median 與 lad2 兩種對齊。目前 Sintel benchmark 走哪一種？自由度差在哪？
- **Q11.6** `eval_sintel.py` 的 pose 用 14 個序列、depth 用 23 個。為什麼不一致？若你自己傳 `--seq_list`，這個區分還在嗎？
- **Q11.7** `default_output_dir(ckpt, TOOL)` 對 `logs/exp_a/ckpts/last.pt` 與 `checkpoints/VGGT-1B.pt` 分別解析出什麼 `<exp>`？為什麼要有這層？
- **Q11.8** trainer 的 channel B 呼叫 `evaluate(args, model=model)`。傳 `model` 與不傳有什麼差別？不傳的話 VRAM 會發生什麼事？

**ledger**：E-05 ~ E-07

---

## U12 — 診斷工具與 docs 對讀（收尾）

**目標**：知道每個結論是哪支 script 產的、該怎麼重跑；最後把 ledger 判定欄補完。

**讀**
- [training/diag/gate_bias_ablation.py](../../training/diag/gate_bias_ablation.py)（348 行）
- [training/diag/gate_quality.py](../../training/diag/gate_quality.py)（200 行）
- [training/eval_utils/gate_common.py](../../training/eval_utils/gate_common.py)（82 行）
- 快掃 `diag/vis/error_growth.py`、`gate_temporal.py`
- 最後回頭讀 [docs/method.md](../method.md) §3 全節 + §8

**題目**
- **Q12.1** `gate_bias_ablation.py` 的四個 mode（`off`/`predicted`/`oracle`/`predicted_xT`）各自傳什麼給 `gate_logits_override`？`off` 為什麼不是「不傳 override」？
- **Q12.2** 承 Q4.2：`off` mode 傳的是一個 uniform 大負值 override。那 `off` 這條 baseline 的 24 個 global block 有幾個被加了 bias？它跟「完全不開 gate 的 base model」是同一件事嗎？
- **Q12.3** `gate_common.oracle_logits_from_masks`（numpy）與 `loss.oracle_gate_logits_from_mask`（torch）刻意不做成同一份。差在哪？兩者的數字可以直接互相比較嗎？
- **Q12.4** `gate_quality.py` 的 macro 與 micro 聚合差在哪？§3.6 表格報的是哪一種？
- **Q12.5** `error_growth.py` 的 `chunk_size` 必須 ≥ 序列長度。若忘了設，圖會長什麼樣、你會得出什麼錯誤結論？
- **Q12.6** §3.6(a) 的 oracle Δ 在 f50 歸零、f12 是 −15%。結合 Q4.2 與 Q12.2 的答案，你現在對這個現象有沒有新的候選解釋？（開放題，寫進 ledger 備註）

**ledger**：G-14、E-08，並回頭補完所有未填欄位

---

## 讀完之後

1. ledger 中所有標記為「被簡化 / 自作主張 / 未實作」的條目，按「會不會影響已發表數字」排序。
2. 影響數字的 → 開 issue 或直接修，並在 `docs/table.md` 的「世代地圖」加一個世代。
3. 不影響數字的 → 決定是「補實作」還是「改文件讓它誠實」。
