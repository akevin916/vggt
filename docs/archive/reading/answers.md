# 自我驗證題 —— 解答

> **先做完 [units.md](units.md) 的題目再看。** 每題答案附 `file:line` 佐證，你可以自己去核。
> 標 ⚠️ 的是**已確認的實作落差**（程式與文件/註解不一致），不是「你讀錯了」。

---

## U0

**Q0.1** `motion_mask`。缺標註時被 `torch.zeros_like(depths)` 零填充（[composed_dataset.py:156-158](../../training/data/composed_dataset.py#L156-L158)），理由是讓 `default_collate` 能把 PO（有 mask）與 TartanAir（無）混在同一個 batch。
副作用：`L_gate` 會在這些幀上要求 `σ(g)→0`（全靜態監督）。TartanAir 本來就是靜態場景，語義正確；但任何「缺檔」造成的零填充會**偽裝成合法的全靜態標籤**，兩者在 batch 裡完全無法區分（見 Q2.1）。
（附帶澄清：§3.6 引用的 `dyn_frac = 0.202` 是 **Sintel eval 端**的 micro 統計，見 [table.md:299](../table.md#L299)，不是訓練資料混合後的比例。要驗證「class imbalance 是不是 `L_gate` 的病因」，需要的是**訓練標籤**的正例比例，而那個數字目前沒有人算過。）

**Q0.2** `scene_flow_gt` 在 dataset 有提供時才加（`:159-160`）；`tracks`/`track_vis_mask`/`track_positive_mask` 只在 `self.load_track` 為真時（`:163-197`）。

**Q0.3** 論文數字 → `training/benchmark/`，輸出 `outputs/eval_sintel/<exp>/`；臨時探索 → `training/diag/`（畫圖再放 `diag/vis/`），輸出 `outputs/<tool>/<exp>/`。路徑一律由 `default_output_dir(ckpt, TOOL)` 解析（[paths.py:64-66](../../training/eval_utils/paths.py#L64-L66)），不要自己拼。

**Q0.4** 因為每一列的 `off` 是**同一個 ckpt、同一組序列、同一次 inference** 下的無 gate 對照；跨 ckpt 共用一個 base 分母會把「ckpt 之間的差」與「gate 的效果」混在一起。

**Q0.5** 防「診斷章節各自下一個結論，彼此矛盾又沒人發現哪個是最新的」。banner 是唯一的「現在相信什麼」來源，其餘章節只放觀測值。

---

## U1

**Q1.1** depth 用 `cv2.INTER_NEAREST`（[dataset_util.py:233-238](../../training/data/dataset_util.py#L233-L238)），image 走一般 resize（線性）。深度/mask 不能線性內插，否則會在物體邊界造出不存在的中間深度值與 0/1 之間的假 mask 值。
**mask 的 aug 同步方式**：不是「多接一個通道」，而是 **RNG state 重放**（[pointodyssey.py:205-223](../../training/data/datasets/pointodyssey.py#L205-L223)）：先 `np.random.get_state()` 存檔 → 正常跑一次 `process_one_image(image, depth)` → 把 RNG 還原 → **再跑一次** `process_one_image(image, motion_gt)`，把 mask 當成偽 depth（因此吃到 `INTER_NEAREST`）→ 最後 `np.random.set_state(post_state)` 還原全域 RNG 流。輸出再用 `> 0.5` 重新二值化。
這個做法的脆弱點：它假設 `process_one_image` 對 image 與 depth 消耗**完全相同的隨機數序列**。若哪天有人在 image-only 的分支加一個隨機操作，mask 就會跟 depth 錯位，而且**不會報錯**。

**Q1.2** 由 `cojitter` + `cojitter_ratio` 控制（[composed_dataset.py:132-139](../../training/data/composed_dataset.py#L132-L139)）：`cojitter` 開且隨機數 > `cojitter_ratio` 時整段共用一組 jitter，否則**逐幀獨立**。
對 `L_static_photo`：逐幀獨立 jitter 會讓「warp 過去的 RGB 應該相等」這個假設失效 —— 同一個世界點在 t 與 t+1 被施了不同的色彩變換。這是 photo run 負結果的候選解釋之一，值得查該 run 的 `cojitter` 設定。

**Q1.3** `DynamicBatchSampler` 以 `max_img_per_gpu`（預設 48）為預算：先抽一個 `img_num`，再 `batch_size = max_img_per_gpu / random_image_num`（[dynamic_dataloader.py:173](../../training/data/dynamic_dataloader.py#L173)）→ **4 幀的 batch 會有較大的 batch size，16 幀的較小**，總影像數大致固定在 48。

**Q1.4** 有，但是間接的：靠 config 的 `data.train.common_config.img_nums` 下界 ≥ 4。程式端 `compute_camera_smooth_loss` 自己也擋（`S < 3` 直接回 0，[loss.py:630-632](../../training/loss.py#L630-L632)），但**不會警告**——若你把 `img_nums` 設成 `[2,3]`，這個 loss 會靜默恆為 0。

**Q1.5** 允許（`get_nearby_ids` 以 `replace=True` 抽樣）。`compute_camera_smooth_loss` 用 `batch["ids"]` 算 `dt`，重複幀給 `dt=0`，由 `pair_valid = dt > 0` 排除、`safe_dt = dt.clamp(min=1.0)` 防除零（[loss.py:635-638](../../training/loss.py#L635-L638)）。

---

## U2

**Q2.1** 拿到 `dynmask_raft/` 的 RAFT 殘差 mask；連 raft 也沒有就是**全零**（[pointodyssey.py:140-146](../../training/data/datasets/pointodyssey.py#L140-L146)）。
**你從 log 察覺不到** —— fallback 沒有任何 log / counter，且零填充的 mask 與「這幀真的全靜態」無法區分。要察覺得到，最小改動是在 dataset `__init__` 掃一次序列、統計缺 `dynmask_inst/` 的序列數並 `logging.warning`，或在每個 epoch 記一個 `frac_label_fallback` meter。**這是 ledger G-10 的核心。**

**Q2.2** 以 GT world track（`trajs_3d`）判斷每個 instance 連通塊內被追蹤點的世界位移是否超過門檻 `ITHR`（絕對值，見 `po_instance_dynmask.py`），動則整塊填滿。風險：門檻是絕對距離、不隨場景尺度或幀間距調整；且「整塊填滿」會把一個部分靜止的大 instance 全標成動態。

**Q2.3** 解 `cave_2` 前半段快速鏡頭：`f_gt` 本身很大時，固定絕對門檻會讓整片背景的殘差都超標 → mask 灌爆。只留 `abs_thr` 就會回到那個狀態（§3.5.2 明說）。實作 `derive_motion_mask(rel_threshold=α)`（[motion_mask.py:81](../../training/data/motion_mask.py#L81)）。

**Q2.4** `compute_ego_flow` 需要 depth、內參、相鄰兩幀的相對 pose —— **全部用 GT**（[motion_mask.py:49](../../training/data/motion_mask.py#L49)）。所以 `m_geo` 只能在有 GT pose+depth 的資料上離線算，可以當**測試端標籤**，但不能當「訓練時即時算」的信號（那需要可微的預測量，即 §3.7 未實作的輸入通道）。

**Q2.5** PO 的 `masks/` 是**逐 instance 的外觀分割**（非黑=前景）：牆、地板、靜止的前景角色全被標。它標的是「這是前景物件」，不是「這東西在動」。用它監督 → gate 學到「PO 室內外觀」，跨到 Sintel 崩（AUC 0.584 → 0.767 的那組對照，§3.5.2）。

**Q2.6** 是**設計**（[composed_dataset.py:154-155](../../training/data/composed_dataset.py#L154-L155) 的註解說明）：靜態場景的正確標籤本來就是全零，提供「負例」讓 gate 不要到處亂開。但它同時把 Q0.1 的「缺檔零填充」偽裝掉了 —— 兩件語義不同的事共用同一個表示。

**Q2.7** cache 落在 `<sintel_root>/mask/`，權威來源是 `meta.json`（[motion_mask.py:107-133](../../training/data/motion_mask.py#L107-L133)）。若 `meta.json` 記錄的參數與當前呼叫不符，應該要 re-derive；**先確認你手上的版本是不是真的比對了 `rel_thr`** —— 若只比對了檔案存在與否，改參數不清 cache 會靜默拿到舊 mask。

---

## U3

**Q3.1**
| attention | shape | 誰跟誰互看 |
|---|---|---|
| frame | `(B*S, P, C)` | 單幀內所有 token 互看 |
| global | `(B, S*P, C)` | 全部幀的全部 token 互看 |
| temporal | `(B*P, S, C)` | 同一空間位置沿 S 幀互看 |
分別在 [aggregator.py:400](../../vggt/models/aggregator.py#L400)、`:428`、`:584`。

**Q3.2** 第 s 幀的 camera token 在 `s * P + 0`（P 是每幀 token 數 = `patch_start_idx + P_patch`）。`tokens[:, :patch_start_idx]` 取到的是**第 0 幀的 camera+register token**，不是所有幀的 special token —— 這正是 Q4.5 的 bug。

**Q3.3** 因為 VGGT 的 pose 是相對於**第 0 幀**定義的，第 0 幀的 camera token 角色不同（參考幀），所以用一組獨立的可學參數（[aggregator.py:598-621](../../vggt/models/aggregator.py#L598-L621)、`:222-223` 的 `(1, 2, 1, embed_dim)`）。

**Q3.4** `n_temporal = aa_block_num // temporal_every = 24 // 3 = 8`（`:184`）。插入條件 `block_iter % 3 == 2`（`:371`）→ 在 `block_iter = 2, 5, 8, 11, 14, 17, 20, 23` 這 8 個 iteration 內執行。
注意順序：`aa_order = [frame, temporal, global]`，所以 temporal 跑在該 iteration 的 frame 之後、global 之前。

**Q3.5** head 的輸入是 `cat([frame_intermediate, global_intermediate], dim=-1)` = `2C`（`:387`）。若 temporal 也 emit，維度會變 `3C`，所有 DPT/camera head 的 `dim_in=2*embed_dim` 都要改，就不能載 pretrained 權重了。

**Q3.6** register token 給位置 0 → 1D RoPE 的旋轉角為 0 → **恆等變換**，等於不施加時間位置編碼。camera token 不比照，是因為它要能區分「我是第幾幀的相機」——那正是 pose 需要的資訊（`:337-346`）。

**Q3.7** 目的相同（init 時是 no-op、可 byte-for-byte 載 pretrained），機制不同：
- temporal：`ls1.gamma = ls2.gamma = 0`（`:203-207`）→ 整個 block 的殘差輸出為 0 → **identity map**。
- gate：`linear2` 的 weight+bias 零初始化（`:51-52`）→ `g ≡ 0` → bias 公式在 g=0 給**精確 0** → attention 不變。
兩者都是「加了東西但 init 時等於沒加」，但 gate 那條**依賴 bias 公式在 g=0 為 0 這個性質**（見 Q4.3）。

---

## U4 ★

**Q4.1** gate 在 `block_iter == gate_block_iter == 7` 的**整個 aa_order 跑完之後**才計算（[aggregator.py:380-383](../../vggt/models/aggregator.py#L380-L383) 在 `for attn_type` 迴圈**之外**）。所以 `block_iter` 0–7 的 global block（`global_idx` 0–7）拿到的 `gate_logits` 仍是 `None`，未被 gate。
→ **被加 bias 的是 global block 8–23，共 16 個**。

**Q4.2** ⚠️ **不一樣。** `gate_bias_source` 在每次 global attention 前重新取值（`:363`），`gate_logits_override` 從第一個 iteration 起就非 `None` → **override 模式下 24 個 global block 全部被加 bias**。
影響：`oracle` 消融比 `predicted` 多 gate 了 8 個 block（0–7），**兩者不是同一個機制的兩種輸入品質**，而是兩種不同的架構配置。「oracle Δ = predicted 的效能上界」這個論述因此不成立 —— oracle 的一部分 gain 可能來自「早八個 block 就開始 gate」，而非「標籤更準」。
**這是 §3.6 那個「oracle 有效但 predicted ≈ off」現象最直接的新候選解釋，需要一個 `oracle_from_block8` 的對照 run 才能拆開。**（ledger G-05）

**Q4.3**
```
bias = clamp(log(2) − softplus(g), max=0)      # aggregator.py:541-543
```
- `g=0`：`softplus(0) = log 2` → `bias = clamp(0, max=0) = 0`，**精確為 0**。純 `−softplus(g)` 在 `g=0` 給 `−log2 ≈ −0.693`，等於把每個 patch key 相對 special key 的 logit 打掉 0.693（權重約減半）→ zero-init 就已經不等於 pretrained VGGT，warm-start 破功。
- `clamp(max=0)`：擋掉 `g<0`（確信靜態）時 `log2 − softplus(g) > 0` 的正 bias，維持 **suppress-only**（只壓不升）。
- `g→+∞`：`bias → log2 − g → −∞`，與 `−softplus(g)` 行為一致。

**Q4.4** 因為傳進來的是 `gate_logits.detach()`（[aggregator.py:441](../../vggt/models/aggregator.py#L441)），bias 路徑上沒有梯度；gate 只由 `L_gate` 的 BCE 訓練。kink 只影響「若有梯度會怎樣」，前向值本身連續。

**Q4.5**
```
q_special: [B, H, S * patch_start_idx, D]     # aggregator.py:522
q_patch  : [B, H, S * P_patch,        D]     # aggregator.py:523
```
不能寫 `q[:, :, :patch_start_idx]`，因為 global 佈局是 **frame-interleaved**：special token 散在每幀開頭。舊版寫 `q[:, :, :n_special]`（`n_special = patch_start_idx * S`）取到的索引因 `n_special ≪ P` 而**全部落在 frame 0 之內** → 只 gate 到第 0 幀的 camera token，其餘幀完全沒有 gate。這是 2026-07-09 `e064087` 修掉的 bug，也是 2026-07-06 之前所有結論作廢的原因（§3.4 註記）。

**Q4.6** `[B, 1, 1, N]`：第 2 維 broadcast 到所有 head，第 3 維 broadcast 到所有 special query，第 4 維是 key 位置。
不需要 query 維度，是因為**所有 special query 共用同一組 key bias** —— 「這個 patch 有多動態」不隨「是哪一個 camera/register token 在問」而變。

**Q4.7** 讀了 `block.norm1`、`block.norm2`、`block.mlp`、`block.ls1`、`block.ls2`、`block.attn.{qkv, q_norm, k_norm, rope, proj, proj_drop, attn_drop, num_heads, head_dim}`（`:502-571`）。
若有人在 `Block.forward` 加一個 norm 或改殘差結構，這裡**不會報錯**（屬性都還在），只會**靜默地跑一條與 ungated path 不同的計算** → gated 與 ungated 兩條路徑行為分歧、且只在開 gate 時發作。**這是最需要一個 unit test 釘住的地方**（ledger G-06）。

**Q4.8** 兩條 path 各自呼叫一次 `F.scaled_dot_product_attention`，dropout 是**兩次獨立抽樣**（`:549`、`:551-555`），而融合版是一次。預設 `attn_drop.p = 0`（VGGT 沒開 attention dropout），所以實務上無差異；但若哪天有人打開它，gated 與 ungated 的正則化強度會不同。

**Q4.9** `enable_gate=False` → `self.gate_predictor is None` → `gate_logits` 恆為 `None`，`predictions` **沒有** `gate_logits` key（[vggt.py:92-93](../../vggt/models/vggt.py#L92-L93)）。
但 `:363` 的 `gate_bias_source` 仍會取到 override → **bias 照樣施加在全部 24 個 global block**。這正是 `diag/gate_bias_ablation.py --force_gate` 拿 pretrained VGGT-1B 做 oracle 消融時走的路徑。

**Q4.10** `block_iter` 永遠不會等於 30（迴圈上界是 `aa_block_num = 24`）→ `gate_logits` 全程 `None` → 沒有 gate、`predictions` 沒有 `gate_logits` → `MultitaskLoss` 的 gate 分支因 key 不存在而**靜默跳過**。**全程無任何錯誤或警告**（ledger G-07）。

---

## U5

**Q5.1** [vggt.py:30](../../vggt/models/vggt.py#L30)：`aa_order = ["frame","temporal","global"] if enable_temporal else ["frame","global"]`。其餘全由 aggregator 依 `aa_order` 自行決定（是否建 `temporal_blocks`、是否插入）。

**Q5.2** 不一樣。`gate_logits` 在 `autocast(enabled=False)` 之**外**寫入（`:92-95`），所以帶著 autocast 的 dtype（bf16）；`pose_enc`/`depth` 等在區塊內、是 fp32。
對 `compute_gate_loss` 的影響：`m_star_patch` 被 `.to(gate_logits.dtype)` 轉成 bf16 才算 BCE（[loss.py:754](../../training/loss.py#L754)）→ 軟標籤（例如 0.0714 這種 1/14 的值）在 bf16 下只有約 3 位有效數字。對 AUC 無妨，但**別拿 BCE 的絕對數值做精細比較**（這與「用 AUC/F1 不用 BCE」的準則剛好同向）。

**Q5.3** `predictions["images"]`，只在 `not self.training` 時寫入（`:143-144`）。給視覺化用。

**Q5.4** `load_state_dict(..., strict=False)`（[vggt_infer.py:69](../../training/eval_utils/vggt_infer.py#L69)）→ 多出來的 `point_head.*` 進 `unexpected`，**不報錯只印數字**。反過來 v3 ckpt 沒有 point head 時，要用 `force_point=True` 建一個隨機初始化的頭，再用 `graft_point_head` 從 VGGT-1B 借權重（`:82-106`）—— 但那個頭讀的是已經漂移過的 trunk，只能當 indicative（該 docstring 自己有警告）。

**Q5.5** 選 gate 的 `σ(g)`。理由是 gate 本來就要算（bias 需要），mask 是**免費副產物**；`motion_head` 是完整的 DPT head，額外顯存與參數，而 v3 的 scope 是「不動幾何頭、省顯存」（§7.2 表：point/track/flow/motion 全關）。

---

## U6 ★

**Q6.1**
- `enable_gate=True` + `loss.gate: null` → 模型算 gate、bias 照施加，但**沒有任何監督**。gate predictor 被凍在 zero-init（若也在 unfreeze 清單裡，則是隨機游走）→ `σ(g) ≡ 0.5`、bias 恆為固定值。**不報錯。**
- `loss.gate` 有設定 + `enable_gate=False` → `predictions` 沒有 `gate_logits` → dispatcher 的三重 guard 讓該分支**靜默跳過**（[loss.py:119](../../training/loss.py#L119)）。**不報錯。**
兩種都是靜默失敗，只能從 TB 上 `loss_gate` 這條線消失/不動看出來。（ledger L-01）

**Q6.2** ⚠️ **簽章有、函式體完全沒用到**（[loss.py:702-703](../../training/loss.py#L702-L703) vs `:725-756`）。它們是 §3.7 未實作的 `m_geo_soft = σ(α_m·(‖f_gt − f_cam‖ − β_m))` 留下的殘骸。若你在 yaml 裡設 `alpha_m: 20`，**什麼事都不會發生**。（ledger L-02）

**Q6.3** 不會在 pooling 那行失敗 —— `adaptive_avg_pool2d` 會照 `(P_h, P_w)` 產出（`:736-739`）。失敗發生在下一步 BCE 的 broadcast：`gate_logits [B,S,P_patch]` vs `target [B,S,P_h*P_w]` 形狀不符 → RuntimeError。
**程式沒有 assert**（`P_patch` 在 `:728` 被解包出來卻從未與 `P_h*P_w` 比對）。若兩者剛好能 broadcast（例如其一為 1），會靜默算錯。（ledger L-03）

**Q6.4** 條件是 `hard_lo is not None and hard_hi is not None`（`:741`）→ 只設一個會**靜默走回 soft label 分支**，沒有警告。（ledger L-04）

**Q6.5** `keep.sum()` 為 0 表示整個 batch 的 patch 全落在 ignore band `(hard_lo, hard_hi)` 內。`clamp_min(1.0)` 防除零（`:749`），此時 loss = 0。以 `hard_lo=0.3, hard_hi=0.7` 而言極不可能，但若有人設成 `0.01/0.99` 就會頻繁發生。

**Q6.6** `(0.5 * 2 − 1) * 30 = 0` → logit 為 0 → bias = `clamp(log2 − softplus(0), max=0) = 0` → **完全不壓抑，等同靜態**。
也就是說 oracle 對「一半動一半靜」的邊界 patch 是**放行**的，不是折衷壓抑。這在物體邊界多的序列上會系統性偏向「不 gate」。

**Q6.7** `if "track" in predictions: raise NotImplementedError`（`:142-143`）—— 不是靜默跳過而是**主動爆炸**，因為 track loss 的舊實作（`:1189-1311`）還沒清乾淨，寧可讓人撞到也不要拿錯的東西算。

**Q6.8** 程式對。run 1 的 config 是 `depth: null`（[inst_g.yaml:123](../../training/config/inst_g.yaml#L123)），且 `depth_head` 也被凍（`:147`）。
影響：§7.1 說 S2 要靠「`depth_head` 凍住但 `L_depth` 保留」當防遺忘錨 —— **S1 完全沒有這個錨**。trunk（global 8–23）在只有 `L_cam + L_gate` 的情況下自由漂移，凍住的 depth head 讀到漂移過的特徵。實測 run1 的 AbsRel 反而從 0.2747 → 0.2136（[vggt_infer.py:88-92](../../training/eval_utils/vggt_infer.py#L88-L92) 註記），所以沒有明顯壞掉，但那是運氣不是設計。（ledger L-05）

---

## U7

**Q7.1** `valid_frame_mask = point_masks[:, 0].sum(dim=[-1,-2]) > 100`（[loss.py:164-166](../../training/loss.py#L164-L166)）—— 有效**深度點**數 > 100，寬鬆版。程式就是文件建議的那一種，一致。
文件反對嚴格版（靜態點 > 100）的理由：多動態場景會把整幀丟掉，而 pose 監督是直接的（`pose_enc` 對 `pose_enc`），動態像素本來就污染不了它，丟幀是純損失。

**Q7.2** `stage_weight = gamma ** (n_stages - stage_idx - 1)`（`:642`，`compute_camera_smooth_loss`；`compute_camera_loss` 同慣例）→ 最後一個 stage 權重 `γ⁰ = 1`，越早的 stage 越小。**遞增至最後 stage**。

**Q7.3** 只有**預測 extrinsics** 拿到梯度（經 `pose_encoding_to_extri_intri` → 投影 → `grid_sample` 的 sampling grid，[loss.py:527-570](../../training/loss.py#L527-L570)）。GT depth（`batch["depths"]`）、GT 內參、影像都是常數，`depth_head` 完全不參與。內參也用 GT（`:525-526` 的註解說明：讓 loss 只隔離 R/T 誤差，FoV 已由 `L_cam` 直接監督）。

**Q7.4** 把 t+1 的 `motion_mask` 用同一個 grid `grid_sample` 過去，若取樣點落在動態區（`sampled_dyn >= dyn_thresh`）就丟棄（`:574-577`）。
**擋不住靜態遮擋**：一個被靜態柱子擋住的背景點，warp 過去會落在柱子上，`motion_mask` 是 0（靜態）→ 通過檢查 → 拿柱子的 RGB 去比背景的 RGB，產生假誤差。要擋需要 depth 一致性檢查（z-buffer），程式沒有做。（ledger P-02）

**Q7.5** `pair_valid = dt > 0` 排除該對；`safe_dt = dt.clamp(min=1.0)` 讓除法不炸；二階項用 `accel_valid = pair_valid[:,1:] & pair_valid[:,:-1]`（`:635-638`）→ 只要相鄰兩個速度中有一個無效，該加速度就不計入。

**Q7.6** `:647-653`。不做的話 `q` 與 `−q` 表示同一個旋轉，但差分會看到一個巨大的假跳動 → 平滑項會去「修」一個根本不存在的抖動。
注意這是**逐分量**的簡化平滑，不是測地線（docstring `:612-613` 自己標明）。

**Q7.7** 二階（加速度）差分。**不需要 `enable_temporal`** —— 它只讀 `predictions["pose_enc_list"]`，跟 aggregator 結構無關（`:621`）。
所以 run 2 的兩個因子（temporal 解凍 / camera_smooth loss）在技術上完全可以拆成兩個獨立 run，§5.4 說的「因子混淆」是實驗設計問題，不是實作限制。

---

## U8

**Q8.1** 兩個：`images` 與 `gate_logits_override`（[trainer.py:1014](../../training/trainer.py#L1014)）。GT depth/extrinsics/intrinsics/mask **完全沒有進模型**，全部只被 `self.loss(y_hat, batch)` 消費（`:1017`）。
這件事很重要：任何「模型偷看 GT」的懷疑，看這一行就能否定。

**Q8.2** `where = (epoch + data_iter / limit_train_batches) / max_epochs`（`:820-821`）。`limit_train_batches` 就是分母，所以 `where` 一定會走到 ~1.0；它與真實 dataloader 長度無關。
反過來說：**改 `limit_train_batches` 會同時改變「一個 epoch 多少資料」與「lr schedule 的推進速度」**，兩個效果綁在一起。

**Q8.3** `limit_train_batches × 每個 batch 的 sample 數`，而每個 batch 的 sample 數由 `max_img_per_gpu / img_num` 動態決定（Q1.3）→ **實際看過的 sample 數在 epoch 之間會浮動**。`len(dataset)`（由各 dataset 的 `len_train` 決定）只影響 sampler 的抽樣分布權重，不決定 epoch 長度。

**Q8.4** 改寫 `extrinsics`、`cam_points`、`world_points`、`depths`，以及存在時的 `scene_flow_gt`（`:973-997`）—— 全部歸一化到以 frame 0 為參考、尺度正規化的座標系。
在 `autocast(enabled=False)` 下做（`:802`）是因為這是幾何運算（矩陣求逆、尺度統計），bf16 精度不足會累積誤差。

**Q8.5** `L_gate` 監督的仍是**模型自己的** `gate_logits` —— override 只換 bias，`predictions["gate_logits"]` 是模型自身輸出（[aggregator.py:363 vs :393](../../vggt/models/aggregator.py#L363)）。
兩者不打架，但要清楚這代表什麼：oracle 訓練模式下，**pose 路徑看到完美 mask，gate predictor 仍在獨立學習** —— 這正是「隔離架構賭注與 gate 品質」的設計意圖。

**Q8.6** ⚠️ `if not math.isfinite(loss.item()): logging.error(...); return`（[trainer.py:939-943](../../training/trainer.py#L939-L943)）。訊息說 "attempting to stop training"，但實際上只是從 `_run_steps_on_batch_chunks` **return** → 跳過該 batch 剩餘 chunk 的 backward 與這一輪的 optimizer step，**訓練繼續跑**。
且 `self.steps[phase]` 已經在 `_step` 裡加過了（`:1025`），meter 也已更新 → **step 計數會前進但沒有梯度更新**。訊息與行為不符。（ledger TR-05）

**Q8.7** `scaler.unscale_(optim.optimizer)` → `gradient_clipper(model)` → `scaler.step()` → `scaler.update()`（`:856-870`）。
順序反了（先 clip 再 unscale）會對**被 GradScaler 放大過的梯度**做 clip，`max_norm=1.0` 這個門檻就完全失去意義（等效門檻變成 `1.0 / scale`，而 scale 是動態的）。

---

## U9

**Q9.1**
| | channel A (`run_val`/`val_epoch`) | channel B (`run_pose_eval`) |
|---|---|---|
| 資料 | held-out PointOdyssey test | Sintel（14 seq pose / 23 seq depth）|
| 長度 | windowed，`img_nums [8,8]` | **全序列**，`chunk_size=0` |
| 產出 | **只有 loss**（`Loss/val_*`）| **只有 metric**（`validation/{pose,depth}_*`）|
| 驅動 | `best_loss.pt` | `best_ate.pt` |
[trainer.py:643-730](../../training/trainer.py#L643-L730) vs `:579-641`。兩邊都不產生對方的訊號 —— 這是 commit `34698ea` 的重點。

**Q9.2** `best_ate.pt` 用報告資料集自己選點 → **選擇偏差**：報出來的 ATE 是「在報告集上跑了 N 個 epoch 取最小值」，樂觀偏移量級見 §8.2。
`best_loss.pt` 用的是 PO-test 的 camera loss，與 Sintel 完全無關 → 選點訊號沒碰過報告資料，是無偏的挑法（代價是它未必挑到 Sintel 上最好的）。

**Q9.3** 那個不可信的 `best.pt` 來自**舊版**的 channel A：當時 channel A 是 windowed Sintel、會產出 ATE 並用它選 `best.pt`。windowed（短片段）ATE 與全序列 ATE 排序不一致 → 選出 ep17（全序列 0.1651）而 `last.pt` 是 0.1253。
現在的程式**不會**再犯同一個錯：`best_ate.pt` 走的是全序列 channel B（`:530, 536-542`）。但 `logs/` 裡既有的舊 `best.pt` 仍是舊規則的產物，**不能拿來比較**。

**Q9.4** 偏高（ATE 變大）。`infer_sequence_chunked` 把序列切成獨立的 forward pass（[vggt_infer.py:201-207](../../training/eval_utils/vggt_infer.py#L201-L207)），跨 chunk 的 global attention 被切斷，每個 chunk 各自以自己的 frame 0 為參考重建，接起來時累積接縫誤差。

**Q9.5** 直接 raise、拒絕啟動（[trainer.py:133-150](../../training/trainer.py#L133-L150)）。擋的是「忘了加 `--resume`，於是從 warm-start 權重重新開始並把跑到一半的 `last.pt` 覆蓋掉」。反向也擋：帶了 `--resume` 但沒有 `last.pt` 也會 raise。

**Q9.6** checkpoint 存 `epoch_completed`（= `data_iter is None`）與 `resume_iter`（`:432-443`）。載入時：`epoch_completed=False` → 回到**同一個 epoch**並設 `_resume_iter`；否則 `prev_epoch + 1`（`:279-291`）。`train_epoch` 開頭用 `_resume_iter` **快轉跳過**前面的 batch（`:793-795`）並在讀取後歸零（`:781-782`）。
**dataloader 的隨機狀態沒有真的還原** —— 是靠 `set_seeds(seed + epoch*100)`（`:496`）讓同一個 epoch 重放同樣的序列，再用計數快轉。所以 resume 後的資料順序一致，但**快轉是真的把那些 batch 讀出來丟掉**（有 I/O 成本）。

**Q9.7** `dist.broadcast(ate_val, src=0)`（`:637-638`）。不做的話各 rank 的 `best_ate` 不同 → 有的 rank 存 checkpoint、有的不存 → DDP 的 checkpoint 寫入不一致，最壞情況是壞檔。

---

## U10 ★

**Q10.1 / Q10.2** ⚠️ **這是本次重讀最重要的發現。**
`freeze_modules` 對 **module name** 做 glob（[freeze.py:48-52](../../training/train_utils/freeze.py#L48-L52)），且 `GLOB_FLAGS` 含 `DOTMATCH`（`*` 會吃掉 `.`）。pattern `"*aggregator.global_blocks.[0-7]*"` 對 `aggregator.global_blocks.10` 的比對是：`[0-7]` 吃掉 `1`，尾部的 `*` 吃掉 `0` → **match**。

實測（`wcmatch.fnmatch`，專案環境）：
```
FROZEN   : 0 1 2 3 4 5 6 7 10 11 12 13 14 15 16 17 18 19 20 21 22 23
TRAINABLE: 8 9
```
→ **所有 S1 run 實際上只訓練了 `global_blocks.8` 與 `global_blocks.9`**（外加 `gate_predictor`、`camera_head`、`camera_token`/`register_token`），不是 config 註解宣稱的 8–23。
影響範圍：run 1 / hard / photo / smooth_temporal / photo_smooth_temporal 全部；§7.1 的 S1 定義、§7.3 的「統一架構」、以及所有 clean 世代的 pose 數字。
`training/logs/inst_g/log.txt:4` 可確認實際使用的就是這個 pattern。（ledger **TR-01**，最高優先）

**Q10.3** 兩種寫法：
```yaml
# (a) 列舉（最不會出錯）
- "*aggregator.global_blocks.0*"   # 但 0 也會吃到 0（沒有 00-07 以外的），需逐一寫 .0 .1 ... .7 並用邊界
# (b) 用 extglob 明確界定 index 邊界（GLOB_FLAGS 已開 EXTMATCH / SPLIT）
- "*aggregator.global_blocks.@(0|1|2|3|4|5|6|7)"          # 只凍 block 本身
- "*aggregator.global_blocks.@(0|1|2|3|4|5|6|7).*"        # 連同子模組
```
關鍵是**讓 index 後面要嘛是字串結尾、要嘛是 `.`**，不能讓尾部的 `*` 自由吃掉後續數字。修完務必用 `named_modules()` 列一次 `requires_grad` 的實際結果驗收，不要再相信 pattern 的直覺。

**Q10.4** 管不到 —— `freeze_modules` 走 `model.named_modules()`，`nn.Parameter` 不是 module。trainer 另外有一段 `frozen_param_names` 做**子字串**比對來處理這類參數（[trainer.py:340-347](../../training/trainer.py#L340-L347)）。v3 的 config 沒有用它 → `camera_token` / `register_token` 是**可訓練**的。

**Q10.5** `_check_every_pattern_used(matched, patterns)`（[freeze.py:54](../../training/train_utils/freeze.py#L54)）檢查每個 pattern 至少 match 到一個 module，打錯字會被抓出來。
但它**只檢查「有沒有 match 到」，不檢查「match 到的是不是你想要的」** —— 正是 Q10.1 那個 bug 能存活的原因：pattern 有 match，只是 match 太多。

**Q10.6** Hydra 的 `defaults: [inst_g, _self_]` 組合繼承。`resume_checkpoint_path` 一改就換 warm-start 來源 → 換一條 lineage。
ledger 上要記的是：**每個 run 的 lineage（從哪個 ckpt warm-start）必須跟數字綁在一起**，這正是 `docs/table.md` 的「世代地圖」在做的事；一個 29 行的 config 讓 lineage 變得極易被忽略。

**Q10.7** 磁碟上**沒有** S0、沒有 S2、沒有 `_raft`、也沒有 `dyn_vggt_v3_s1_oracle_camera_only.yaml`。S0 只剩一個 checkpoint（`training/checkpoints/inst_gate_init.pt`）。
`oracle_camera_only` 被 [trainer.py:113](../../training/trainer.py#L113) 的 docstring 與 [loss.py:762](../../training/loss.py#L762) 的 docstring 引用，程式路徑（`oracle_gate` conf → `_step:1008-1013`）是活的，但**沒有 config 能驅動它** → 那個消融從未跑過。（ledger TR-03）

**Q10.8** 開放題，去 TB 的 `Grad/*` meter 查。重點是意識到：`gate_predictor`（一個 3 層 MLP，梯度來自 BCE）與 `camera_head`（梯度來自 L1 pose loss × weight 5.0）共用一個 `max_norm=1.0` 的分組（[inst_g.yaml:151-152](../../training/config/inst_g.yaml#L151-L152)）—— 若兩者尺度差一個量級以上，clip 實際上只在約束其中一個。

---

## U11

**Q11.1** 少了 `gate_logits`、`world_points`、`world_points_conf`（[vggt_infer.py:209-217](../../training/eval_utils/vggt_infer.py#L209-L217) 只 concat `extrinsic/intrinsic/pose_enc/depth`），而且 `want_point` 根本沒往下傳。
呼叫端**靜默拿不到** —— 例如 `gate_bias_ablation.py:185` 的 `if "gate_logits" in pred` 會直接判偽，`predicted_xT` 模式被跳過（只印一行 warn）。（ledger E-05）

**Q11.2** 見 Q9.4：每個 chunk 是獨立的 forward，各自以自己的第一幀為參考重建，chunk 邊界處的相對 pose 沒有任何 attention 連結 → 接起來的軌跡在接縫處斷裂，全域 Sim(3) 對齊後這些斷裂全部計入 ATE。

**Q11.3** 逐個 head/block family 從 key 前綴獨立偵測（`:44-52`）：`gate_predictor` in key → `enable_gate`；`temporal` in key → `enable_temporal`；`depth_head`/`point_head`/`motion_head`/`flow_head` 各自 `startswith`。
S0 ckpt（有 gate、無 temporal）→ 建出 `enable_gate=True, enable_temporal=False, enable_depth=?`（看有沒有存 depth_head）的模型，`enable_track=False` 恆定，`enable_camera=True` 恆定。
`force_gate=True` 可以在完全沒有 gate 權重的 ckpt（如 VGGT-1B）上硬建一個隨機 gate —— 此時只有 `off`/`oracle` 模式有意義（`:41-43` 的註解）。

**Q11.4** Sim(3)：`align=True, correct_scale=True`（[metrics_pose.py:75-76](../../training/eval_utils/metrics_pose.py#L75-L76)）。所以 ATE 對**全域的旋轉、平移、尺度**都不敏感 —— 這正是單目方法必須這樣報的原因（尺度不可觀測）。RPE 用 `delta=1 frame, all_pairs=True`（`:83-100`）。

**Q11.5** Sintel benchmark 走 **lad2（scale+shift，2 DOF）**：`eval_sequence_depth(..., align_with_lad2=True)` 是預設（[metrics_depth.py:131-146](../../training/eval_utils/metrics_depth.py#L131-L146)），且是**整段序列一次全域擬合**，對齊 MonST3R 協定。
median 是 per-frame、1 DOF、逐次呼叫，目前 repo 內沒有 caller。**兩者數字不可互相比較。**

**Q11.6** 因為 MonST3R 的協定就是這樣：pose 只在 14 個序列上報、depth 在全部 23 個。為了能跟論文表格對齊才照抄。
若你傳 `--seq_list`，這個區分**會被覆蓋** —— pose 與 depth 都用你給的那組序列（`eval_sintel.py` 的協定分支）。所以自訂序列跑出來的數字**不能**跟表 4 的 MonST3R 對照列並排。

**Q11.7** `logs/exp_a/ckpts/last.pt` → 找到路徑中的 `ckpts` 層，取其 parent → `<exp> = exp_a`（[paths.py:44-57](../../training/eval_utils/paths.py#L44-L57)）。
`checkpoints/VGGT-1B.pt`（不含 `ckpts/`）→ fallback 到 `logs/train/VGGT-1B` → `<exp> = VGGT-1B`。
這層存在的理由：讓輸出自帶 ckpt 身分，換 ckpt 重跑不會靜默覆蓋前一次的圖與 json，而且圖與 json 用同一把鑰匙對得起來。

**Q11.8** 傳 `model` 就直接用記憶體裡的活模型；不傳則 `evaluate` 自己從 `args.ckpt` 讀盤建第二份模型 → 訓練中的模型 + eval 模型同時在 VRAM 上，32GB 的 5090 很可能 OOM。這就是 channel B 要能傳 model 的理由（[trainer.py:616](../../training/trainer.py#L616)）。

---

## U12

**Q12.1**
- `off`：`torch.full((1, s, ph*pw), -k)`，均勻大負值（[gate_bias_ablation.py:170](../../training/diag/gate_bias_ablation.py#L170)）
- `oracle`：GT mask → `oracle_logits_from_masks`，靜態 `−k`、動態 `+k`（`:171`）
- `predicted`：`gate_logits_override=None`，用模型自己的（`:182`）
- `predicted_xT`：拿 `predicted` 這一輪回傳的 `gate_logits` 乘上溫度 T 再當 override（`:186-191`）

`off` 不寫成「不傳 override」，是因為那樣會變成 `predicted`（模型自己的 gate 會生效）。要得到「無 gate」必須主動塞一個讓 bias 歸零的 override。

**Q12.2** 24 個**全部**走了 gated path（override 非 None，見 Q4.2）。但 bias 的數值是 `clamp(log2 − softplus(−30), max=0) = clamp(0.693, max=0) = 0` → **數值上等於沒有 bias**。
所以 `off` 在數學上等價於無 gate，但**計算路徑不同**（走 `_gated_global_block_forward` 而非原生 `Block`）。eval 時 dropout=0，兩者應該只差浮點非決定性。
→ `off` 是有效的對照。**但 `oracle` 不是**：它在 block 0–7 也施加了真實的 bias，而 `predicted` 沒有（ledger G-05）。

**Q12.3** `gate_common.oracle_logits_from_masks` 是 numpy/cv2 版，pooling 用 cv2 resize、輸入是已讀進來的 mask 陣列；`loss.oracle_gate_logits_from_mask` 是 torch 版、用 `adaptive_avg_pool2d`、輸入是 batch tensor（[gate_common.py:36-45](../../training/eval_utils/gate_common.py#L36-L45) vs [loss.py:760](../../training/loss.py#L760)）。
兩者**不是 bit-identical**（不同的 resample kernel），邊界 patch 的值會略有差異。同一個實驗內部一致即可；**跨（訓練端 oracle vs 診斷端 oracle）的數字不宜當精確對照。**

**Q12.4** macro = 先逐序列算 AUC/F1 再平均（每個序列等權）；micro = 把所有 patch 池在一起算一次（長序列權重大）。§3.6(c) 的表格明確標 `(macro)`。

**Q12.5** `error_growth.py` 若 `chunk_size < 序列長度`，`infer_sequence_chunked` 會切成獨立 pass（Q9.4）→ 每個 chunk 邊界產生一個誤差跳階 → 圖上會出現**週期性的階梯狀誤差成長**，你會錯誤地下結論「pose 誤差隨 frame index 累積漂移」，但那其實是你自己的 chunking 造出來的。

**Q12.6** 開放題。結合 Q4.2 / Q12.2 的結論，一個具體候選是：
> `oracle` 之所以在短序列有 −11~−15% 而 `predicted` 幾乎為 0，**部分原因可能不是標籤品質，而是 oracle 額外 gate 了 block 0–7**。
可證偽的實驗：造一個「只從 block 8 起施加的 oracle」（在 `forward` 裡把 override 也延後到 `block_iter > gate_block_iter` 才生效），與現行 oracle 對比。
- 若新 oracle 的 Δ 明顯縮小 → 差距來自 gate 的**施加範圍**，`gate_block_iter` 是一個該調的超參，`predicted` 的瓶頸敘事要重寫。
- 若幾乎不變 → 排除這個解釋，回到 class imbalance / 欠自信那條線。

這個實驗很便宜（改幾行 + 一次 ablation），**建議排在 `pos_weight` 之前**。
