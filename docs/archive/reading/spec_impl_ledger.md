# 想法 vs 實作 對照帳（spec ↔ impl ledger）

> **用途**：`docs/method.md` 寫的是「我想做什麼」，程式寫的是「AI 實際做了什麼」。
> 這份表把前者拆成可逐條查核的主張，讓你在讀完 [reading/units.md](units.md) 的每個單元後**立刻**判定。
>
> **判定欄由你填，不是由 AI 預填** —— 那是這整件事的重點。四選一：
>
> | 判定 | 意思 |
> |---|---|
> | `一致` | 程式做的就是文件說的 |
> | `簡化` | 方向對，但省了文件描述的某個細節（要在備註寫省了什麼） |
> | `自作主張` | 程式做了文件沒說、或與文件相反的事 |
> | `未實作` | 文件寫了，程式裡沒有 |
>
> 填法：把 `☐` 改成判定詞。備註欄寫「省了什麼 / 多做了什麼 / 影響哪些數字」。

---

## 0. 已驗證的落差（重讀前先看，避免重複發現）

這四條是規劃階段掃程式時就撞到的，已用程式或實驗驗證過。**判定欄仍留給你**，因為你要自己讀過才算真的知道。

| ID | 事實 | 佐證 | 影響 | 判定 |
|---|---|---|---|---|
| **X-1** | `frozen_module_names` 的 `"*aggregator.global_blocks.[0-7]*"` 因 glob 尾部 `*` 會吃掉後續數字，實際凍住 `0-7 + 10-23`，**只有 block 8、9 可訓練** | `wcmatch.fnmatch` 實測（見 [answers.md Q10.1](answers.md)）；pattern 出現在 `logs/inst_g/log.txt:4` | **所有 S1 run**；§7.1 的 S1 定義、§7.3 的四 run 架構、全部 clean 世代 pose 數字 | ☐ |
| **X-2** | `gate_logits_override` 從 `block_iter=0` 就生效，而模型自身的 gate 只從 block 8 生效 → `oracle` 與 `predicted` 不是同一架構配置 | [aggregator.py:363](../vggt/models/aggregator.py#L363) vs `:380-383` | §3.6(a)「oracle 頭空 vs predicted 不兌現」的整套敘事 | ☐ |
| **X-3** | `compute_gate_loss` 的 `alpha_m` / `beta_m` 參數在函式體內完全未使用 | [loss.py:702-756](../training/loss.py#L702-L756) | 無（死參數），但會誤導以為 §3.7 的 soft threshold 已實作 | ☐ |
| **X-4** | `infer_sequence_chunked` 在 chunked 路徑丟棄 `gate_logits` / `world_points` | [vggt_infer.py:209-217](../training/eval_utils/vggt_infer.py#L209-L217) | 任何 `chunk_size>0` 且需要 gate 輸出的診斷 | ☐ |

---

## 1. 貢獻① Gate Mechanism（§3）

| ID | 設計主張（出處） | 對應實作 | 判定 | 備註 |
|---|---|---|---|---|
| G-01 | gate predictor 的輸入是**中段 contextualized patch token**，切掉 camera+register，沒有 concat 任何其他通道（§3.1） | `aggregator.py:380-383`、`:54-64` | ☐ | |
| G-02 | 末層 zero-init → 第 0 步 `g≡0` → 前向 byte-for-byte 等於 pretrained VGGT（§3.1 warm-start） | `aggregator.py:51-52` + `:541-543` | ☐ | |
| G-03 | 放在 ~1/3 深度（`gate_block_iter=7`），理由是「特徵夠成熟 + 後段仍有 2/3 block 可門控」（§3.1） | `aggregator.py:119`、`:380` | ☐ | 實際被門控的是幾個 block？ |
| G-04 | bias `= min(0, softplus(0) − softplus(g))`，只加在 camera/register query 列、patch key 欄（§3.2） | `aggregator.py:540-544` | ☐ | |
| G-05 | 「後段每個 global block 都套此 bias」（§3.2 結尾） | `aggregator.py:363` + `:380-383` | ☐ | **與 X-2 直接相關** |
| G-06 | patch↔patch attention「一字不改」，深度品質完全保留（§3.2 / §3.4） | `aggregator.py:549` vs `vggt/layers/block.py` | ☐ | 手刻 block 與原生 block 真的等價嗎？ |
| G-07 | 兩條 path 拆分讓顯存增加 ≈ 0（§3.4） | `aggregator.py:548-560` | ☐ | 有實測過嗎？ |
| G-08 | frame-interleaved 切分已修正（§3.4 的 `e064087` 註記） | `aggregator.py:517-523`、`:557-560` | ☐ | |
| G-09 | bias 走 detach，pose 梯度不反灌 gate（§3.1-1） | `aggregator.py:441` vs `loss.py:725` | ☐ | |
| G-10 | PO 訓練標籤是 `m*_inst`（instance × GT scene-flow），§3.5.2(a) | `pointodyssey.py:108-146` | ☐ | 三層 fallback 讓實際標籤可能不是它 |
| G-11 | Sintel / 測試端標籤是 `m_geo`（GT `.flo` 的幾何殘差 + 相對門檻），§3.5.2(b) | `motion_mask.py:49,81` | ☐ | |
| G-12 | 「兩者語義一致 → 跨域不崩」（§3.5.2 分工段） | 兩支標籤 script 的門檻定義 | ☐ | 兩個門檻的物理單位真的可比嗎？ |
| G-13 | `σ(g)` 直接當動態 mask 副產物，不需要額外 head（§3.1-3 / §7.2） | `vggt.py:92-93`、`:43` | ☐ | |
| G-14 | C-1（hard label + ignore band）砍掉 boundary floor（§3.5.1） | `loss.py:741-750` | ☐ | 只設一個門檻會靜默失效 |
| G-15 | §3.7 的兩項（`m_geo_soft` 軟目標、幾何殘差輸入通道）**明示未實作** | — | ☐ | 確認程式裡沒有半成品殘骸（見 X-3） |

---

## 2. 損失函數（§5.3.1 / §6）

| ID | 設計主張（出處） | 對應實作 | 判定 | 備註 |
|---|---|---|---|---|
| L-01 | 「`MultitaskLoss.forward` 只在對應 config block 存在時才加該項」，loss 純靠 yaml 開關（§6） | `loss.py:49-147` | ☐ | 開關不匹配時是靜默還是報錯？ |
| L-02 | `L_gate = BCE(σ(g), m*_patch)`，`m*_patch` 是 average-pool 的軟機率（§3.5.1） | `loss.py:725-755` | ☐ | |
| L-03 | patch 格與 `gate_logits` 的解析度天然對齊（§3.1「零插值」） | `loss.py:728-739` | ☐ | 有沒有 assert？不對齊時怎麼失敗？ |
| L-04 | 總 loss `= L_cam + L_depth + λ_g·L_gate [+ λ_p·L_static_photo] [+ λ_s·L_camera_smooth]`（§6） | `loss.py:65-139` + 各 config 的 `loss:` 區塊 | ☐ | run 1 的 `L_depth` 在哪？ |
| L-05 | 「兩者梯度都只進預測 pose，不碰幾何頭」（§6 中括號段） | `loss.py:499-502`、`:621` | ☐ | |
| L-06 | `L_cam` 的 `valid_frame` 維持寬鬆的「有效點數 > 100」（§6 結尾） | `loss.py:164-166` | ☐ | |
| L-07 | `motion`/`flow`/`reproj`/`tsmooth` 是 v1/v2 遺留、v3 未開（§4.3-1） | `loss.py:365-486, 684-700` + 各 config `null` | ☐ | |

---

## 3. 貢獻② Motion Loss（§4）

| ID | 設計主張（出處） | 對應實作 | 判定 | 備註 |
|---|---|---|---|---|
| P-01 | 只在 GT 靜態區 + 有效深度上算（§4.2） | `loss.py:540` | ☐ | |
| P-02 | 遮擋防護：把 t+1 的 mask warp 過去，落在動態區則丟棄（§4.2） | `loss.py:574-577` | ☐ | 靜態遮擋擋得住嗎？ |
| P-03 | 梯度只進預測 extrinsics，GT depth/內參/影像皆常數，depth head 完全不碰（§4.2） | `loss.py:527-570` | ☐ | 對照 run 3 的 config 有沒有解凍 depth_head |
| P-04 | 用 `pose_enc_list[-1]`（§4.2） | `loss.py:514` | ☐ | |
| P-05 | 變體 3'（預測深度反投影）**列為獨立 ablation、不當預設**（§4.2 變體） | 有沒有對應 config / 程式分支？ | ☐ | |

---

## 4. 貢獻③ Temporal Attention（§5）

| ID | 設計主張（出處） | 對應實作 | 判定 | 備註 |
|---|---|---|---|---|
| T-01 | `enable_temporal` → `aa_order` 變 `[frame, temporal, global]`，每 3 個 aa-block 插一個，共 8 個（§5.1） | `vggt.py:30`、`aggregator.py:183-184, 371` | ☐ | 補丁式寫法看要不要改 |
| T-02 | temporal block 只更新 streaming tokens、不 emit intermediate → head 輸入維持 `2C`（§5.1） | `aggregator.py:575-595` vs `:385-388` | ☐ | |
| T-03 | 獨立 1D 時間 RoPE；camera+patch 給真實 frame index、register 給 0（§5.1） | `aggregator.py:131, 337-346` | ☐ | |
| T-04 | LayerScale γ=0 warm-start，加了 temporal 仍能載 VGGT-1B（§5.2） | `aggregator.py:201-207` | ☐ | |
| T-05 | `camera_smooth` 罰 2 階、Δt 正規化、四元數 sign-fix、FoV 不平滑、需 `S≥3`（§5.3） | `loss.py:594-681` | ☐ | |
| T-06 | 「`camera_smooth` 不需要 temporal 才能算」→ 兩個因子技術上可拆（§5.3 註） | `loss.py:621` | ☐ | 那為什麼 run 2 沒拆？ |
| T-07 | temporal 是 §3.3「全動態幀借鄰幀」的載體（§5.1 結尾） | — | ☐ | 有任何程式或實驗支持這個因果嗎？ |

---

## 5. 訓練策略（§7）

| ID | 設計主張（出處） | 對應實作 | 判定 | 備註 |
|---|---|---|---|---|
| TR-01 | S1 解凍 = gate predictor + **後段 global(8–23)** + camera head（§7.1 表） | `inst_g.yaml:141-148` + `train_utils/freeze.py:48-52` | ☐ | **見 X-1** |
| TR-02 | S1 凍結 = patch_embed + frame_blocks + 前段 global(0–7) + depth/point head（§7.1 表） | 同上 | ☐ | `camera_token`/`register_token` 算哪一類？ |
| TR-03 | 四個 run 統一架構（都 `enable_gate + enable_temporal`），run 1 的 temporal 凍住且 γ=0（§7.3 統一架構段） | 四份 config 的 `model:` 與 `frozen_module_names` | ☐ | |
| TR-04 | 「四個 run 必須跑到同一個 epoch 預算」（§7.3 ⚠️） | 四份 config 的 `max_epochs` × `limit_train_batches` | ☐ | 實際跑到的 epoch 呢？ |
| TR-05 | S0（只訓 gate、快取凍結特徵、顯存近零）（§7.1 / §7.2(b)） | 有 S0 config 嗎？有快取程式嗎？ | ☐ | |
| TR-06 | S2 保幾何版（全 global + gate + camera head 解凍、`L_depth` 當錨、混 30% 靜態）（§7.1） | — | ☐ | |
| TR-07 | oracle-gate 訓練消融（`dyn_vggt_v3_s1_oracle_camera_only`）（`trainer.py:113` docstring） | `trainer.py:1008-1013` 程式活著，config 不存在 | ☐ | |
| TR-08 | 「gradient checkpointing（已開）、bf16 autocast（已開）」（§7.2(c)） | `aggregator.py:410-411` 等、`trainer.py` 的 amp 設定 | ☐ | gated path 的 checkpoint 行為與 ungated 一樣嗎？ |
| TR-09 | 非有限 loss 時「attempting to stop training」 | `trainer.py:939-943` | ☐ | 訊息與實際行為一致嗎？ |

---

## 6. 評測協定（§8）與工具層

| ID | 設計主張（出處） | 對應實作 | 判定 | 備註 |
|---|---|---|---|---|
| E-01 | `logs/` 只放訓練產物、`outputs/<tool>/<exp>/` 放訓練後一切（CLAUDE.md / `eval_utils/__init__.py`） | `paths.py` + 各 script 的 `out_dir` 預設 | ☐ | 有沒有 script 自己拼路徑？ |
| E-02 | channel A 只有 loss、channel B 只有 metric，互不產生對方訊號（`default.yaml:15-21`） | `trainer.py:643-730` vs `:579-641` | ☐ | |
| E-03 | `best_loss.pt` 的選點訊號「從不碰報告資料」（`run_train` docstring） | `trainer.py:543-552` | ☐ | |
| E-04 | channel B 用 `chunk_size=0`，因為 chunking 會灌大 ATE | `trainer.py:613` + `default.yaml` | ☐ | |
| E-05 | `infer_sequence_chunked` 是 `infer_sequence` 的等價擴充 | `vggt_infer.py:191-217` | ☐ | **見 X-4** |
| E-06 | pose 用 14 seq、depth 用 23 seq，對齊 MonST3R（§8.1） | `benchmark/eval_sintel.py` | ☐ | `--seq_list` 會不會破壞它？ |
| E-07 | depth 用 lad2（scale+shift，全序列一次擬合）對齊 MonST3R | `metrics_depth.py:131-146` | ☐ | |
| E-08 | `off` mode 是有效的「無 gate」對照（`gate_bias_ablation.py`） | `gate_bias_ablation.py:170, 178` | ☐ | bias 數值真的是 0 嗎？路徑一樣嗎？ |
| E-09 | 訓練雜訊帶 ≈ ±0.02（±13%），小於此的改善不可信（§8.3 / table.md） | — | ☐ | 目前哪些結論其實在雜訊帶內？ |
| E-10 | 診斷端 oracle（numpy）與訓練端 oracle（torch）刻意分離 | `gate_common.py:36-45` vs `loss.py:760` | ☐ | 數字可互比嗎？ |

---

## 填完之後

1. 把所有非 `一致` 的條目按「**是否影響已產出的數字**」分成兩堆。
2. 影響數字的 → 在 [table.md](../table.md) 的「世代地圖」新增一個世代並標注哪些列作廢。
3. 不影響數字的 → 二選一：**補實作**，或**改文件讓它誠實**。不要留著「文件說有、程式沒有」的狀態。
4. 每條 `自作主張` 都值得回頭問一句：AI 當初為什麼這樣改？是它偷懶，還是原設計在實作上真的行不通？後者要把理由寫回 method 文件。
