# 數據表 — 醫學域第二批（2026-09 SCARED ablation）

> **建立 2026-09-30，2026-10-02 改為三條線分表。只記數字，不下結論。** 結論寫在 [status.md](../status.md)。
> 第一批（2026-08，depth_head 凍結底座）在 [medical.md](medical.md)，C3VD 與私人資料集也在那份。
> **同一個數字只准存在一份表裡** —— 跨檔案引用要用連結，不要複製。
> 唯一例外：兩個底座（`vanilla`、`vanilla_frozen`）與完整模型 A 在 §1–§3 每張表都重列一次，當作該線的對照。

## 讀表前

- **底座有兩個，不可混比。** 本批的主體是 `scared_cam_vanilla_depth`（表中 `vanilla`；depth_head **可訓練** + `loss.depth`
  alpha 0.02）；第一批與 dual 前三代用的是 `scared_cam_vanilla`（表中 `vanilla_frozen`；depth_head 凍結）。兩者的 pose 起點
  分別是 0.0851 與 0.0865，depth 單張則是 0.0501 與 0.0561。「深度頭解凍」欄 ✓ = 建在前者之上。
- **四個協定**：depth 單張（`--single_view`，唯一能與論文單目表比）、depth 多張（`chunk_size 64 overlap 16`）、
  pose chunk 64、pose chunk 5（`chunk_size 5 overlap 4`，最接近 AF-SfMLearner 的公開設定）。
  depth 是 AbsRel、afsfm 協定、`test/depth` split 550 幀；pose 是 5 幀 snippet ATE、`test/pose` split。
- **雜訊門檻**（同一個 run 的 `best_ate` 與 `epoch_10` 之間的擺動上限，7–8 個 run 的最大值）：
  depth 單張 **1.0%**、depth 多張 **2.3%**、pose chunk 64 **3.4%**、pose chunk 5 **2.4%**。
  判斷任何差異是否有意義都用這組門檻。
- **全部單一 seed**，沒有任何重複實驗。
- **ckpt（2026-10-03 起）**：`best_ate` 的挑選機制（val ATE）與 test 排名不一致，暫時**逐格**取 `epoch_10` 與 `best_ate` 中較好的值。
  **ᵇ** = 該格來自 `best_ate`，無標記 = `epoch_10`。同一列可能混用兩支權重；這等於用 test 挑 ckpt，數字偏樂觀。
  兩支的原始值見附錄 A.2。`vanilla_frozen`、`frame_dino` 本機只有 `best_ate`；† 的 run 只有一個值、ckpt 未記錄。
- **粗體** = 該欄在同一組內的最佳值。每張表分兩組，第二組從 A 那一列開始（§3 從 `geom_smooth` 開始），對應投影片的綠底。
- **成分代號**：t = temporal block、f = 光流幾何一致 loss、s = 軌跡二階平滑 loss、i = 光照 token、d = 距離帶 attention bias。
  A = t+f+s（`scared_cam_depth_tfs`）。
- **†** = 在另一台機器跑的（同 branch `illu-vggt`、同 config、同底座），本機 `outputs/` 沒有對應 json。
- **空白 = 沒跑**，不是 0。

---

## 1. Attention bias 線

| run | 深度頭解凍 | 訊號來源 | 使用方式 | 預測精度 | token 範圍 | 其他成分 | depth 單張 | depth 多張 | pose c64 | pose c5 |
|---|---|---|---|---|---|---|---|---|---|---|
| `vanilla` | ✓ | — | — | — | — | | 0.0499ᵇ | 0.0450 | 0.0711 | 0.0851 |
| `vanilla_frozen` | | — | — | — | — | | 0.0561ᵇ | 0.0454ᵇ | 0.0764ᵇ | 0.0865ᵇ |
| `frame_dino` | | DINO | 正負交替 | 每層 | camera | | 0.0529ᵇ | **0.0422ᵇ** | 0.0709ᵇ | 0.0809ᵇ |
| `frame_time` | | 幀號差 | 正負交替 | 每層 | camera | | 0.0541 | 0.0458 | **0.0676** | 0.0789 |
| `band_frozen` | | 幀號差 | 高斯帶 | 每 head | camera | | 0.0550ᵇ | 0.0439ᵇ | 0.0710ᵇ | **0.0737ᵇ** |
| `band_camera` | ✓ | 幀號差 | 高斯帶 | 每 head | camera | | 0.0495 | 0.0442 | 0.0707 | 0.0774ᵇ |
| `head_dino` | ✓ | DINO | 高斯帶 | 每 head | 全部 | | **0.0491** | 0.0444 | 0.0725 | 0.0845 |
| `token_dino` | ✓ | DINO | 高斯帶 | 每 token | 全部 | | 0.0501 | 0.0434 | 0.0728 | 0.0850 |
| A（`tfs`） | ✓ | — | — | — | — | t+f+s | 0.0498 | **0.0409ᵇ** | **0.0609** | **0.0709** |
| A+d（`tfsd`）† | ✓ | 幀號差 | 高斯帶 | 每 head | camera | t+f+s | 0.0499 | 0.0421 | 0.0634 | 0.0751 |
| A+i+d（`tfsdi`）† | ✓ | 幀號差 | 高斯帶 | 每 head | camera | t+f+s+i | **0.0490** | 0.0428 | 0.0640 | 0.0728 |

`tfsdi` 的光照 token weight 為 0.05。

---

## 2. 光照 token 線

| run | 深度頭解凍 | loss weight | 其他成分 | depth 單張 | depth 多張 | pose c64 | pose c5 |
|---|---|---|---|---|---|---|---|
| `vanilla` | ✓ | — | | 0.0499ᵇ | 0.0450 | 0.0711 | 0.0851 |
| `vanilla_frozen` | | — | | 0.0561ᵇ | 0.0454ᵇ | 0.0764ᵇ | 0.0865ᵇ |
| `illu_msr` | ✓ | 1.0 | | **0.0493** | 0.0453 | 0.0711 | **0.0798** |
| `illu_msr`（w=0.2）† | ✓ | 0.2 | | 0.0496 | **0.0442** | **0.0696** | 0.0814 |
| A（`tfs`） | ✓ | — | t+f+s | 0.0498 | **0.0409ᵇ** | 0.0609 | **0.0709** |
| A+i（`tfsi`） | ✓ | 1.0 | t+f+s | 0.0501 | 0.0442 | 0.0647 | 0.0749 |
| A+i（`tfsi_w005`）† | ✓ | 0.05 | t+f+s | 0.0495 | 0.0433 | **0.0603** | 0.0717 |
| A+d（`tfsd`）† | ✓ | — | t+f+s+d | 0.0499 | 0.0421 | 0.0634 | 0.0751 |
| A+d+i（`tfsdi`）† | ✓ | 0.05 | t+f+s+d | **0.0490** | 0.0428 | 0.0640 | 0.0728 |

illu corr（光照 token 預測與 MSR 目標的相關，val 累積平均）：`tfsi` 0.8432、`illu_msr`（w=1.0）0.8378、
`tfsi_w005` 0.7347（逐 epoch ep5–ep9：0.6680 / 0.7123 / 0.6801 / 0.7156 / 0.7347）。

---

## 3. 時間資訊載體線

| run | 時間資訊載體 | 其他成分 | depth 單張 | depth 多張 | pose c64 | pose c5 |
|---|---|---|---|---|---|---|
| `vanilla` | — | | 0.0499ᵇ | 0.0450 | 0.0711 | 0.0851 |
| `rope3d` | global RoPE 重切（head_dim 24/24/16 = y/x/t） | | 0.0536ᵇ | 0.0504ᵇ | 0.1110ᵇ | 0.0945ᵇ |
| `time_rope`（`ropet`） | global RoPE 疊加時間相位（768 參數） | | **0.0495** | 0.0446 | 0.0731 | 0.0786 |
| `temporal_block`（`temporal`） | 新加 8 個 temporal block（100.8M） | | 0.0501 | **0.0426ᵇ** | **0.0669** | **0.0753** |
| `temporal_shared` | temporal block 借用 frame block 權重（只學 16k LayerScale） | | 0.0498ᵇ | 0.0428ᵇ | 0.0707 | 0.0841 |
| `time+temporal`（`ropet_temporal`） | 疊加時間相位 + temporal block | | 0.0502 | 0.0438 | 0.0724 | 0.0765 |
| `geom_smooth`（`flowgeom_smooth`） | — | f+s | 0.0501 | 0.0426ᵇ | 0.0671 | 0.0811ᵇ |
| A（`tfs`） | temporal block | f+s | **0.0498** | **0.0409ᵇ** | **0.0609** | **0.0709** |
| A'（`fs_ropet`）† | 疊加時間相位 | f+s | 0.0504 | 0.0443 | 0.0770 | 0.0761 |

---

## 附錄 A. 每個 run 的身分與訓練紀錄

### A.1 run 名對照與 val ATE

**val ATE** 欄是 trainer 的 channel B（每 epoch 在 val 全序列上算，用來選 `best_ate.pt`），單位 mm。
它與 test 的排名並不一致，只作為訓練過程的紀錄。

| 表中名稱 | config / exp | ckpt | val ATE 最佳 | 備註 |
|---|---|---|---|---|
| `vanilla` | `scared_cam_vanilla_depth` | 逐格 | 1.1212 @ep6 | depth head 解凍 + `loss.depth` α=0.02 |
| `vanilla_frozen` | `scared_cam_vanilla` | best_ate（僅此一支） | 0.9705 @ep3 | depth head 凍結 |
| `frame_dino` | `scared_cam_dual` | best_ate（僅此一支） | 0.9243 @ep6 | 舊機制，見下方 ⚠️ |
| `frame_time` | `scared_cam_dual_gap` | 逐格 | 1.0849 @ep6 | 舊機制，見下方 ⚠️ |
| `band_frozen` | `scared_cam_dual_kern` | 逐格 | 1.0509 @ep7 | 距離帶 bias（v1） |
| `band_camera` | `scared_cam_depth_dual` | 逐格 | 1.1856 @ep6 | 每 head 幀距高斯帶（792 參數），camera scope |
| `head_dino` | `scared_cam_depth_headdino` | 逐格 | 1.0622 @ep6 | |
| `token_dino` | `scared_cam_depth_tokdino` | 逐格 | 1.0578 @ep5 | per-token 自適應 DINO 距離帶 |
| A | `scared_cam_depth_tfs` | 逐格 | 1.0285 @ep7 | temporal block + 兩個 loss |
| `tfsi` | `scared_cam_depth_tfsi` | 逐格 | 1.1163 @ep3 | 光照 token w=1.0（4.2M 參數） |
| `illu_msr` | `scared_cam_depth_illu_msr` | 逐格 | 1.2755 @ep2 | 光照 token + MSR 目標 |
| `temporal` | `scared_cam_depth_temporal` | 逐格 | 0.9001 @ep7 | |
| `ropet` | `scared_cam_depth_ropet` | 逐格 | 1.2166 @ep3 | 無新模組 |
| `ropet_temporal` | `scared_cam_depth_ropet_temporal` | 逐格 | 1.2616 @ep6 | |
| `temporal_shared` | `scared_cam_depth_temporal_shared` | 逐格 | 1.0774 @ep7 | |
| `rope3d` | `scared_cam_depth_rope3d` | 逐格 | 4.1577 @ep6 | |
| `flowgeom_smooth` | `scared_cam_depth_flowgeom_smooth` | 逐格 | 1.0981 @ep7 | 純 loss 端，0 新參數 |

⚠️ `scared_cam_dual` 與 `scared_cam_dual_gap` 的前向（`aggregator.dual_stream_p` × DINO 相似度）
已於 2026-09-23 被距離帶 bias 取代，`load_vggt_for_eval` 會拒絕載入這兩支 ckpt。要重評必須
`git checkout 18aa9fc`。表中數字取自當時的 json。

### A.2 兩支權重的原始值（`epoch_10` / `best_ate`）

主表逐格取兩者中較低者。

| run | depth 單張 | depth 多張 | pose c64 | pose c5 |
|---|---|---|---|---|
| `vanilla` | 0.0501 / 0.0499 | 0.0450 / 0.0451 | 0.0711 / 0.0736 | 0.0851 / 0.0861 |
| `tfs` | 0.0498 / 0.0501 | 0.0413 / 0.0409 | 0.0609 / 0.0624 | 0.0709 / 0.0724 |
| `tfsi` | 0.0501 / 0.0501 | 0.0442 / 0.0473 | 0.0647 / 0.0721 | 0.0749 / 0.0846 |
| `temporal` | 0.0501 / 0.0505 | 0.0432 / 0.0426 | 0.0669 / 0.0689 | 0.0753 / 0.0764 |
| `ropet_temporal` | 0.0502 / 0.0502 | 0.0438 / 0.0447 | 0.0724 / 0.0780 | 0.0765 / 0.0794 |
| `temporal_shared` | 0.0499 / 0.0498 | 0.0438 / 0.0428 | 0.0707 / 0.0714 | 0.0841 / 0.0861 |
| `ropet` | 0.0495 / 0.0543 | 0.0446 / 0.0483 | 0.0731 / 0.0788 | 0.0786 / 0.0827 |
| `rope3d` | 0.0547 / 0.0536 | 0.0512 / 0.0504 | 0.1163 / 0.1110 | 0.0956 / 0.0945 |
| `dual（band_camera）` | 0.0495 / 0.0499 | 0.0442 / 0.0445 | 0.0707 / 0.0735 | 0.0777 / 0.0774 |
| `headdino` | 0.0491 / 0.0497 | 0.0444 / 0.0448 | 0.0725 / 0.0747 | 0.0845 / 0.0851 |
| `tokdino` | 0.0501 / 0.0516 | 0.0434 / 0.0447 | 0.0728 / 0.0769 | 0.0850 / 0.0860 |
| `illu_msr` | 0.0493 / 0.0537 | 0.0453 / 0.0481 | 0.0711 / 0.0770 | 0.0798 / 0.0893 |
| `flowgeom_smooth` | 0.0501 / 0.0504 | 0.0427 / 0.0426 | 0.0671 / 0.0681 | 0.0821 / 0.0811 |
| `flowgeom` | 0.0502 / 0.0506 | 0.0420 / 0.0429 | 0.0696 / 0.0713 | 0.0822 / 0.0822 |
| `smooth` | 0.0495 / 0.0500 | 0.0440 / 0.0437 | 0.0681 / 0.0699 | 0.0859 / 0.0864 |
| `dual_kern（band_frozen）` | 0.0555 / 0.0550 | 0.0441 / 0.0439 | 0.0713 / 0.0710 | 0.0747 / 0.0737 |
| `dual_kern_wide` | 0.0552 / 0.0543 | 0.0447 / 0.0436 | 0.0703 / 0.0712 | 0.0769 / 0.0755 |
| `dual_gap（frame_time）` | 0.0541 / 0.0546 | 0.0458 / 0.0512 | 0.0676 / 0.0687 | 0.0789 / 0.0807 |

來源 `outputs/eval_scared/{test_depth_0f_single_afsfm,test_depth_0f_afsfm,test_pose_0f,test_pose_0f_chunk5}/<exp>_<ckpt>/results.json`。

### A.3 pose 的逐序列值（`epoch_10`，ds3 / ds5）

| run | chunk 64 | chunk 5 |
|---|---|---|
| `vanilla_depth` | 0.0565 / 0.0857 | 0.0694 / 0.1008 |
| `tfs` | 0.0472 / 0.0746 | 0.0571 / 0.0848 |
| `tfsi` | 0.0512 / 0.0782 | 0.0599 / 0.0899 |
| `temporal` | 0.0531 / 0.0807 | 0.0596 / 0.0910 |
| `dual` | 0.0551 / 0.0862 | 0.0601 / 0.0953 |
| `ropet` | 0.0571 / 0.0892 | 0.0628 / 0.0944 |
| `ropet_temporal` | 0.0587 / 0.0861 | 0.0610 / 0.0920 |
| `illu_msr` | 0.0549 / 0.0872 | 0.0645 / 0.0951 |
| `flowgeom_smooth` | 0.0527 / 0.0816 | 0.0658 / 0.0985 |
| `rope3d` | 0.0977 / 0.1350 | 0.0736 / 0.1176 |

ds3 = `test/pose/dataset3/keyframe4`（834 幀）、ds5 = `dataset5/keyframe4`（411 幀）。

### A.4 不屬於三條線的 run

| run | 成分 | depth 單張 | depth 多張 | pose c64 | pose c5 | val ATE 最佳 | ckpt |
|---|---|---|---|---|---|---|---|
| `scared_cam_depth_flowgeom` | f | 0.0502 | 0.0420 | 0.0696 | 0.0822 | 0.9837 @ep6 | 逐格 |
| `scared_cam_depth_smooth` | s | 0.0495 | 0.0437ᵇ | 0.0681 | 0.0859 | 1.0545 @ep7 | 逐格 |
| `scared_cam_dual_kern_wide` | v1 距離帶放寬 clamp（凍結底座） | 0.0543ᵇ | 0.0436ᵇ | 0.0703 | 0.0755ᵇ | 1.1970 @ep6 | 逐格 |
| `VGGT-1B` 零樣本 | — | 0.0758 | 0.0488 | 0.1209 | 0.1421 | — | — |

自監督線（不同訓練目標，僅作參考）：`scared_selfsup_ca3` epoch_10 為 0.0629 / 0.0486 / 0.0660 / 0.0775，
`scared_selfsup_sm05` epoch_10 為 0.0867 / 0.0734 / 0.0649 / 0.0752。

---

## 附錄 B. 專項量測（非 benchmark）

### B.1 3D RoPE 重切的零訓練代價

`vanilla_depth` 的權重，不訓練，只把 head_dim 從 32/32 改成 24/24/16 並把 t 全設 0：

| 條件 | pose c64 | pose c5 |
|---|---|---|
| 原樣 2D RoPE | 0.0711 | 0.0851 |
| 3D RoPE，t≡0 | 0.1922（+170%） | 0.1801（+112%） |

來源 `outputs/rope3d_precheck/scared_cam_vanilla_depth/summary.json`。

### B.2 illumination token 的切斷檢驗

每層把該 token 的 hidden state 歸零，其餘不動；`cut_reg` 是對同一個模型的 register token #1 做同樣處理。

| ckpt / 場地 | 原樣 | token 輸入歸零 | 每層切斷 token | 切斷 register（對照） |
|---|---|---|---|---|
| `tfsi` epoch_10 / SCARED test/pose c64 | 0.0647 | 0.0649（+0.3%） | 0.0651（+0.7%） | 0.0653（+1.0%） |
| `c3vd_cam_vanilla_illu_phys` epoch_10 / C3VD test | 0.4129 | 0.4125（−0.1%） | 0.4192（+1.5%） | 0.4243（+2.8%） |

來源 `outputs/illu_path_ablation/<exp>/`。

### B.3 各 loss 在共享 trunk 上的梯度（`tfsi` epoch_10）

6 個 4 幀 batch，梯度取自 `aggregator.global_blocks.8-23` 與 `camera_token`（302.4M 參數）。

| loss | 梯度範數 | 與 `loss_illu` 的餘弦 |
|---|---|---|
| `loss_conf_depth` | 1.91e+02 | −0.002 |
| `loss_illu` | 3.98e+00 | — |
| `loss_reg_depth` | 5.35e-01 | −0.027 |
| `loss_FL` | 3.84e-01 | −0.036 |
| `loss_camera` | 1.93e-01 | −0.041 |
| `loss_flow_geom` | 3.82e-02 | −0.047 |
| `loss_T` | 2.70e-02 | +0.006 |
| `loss_camera_smooth` | 9.05e-03 | +0.019 |
| `loss_R` | 5.17e-03 | +0.011 |

同一批量到的最負配對（與光照無關）：`loss_FL` 對 `loss_R` −0.362、`loss_R` 對 `loss_camera` −0.323。
來源 `scratchpad/grad_conflict_tfsi.log`（一次性 script，未進版控）。

### B.4 patch-scope 距離帶 bias 的成本

單層 attention，bf16，64 幀（N = 66624）與 12 幀 fwd+bwd（N = 12492）：

| 條件 | 64 幀 forward | 12 幀 fwd+bwd |
|---|---|---|
| flash，無 bias | 85.8 ms（1.00×） | 11.2 ms（1.00×） |
| flex，score_mod 空轉 | 148.5 ms（1.73×） | 19.5 ms（1.75×） |
| flex，加一個常數 | 136.3 ms（1.59×） | 19.5 ms（1.75×） |
| flex，本專案的索引查表 | 346.8 ms（4.04×） | 36.4 ms（3.26×） |

整個模型的層級：訓練單步 camera scope 1.39 s、patch scope 9.69 s（6.96×）；
64 幀推論 無 bias 3.49 s、camera scope 3.59 s、patch scope 7.97 s（2.28×）。
記憶體：patch scope 12 幀 fwd+bwd 17.3 GiB、50 幀 forward 18.0 GiB（舊的物化 mask 路徑在 batch 12 直接 OOM）。
來源 `scratchpad/flex_overhead.py`、`flex_speed.py`、`flex_infer_speed.py`。

---

## 附錄 C. 缺口

| 缺口 | 現況 |
|---|---|
| seed | 全部單一 seed，零重複實驗 |
| leave-one-out | 有 `tfsd` 之後仍缺「拿掉 temporal」（f+s+d）與「拿掉 loss 端」（t+d）兩列 |
| 微調過的 MonST3R | baseline 政策要求，至今未訓練 |
| 跨資料集 | 本批沒有任何 ckpt 在 C3VD 上評過 |
| 私人資料集 | lesion / gastric 仍是 2026-08 的舊 ckpt（見 [medical.md](medical.md) §3） |
| † 的 run | `tfsd`、`tfsdi`、`tfsi_w005`、`illu_msr` w=0.2、`fs_ropet` 的 json 與 ckpt 都不在本機；ckpt 取 epoch_10 或 best_ate 未記錄 |
| context 長度不對稱 | `dual` 的距離除以 (S−1)、時間相位 RoPE 吃絕對幀號，兩者都是 c5 有效、c64 無效或反向；修正版未測 |
