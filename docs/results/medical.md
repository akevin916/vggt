# 數據表 — 醫學域（SCARED / C3VD / 私人內視鏡）

> **域：醫學。** 自然場景（Sintel / PointOdyssey）的數字在 [results/natural.md](natural.md)，
> 對外 published 方法的對照在 [results/sota.md](sota.md)。
> **同一個數字只准存在一份表裡**——在這三份之間引用要用連結，不要複製貼上。
>
> 2026-08-25。**只記數字，不下結論。** 每格都標了來源與設定。
> 結論在 [status.md](../status.md)，每個 arm 的身分與下場在 [experiments.md](../experiments.md)。

## 讀表前

- **單位**：SCARED / C3VD 的 ATE 與 depth 誤差都是 **mm**。`ate_rel` = ATE ÷ 軌跡 bbox 對角線。
- **輸入張數是隱藏變因**。發表的表格是單目（一次一張），我們有些數字讓模型一次看很多張。
  零樣本 VGGT-1B 光靠 multi-view 就把 depth 從 0.0758 拉到 0.0488——**排名很大一部分是輸入設定決定的**，
  所以每張表都標輸入。
- **本表只收 benchmark 的 test 數字**。訓練中的 channel B（每 epoch 在 val 上算、用來選 `best_ate.pt`）
  另外標明，不與 test 混排。
- **空白 = 還沒跑**，不是 0。

---

## 1. SCARED

**協定跟 AF-SfMLearner 對齊，逐幀驗證過**：

| AF 的檔案 | 用途 | 我們的 | 驗證 |
|---|---|---|---|
| `test_files.txt`（550 幀 / 7 序列） | depth | `test/` | 幀數與幀號**完全相同** |
| `test_files_sequence1/2.txt` | pose | `pose_seq/dataset{5,3}/keyframe4`（411 / 834 幀） | 同序列，我們多存 frame 0 |

`pose_seq` 是同一段影片逐幀重抽（連續），`test/` 是官方稀疏抽樣（間隔 1–296 幀）。
兩者都屬 keyframe4 = test，**沒進訓練**（train 用 keyframe1/2，val 用 keyframe3）。
⚠️ `test/dataset3/keyframe4`（79 幀）和 `pose_seq/dataset3/keyframe4`（834 幀）**同名不同物**，引用要帶 split。

### 1.1 Depth（test split，afsfm 協定）

來源 `outputs/eval_scared/test_0f{,_single}_afsfm/`｜08-25 重跑

| ckpt | 單張輸入 | 多張輸入 |
|---|---|---|
| `VGGT-1B` 零樣本 | 0.0758 | 0.0488 |
| `vanilla` | 0.0561 | 0.0454 |
| `b16` | 0.0563 | 0.0459 |
| `b16_gg` | 0.0550 | 0.0449 |
| `smooth_temporal` | **0.0537** | 0.0434 |
| `wide` | 0.0550 | **0.0419** |

（AbsRel。單張 = `--single_view`，**只有這欄能跟論文的單目表比**；多張 = `chunk_size 64 / overlap 16`。）

<details><summary>完整欄位 sq_rel / rmse / log_rmse / δ1</summary>

| ckpt | 單張 | 多張 |
|---|---|---|
| `VGGT-1B` | 0.7671 / 6.5298 / 0.1046 / 0.9472 | 0.3742 / 4.6048 / 0.0735 / 0.9784 |
| `vanilla` | 0.4503 / 4.9296 / 0.0792 / 0.9757 | 0.3305 / 4.2538 / 0.0675 / 0.9831 |
| `b16` | 0.4626 / 5.0053 / 0.0804 / 0.9715 | 0.3341 / 4.3075 / 0.0677 / 0.9839 |
| `b16_gg` | 0.4440 / 4.9193 / 0.0785 / 0.9757 | 0.3176 / 4.2263 / 0.0663 / 0.9852 |
| `smooth_temporal` | 0.4256 / 4.7842 / 0.0767 / 0.9779 | 0.3089 / 4.0898 / 0.0647 / 0.9855 |
| `wide` | 0.4544 / 4.8946 / 0.0789 / 0.9739 | 0.3219 / 4.1107 / 0.0658 / 0.9841 |

</details>

### 1.2 Pose（pose_seq split，snippet ATE）

AF 的 5-frame snippet 協定。**只能跟 AF-SfMLearner 的表比**——EndoSfM3D 那系用全序列 evo ATE，
我們產不出（VGGT 單次上限 80 幀，Sim3 拼接讓 ATE 擺動 −1%~+32%，`diag/stitch_error.py`）。是不估，不是估錯。

chunk 64 和 chunk 5 分開記：兩者算指標的方式相同，但**預測本身不同**——
chunk 64 每個窗的位姿來自 64 幀 context，chunk 5 則 context 就是那 5 幀（更貼近 AF）。不可互相取代。

**chunk 64 / overlap 16**（`pose_seq_0f/`，08-25 10:28）

| ckpt | ds3 | ds5 | **mean** |
|---|---|---|---|
| `VGGT-1B` 零樣本 | 0.1159 | 0.1258 | 0.1209 |
| `vanilla` | 0.0626 | 0.0902 | 0.0764 |
| `b16` | 0.0586 | 0.0826 | 0.0706 |
| `b16_gg` | 0.0565 | 0.0832 | 0.0699 |
| `smooth_temporal` | 0.0478 | 0.0716 | **0.0597** |
| `wide` | 0.0585 | 0.0825 | 0.0705 |

**chunk 5 / overlap 4**（`pose_seq_0f_chunk5/`，08-25 11:25）

| ckpt | ds3 | ds5 | **mean** |
|---|---|---|---|
| `VGGT-1B` 零樣本 | 0.1294 | 0.1549 | 0.1421 |
| `vanilla` | 0.0712 | 0.1017 | 0.0865 |
| `b16` | 0.0675 | 0.0973 | 0.0824 |
| `b16_gg` | 0.0663 | 0.0956 | 0.0810 |
| `smooth_temporal` | 0.0567 | 0.0844 | **0.0705** |
| `wide` | 0.0723 | 0.1007 | 0.0865 |

> ✅ `b16_gg` 這列與 08-19 舊值（0.0663 / 0.0956 / 0.0810）**一位不差**，確認舊那筆是 chunk 5，
> 也驗證 pipeline 沒有漂移。

**兩種 chunk 的差異**：chunk 5 的數字**全部比 chunk 64 差**（+11% ~ +23%），
context 從 5 幀放大到 64 幀對每個 arm 都有幫助。排名大致相同（`smooth_temporal` 兩邊都最好），
但 `wide` 在 chunk 5 掉到跟 `vanilla` 並列最差（0.0865）。

| ckpt | chunk 5 | chunk 64 | 差 |
|---|---|---|---|
| `VGGT-1B` | 0.1421 | 0.1209 | −15% |
| `vanilla` | 0.0865 | 0.0764 | −12% |
| `b16` | 0.0824 | 0.0706 | −14% |
| `b16_gg` | 0.0810 | 0.0699 | −14% |
| `smooth_temporal` | 0.0705 | 0.0597 | −15% |
| `wide` | 0.0865 | 0.0705 | −19% |

⚠️ **test split 不能算 snippet ATE**（幀不連續，`eval_scared.py:168` 會 skip）。
舊 `test_0f/results.json` 裡 `b16` 的 **0.6148 是守衛加入前的失效值，不要用**。

### 1.3 選 ckpt 的指標跟 test 不一致

| arm | channel B val ATE（選 ckpt 用） | test pose (chunk 64) | test depth 單張 |
|---|---|---|---|
| `smooth_temporal` | 0.9306 | **0.0597** | **0.0537** |
| `wide` | **0.8240** | 0.0705 | 0.0550 |

`wide` 只在 depth 多張輸入領先（0.0419）。pose 兩種 chunk 都是 `smooth_temporal` 最好。
**「哪個是最好的版本」目前沒有單一答案**，看用哪個指標。

---

## 2. C3VD

### 2.1 Test（`test_50f/`，5 序列 × 50 幀，monst3r 協定 max_depth 100）

| ckpt | ATE | ate_rel | rpe_t | rpe_r | AbsRel | δ1 |
|---|---|---|---|---|---|---|
| `VGGT-1B` 零樣本 | 2.6343 | 0.0985 | 1.1681 | 0.7625 | 0.2252 | 0.6186 |
| `c3vd_cam_vanilla` best_ate | 0.5978 | 0.0218 | 0.2356 | 0.2173 | 0.0874 | 0.9584 |
| `c3vd_cam_gts` best_ate | **0.4329** | 0.0176 | 0.1865 | 0.1859 | **0.0736** | 0.9649 |
| `c3vd_cam_gts` epoch_5 | 0.4642 | 0.0181 | 0.1946 | 0.1841 | 0.0977 | 0.9343 |
| `c3vd_cam_gts` best_loss ⚠️ | 0.4689 | 0.0181 | 0.1987 | 0.1900 | 0.0953 | 0.9346 |

sq_rel / rmse / log_rmse / δ2 / δ3：
`VGGT-1B` 2.6733 / 9.6971 / 0.2649 / 0.9022 / 0.9785｜
`vanilla` 0.3244 / 3.6219 / 0.1107 / 0.9971 / 0.9997｜
`gts best_ate` 0.2478 / 3.0627 / 0.0974 / 0.9967 / 0.9996｜
`gts ep5` 0.4010 / 3.9649 / 0.1230 / 0.9944 / 0.9996｜
`gts best_loss` 0.3823 / 3.8607 / 0.1206 / 0.9952 / 0.9997

⚠️ `best_loss.pt` 那列是 **08-24 22:14 當時的檔案**（ep3），該檔 08-25 08:29 已被 **ep6** 覆寫，重跑不可復現。

### 2.2 逐 epoch（channel B，val = `trans/t3_b` 24 幀）

| run | ep0 | ep1 | ep2 | ep3 | ep4 |
|---|---|---|---|---|---|
| `c3vd_cam_vanilla` | **0.4021** | 0.4118 | 0.4401 | — | — |
| `c3vd_cam_gts` | **0.3589** | 0.4296 | 0.4194 | 0.4218 | 0.4715 |

零樣本 `VGGT-1B` 同設定 0.4991。

**兩個 run 的 `best_ate.pt` 都是 epoch 0**，之後再沒更新過——上表「我們的版本」就是第一個 epoch 的權重。
test 也同向：ep0 0.4329 < ep5 0.4642 < best_loss 0.4689。訓練在 ep6 之後暫停。

---

## 3. 私人資料集

### 3.1 lesion（病灶）— warp PSNR

⚠️ **不是精度指標**。這批資料沒有標定，PSNR 量的是 **depth 與 pose 合不合得起來**（自洽性）。
`no spec` = 排除飽和高光（任一通道 ≥ 250/255）的像素——內視鏡光源黏在鏡頭上，高光跟著相機動，
幾何 warp 重現不了它。

| ckpt | 病灶1 (12) | 病灶2 (12) | 病灶3 (19) |
|---|---|---|---|
| `VGGT-1B` 零樣本 | 20.127 / 20.027 | 22.992 / 23.550 | 26.637 / 27.146 |
| `scared_cam_vanilla` | 19.403 / 19.038 | 23.578 / 23.841 | 25.758 / 25.978 |
| `scared_cam_b16` | 17.420 / 17.004 | 22.307 / 22.853 | 23.956 / 24.240 |
| `scared_cam_b16_gg` | 17.768 / 17.283 | 22.405 / 22.822 | 23.769 / 24.059 |
| `smooth_temporal` | 19.018 / 18.585 | 23.280 / 23.724 | 26.006 / 26.179 |
| `wide` | 18.139 / 17.655 | 22.199 / 22.716 | 24.961 / 25.202 |

（`PSNR / PSNR_no_specular`，pair 模式。specular_frac：病灶1 0.342、病灶2 0.280、病灶3 0.266。）

兩個版本差 **≤ 0.56 dB**，排序幾乎不變（只有病灶2 的 `b16`/`b16_gg` 互換，本來就差 0.1 dB）。
病灶1 扣掉高光反而變差，因為兩張圖的高光重疊時「白對白」殘差近 0，等於白送分。

`seq` / `seq2` 只產點雲，**點數由幀數×像素數決定、不帶模型資訊**：
病灶1/2 seq 3,187,68x（MonST3R 2,335,692）、病灶3 seq 5,047,16x（MonST3R 3,698,179）。

> MonST3R 的 PSNR 是**設計上沒有**，不是漏跑：它沒在內視鏡微調（不算有效 baseline），
> 且 pair-PSNR 是相對 VGGT 的 depth scale 定義的，兩者 scale 不同無法比。

### 3.2 gastric（胃）— 沒有數值指標

單目、無標定，只有點雲。`seq_video1_0135_0139`（30 幀）：
`VGGT-1B` 與 `scared_cam_b16` 的 ply_points **完全相同**（6,892,290），MonST3R 6,569,160。
這欄不帶模型資訊，只能做視覺對照。

---

## 4. 缺口

| 缺口 | 現況 |
|---|---|
| 「最好的版本」沒有單一答案 | channel B 選 `wide`，test 上 `smooth_temporal` 較好（§1.3）。報告要先決定用哪個指標敘事 |
| `results/sota.md` §3.3 的 0.0550 出處未定 | `b16_gg` 與 `wide` 都是 0.0550，無法回推。該文件**未修改** |
| `eval_scared.py` docstring 第 10 行 | 寫 test split「stride 8-38」，實測 1–296。**未修改** |
| SCARED 全序列 evo ATE | 不報（拼接誤差）。EndoSfM3D 系的 pose 欄因此填不了 |
| C3VD MonST3R 對照 | **已決定不做** |
| C3VD `best_loss.pt` 已被 ep6 覆寫 | 表上該列重跑不可復現 |
| gastric 數值指標 | **已決定不做** |
| lesion 的 MonST3R PSNR | **設計上沒有**，見 §3.1 |
| lesion 缺「微調過的 MonST3R」 | baseline 政策要求對照組是微調過的；MonST3R 至今未微調 |
| `scared_cam_b16_gg` 未跑完 10 epoch | ep8 被 SIGTERM，其 `best_ate.pt` 是 8-epoch 產物 |
