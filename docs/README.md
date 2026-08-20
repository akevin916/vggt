# Dyn-VGGT 文件索引

## 現行

| 文件 | 用途 |
|---|---|
| [dyn_vggt_method_v3.md](dyn_vggt_method_v3.md) | 現行方法設計（運動門控相機聚合），含 stage/課程定義（§8） |
| [dyn_vggt_execution.md](dyn_vggt_execution.md) | 訓練/評測操作手冊（指令、SOP、已知坑） |
| [dyn_vggt_history.md](dyn_vggt_history.md) | v1→v2→v3 一頁式沿革摘要 |
| [monst3r_loss_diff.md](monst3r_loss_diff.md) | MonST3R vs Dyn-VGGT 的 loss 逐項對照（公式、變數、梯度流向；不含數字） |
| [ego_flow_result.md](ego_flow_result.md) | `L_ego_flow` 訓練線的結果（**陰性**）、機制證據、與 TTO 線的對照、七個已知的坑 |
| [host_instability.md](host_instability.md) | **主機內核崩潰調查（未解決）** —— 症狀辨識、已排除的原因、待做的測試、崩潰後的清理程序 |
| [scared_dataset.md](scared_dataset.md) | SCARED 內視鏡資料集規格（轉檔後佈局、座標/單位慣例、取樣規則、陷阱）—— 寫 `data/datasets/scared.py` 的依據 |
| [package.md](package.md) | VGGT 原版套件安裝方式（與 dyn-vggt 實驗無關） |

## 已封存（`archive/`）

| 文件 | 用途 |
|---|---|
| [archive/checkpoints.md](archive/checkpoints.md) | **權重登記表** —— 每支 checkpoint 做了什麼、數字多少、能不能刪；末節是 v1/v2 方法摘要 |

v1/v2 的三份長文件（`dyn_vggt_method_v1.md`、`dyn_vggt_method_v2.md`、`dyn_vggt_implementation.md`）已於 2026-08-20 刪除，內容壓縮進 `archive/checkpoints.md` §2。
