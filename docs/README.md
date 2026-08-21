# Dyn-VGGT 文件索引

> 每份文件只有一個職責。找數字看 `table.md`，找方法看 `method.md`，找指令看 `execution.md` ——
> 三者刻意不互相複製內容。

## 方法與數據

| 文件 | 用途 |
|---|---|
| [method.md](method.md) | **現行方法**（運動門控相機聚合）。含進度 banner（現在信什麼）與 stage 定義（§8） |
| [table.md](table.md) | **所有評測數字的唯一權威**。含世代地圖（讀任何數字前先確認）、Δ% 基準、已知的協定缺口 |
| [monst3r_design.md](monst3r_design.md) | 與 MonST3R 的逐項設計對照：flow/photometric、平滑先驗、動態遮罩（公式與梯度流向，**不含數字**）|
| [ego_flow.md](ego_flow.md) | `L_ego_flow` 訓練線 —— **陰性收束**。數字、機制證據、與 TTO 線的對照、七個已知的坑 |

## 操作

| 文件 | 用途 |
|---|---|
| [execution.md](execution.md) | 訓練 / 中斷恢復 / 評測 / 診斷的完整指令與已知坑（`CLAUDE.md` 的展開版）|
| [machine_error.md](machine_error.md) | **主機內核崩潰（未解決）** —— 症狀辨識、已排除的原因、待做的測試、崩潰後的清理程序。跑長訓練前先讀 |
| [package.md](package.md) | 上游 VGGT 的套件安裝說明（與本研究無關，保留以減少 merge 衝突）|

## 資料集與結果

| 文件 | 用途 |
|---|---|
| [scared_dataset.md](scared_dataset.md) | SCARED 規格：轉檔後佈局、座標/單位慣例、取樣規則、陷阱清單，以及 §9 gate 在 SCARED 的實測 |
| [scared_results.md](scared_results.md) | SCARED 對 EndoSfM3D / AF-SfMLearner 的數字 |
| [pointcloud_notes.md](pointcloud_notes.md) | 逐場景的手寫點雲觀察（背景 / 動態物體 / 尺度·drift·floaters）—— 人的判斷，重跑產生不出來 |

## 已封存

| 文件 | 用途 |
|---|---|
| [archive/checkpoints.md](archive/checkpoints.md) | **權重登記表** —— 每支 checkpoint 做了什麼、數字多少、能不能刪；含舊名→新名對照（§1.5）與 v1/v2 方法摘要（§2）|

## 不進版控

`planning/`（週計畫、交接、backlog）與 `reading/`（程式重讀單元、自我驗證題、spec↔impl 對照帳）是個人工作文件，已列入 `.gitignore`。
