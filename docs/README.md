# Dyn-VGGT 文件索引

```
docs/
  status.md          現在信什麼、下一步做什麼      ← 每次回到專案先讀這份
  method.md          方法怎麼運作（只寫機制）
  experiments.md     每個 run 的身分與下場
  checkpoints.md     每支權重的身分與能不能刪
  ────────────────────────────────────────────
  results/           數字（依域分，互不重疊）
  topics/            單一主題的完整交代（寫完就很少動）
  ops/               怎麼跑、壞掉怎麼辦
  report/            已凍結的報告產物
```

> **路由規則**：結論 → `status.md`／機制 → `method.md`／某個 run 的下場 → `experiments.md`／
> 某支權重 → `checkpoints.md`／數字 → `results/`。
> **這五類刻意不互相複製內容**，跨檔一律用連結。散文裡提到別的文件時，寫**相對 `docs/` 的路徑**
> （例如 `results/natural.md`），不要只寫檔名。

## 頂層四份 —— 會跟著實驗一直改

| 文件 | 回答什麼問題 |
|---|---|
| **[status.md](status.md)** | **「現在信什麼、下一步做什麼」** —— 全 repo 唯一的結論來源。三條線的狀態、已收掉的線、優先序 |
| **[method.md](method.md)** | **「這個方法怎麼運作」** —— 只寫機制，不寫數字也不下結論。§10 是文件↔程式對照表 |
| **[experiments.md](experiments.md)** | **「這個 run 是誰、能不能用」** —— warm-start、跑到哪、代表 ckpt、下場。§6 是失敗的病因檔案 |
| **[checkpoints.md](checkpoints.md)** | **「這支權重是誰、能不能刪」** —— 舊名→新名對照、已刪清單、被否決的 v1/v2 摘要 |

## `results/` —— 數字，依域分

| 文件 | 域 | 內容 |
|---|---|---|
| [results/natural.md](results/natural.md) | 自然場景 | Sintel / PointOdyssey 的自家消融（表 1–4 + 世代地圖） |
| [results/medical.md](results/medical.md) | 醫學 | SCARED / C3VD / 私人內視鏡的自家消融 |
| [results/sota.md](results/sota.md) | 對外 | 我們 vs 已發表方法（只放對照與協定辯護） |

**同一個數字只准存在一份表裡。** 前兩份是「我們的 arm 互比」，第三份是「跟論文比」。

## `topics/` —— 單一主題的完整交代

| 文件 | 內容 |
|---|---|
| [topics/ego_flow.md](topics/ego_flow.md) | `L_ego_flow` 這條線的**陰性收束**：數字、機制證據、七個坑。附錄 A 收著三張不可重跑的 flow 探針表 |
| [topics/monst3r_design.md](topics/monst3r_design.md) | 與 MonST3R 的逐項設計對照（公式與梯度流向，**刻意不含數字**） |
| [topics/scared_dataset.md](topics/scared_dataset.md) | SCARED 規格：轉檔佈局、座標/單位慣例、取樣規則、陷阱清單 |
| [topics/video_pipeline.md](topics/video_pipeline.md) | 最終影片的配方與耗時，**以及三個模型的推論時間實測**（成本軸宣稱的資料來源） |
| [topics/pointcloud_notes.md](topics/pointcloud_notes.md) | 逐場景的手寫點雲觀察 —— 人的判斷，重跑產生不出來 |

## `ops/` —— 怎麼跑、壞掉怎麼辦

| 文件 | 內容 |
|---|---|
| [ops/execution.md](ops/execution.md) | 訓練／中斷恢復／評測／診斷的完整指令與坑（`CLAUDE.md` 的展開版） |
| [ops/machine_error.md](ops/machine_error.md) | **主機內核崩潰（未解決）**：症狀辨識、已排除的原因、崩潰後清理程序。**跑長訓練前先讀** |
| [ops/upstream_package.md](ops/upstream_package.md) | 上游 VGGT 的安裝說明。與本研究無關，保留只為減少 merge 衝突 |

## `report/` —— 已凍結，不再更新

[report/midterm_report.md](report/midterm_report.md)、`report.tex`、`references.bib`、`slides/midterm_deck.html`
是期中報告的 **2026-08-26 快照**。之後的 depth arm、lowalpha、照明線都不在裡面 ——
要新內容請看頂層四份，不要改這裡（改了會跟交出去的版本對不上）。

## 不進版控

`planning/`（週計畫、交接、backlog）與 `reading/`（程式重讀、自我驗證題、spec↔impl 對照帳）
是個人工作文件，已列入 `.gitignore`。
