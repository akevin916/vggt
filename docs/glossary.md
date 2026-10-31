# Glossary

One name per thing across the project. Append new terms; changing an existing definition needs approval first.

| 詞 | 定義 |
|---|---|
| 欄遮罩（column mask） | global attention 的 key 端遮罩：每個 global 層一組保留權重 g_ℓ(p′)=σ(θ_ℓ(p′))，p′ 是幀內 patch 位置，所有非參考幀共用；第 0 幀與 special token 不遮。實作：`Aggregator(enable_colmask=True)`。 |
| ρ\*（目標保留比例） | 欄遮罩在全部層平均後保留的 key 比例。0.25 = 留 25%。各層可以不同，只約束平均。 |
| ρ_t | 訓練中當下允許的平均保留比例，從 1 線性降到 ρ\*。 |
| 對角（AVGGT diagonal） | query 對自己那個 key 不加遮罩。 |
| 平均項（mean token） | 多接一個 K/V token，內容是被丟掉的 key 的（1−g 加權）平均，代替它們參與 softmax。 |
| c64 | SCARED test 評測，每 64 幀一次 forward、重疊 16 幀（`eval_scared.py --chunk_size 64 --overlap 16`）。 |
