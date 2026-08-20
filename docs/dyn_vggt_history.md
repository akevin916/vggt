# Dyn-VGGT：方法沿革（v1 → v2 → v3）

> 一頁式摘要，只留「為什麼改版」的決策脈絡；v1/v2 的方法細節與各版本權重的數字見 [archive/checkpoints.md](archive/checkpoints.md)。

---

## v1：運動解耦雙場表示（[archive/checkpoints.md §2.1](archive/checkpoints.md)）

把每像素世界點分解成 `X = X^can + m·Δ`（靜態正則點 + 動態概率 × 殘差位移），另加時空聚合器（temporal attention）、動態分割頭、場景流頭、4D 全局對齊。

**為何放棄**：診斷發現兩個架構級缺陷（非訓練/數據問題）：
1. `m·Δ` 是雙線性項，只監督組裝後的和 `X`，`(m, Δ)` 有無限多組解 → `Δ` 學不起來（動態/靜態 `Δ` 中位數比值 0.77–1.16，理想應 ≫1，即使 mask 已經很準也一樣）。
2. 動態區 `X^can` 因 `(1−m)` 屏蔽而無監督，變成自由變量 → 網絡走「強預訓練 point 頭」的捷徑把運動塞進 `X^can`，`Δ` 閒置 ≈ 0。

---

## v2：診斷、未形成新架構（[archive/checkpoints.md §2.2](archive/checkpoints.md)）

本質是 v1 失敗診斷的整理版，確認上述雙線性不可辨識性是**架構病態**、且學習式動態 mask 有**跨域崩塌**（PointOdyssey AUC≈0.88 → Sintel≈0.45，與雙場病態是兩個獨立問題）。未提出替代架構，是 v3 的前置診斷。

---

## v3：運動門控相機聚合（現行，[dyn_vggt_method_v3.md](dyn_vggt_method_v3.md)）

放棄一切幾何重表示，範圍收斂到「VGGT 唯一輸給 MonST3R 的指標——動態場景相機 pose」。核心改動：把動態屏蔽從 **loss 端搬到 attention 端**——中段 gate predictor 出 patch 動態 logit，讓 camera/register token 在 global attention 只聚合靜態 patch；門控信號用 domain-invariant 幾何殘差監督，根治跨域崩塌；depth/point 頭原樣不動。