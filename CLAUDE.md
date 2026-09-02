# CLAUDE.md

## 這個 repo 是什麼

這是 **VGGT**（Visual Geometry Grounded Transformer）的 fork，承載 **Dyn-VGGT** 研究線：讓 VGGT 的相機 **pose** 在動態場景更 robust。上游 VGGT 是 feed-forward transformer，從 N 個 view 預測相機參數、depth、point map、track。fork 目前的主線是 **motion-gated camera aggregation**（舊文件稱 v3；見 [docs/method.md](docs/method.md)，v1/v2 已封存/放棄）。

**文件從 [docs/README.md](docs/README.md) 進**：結論看 `docs/status.md`、機制看 `docs/method.md`、某個 run 的下場看 `docs/experiments.md`、某支權重看 `docs/checkpoints.md`、數字看 `docs/results/`。

base model 程式碼（`vggt/`）在所有 `enable_*` flag 關閉時與 pretrained `VGGT-1B` 權重**保持 byte-for-byte 相容** —— 新行為都藏在 flag 後面，checkpoint 才載得進來。

## 環境與指令

- **環境**：conda `vggt-dyn`（`conda activate vggt-dyn`；torch 2.12+cu130）。預設 `python` 沒有 torch。套件先裝一次：`pip install -e .`。
- **硬體**：單張 RTX 5090（32 GB）。
- **預設工作目錄在 `training/`** —— 多數指令都假設你在這裡。

訓練（需要 `LOCAL_RANK`/`RANK`，故用 torchrun 而非純 `python`）：
```bash
cd training && torchrun --nproc_per_node=1 launch.py --config <name>
```
config name = `training/config/` 裡的檔名去掉 `.yaml`（如 `inst_g`）。Hydra 載入後 `launch.py` 建 `Trainer(**cfg)`。

沒有配置 test suite / linter。改動要驗證就跑 `training/diag/` 或 `training/benchmark/` 下對應的 script（每支都是獨立、帶 argparse 的 `python … .py`）。

## 架構 —— v3 gate 機制

整個方法是一個 idea 串過三個檔，要一起讀：

- **`vggt/models/aggregator.py`** —— `GatePredictor`（aggregator 中段的 MLP）在 `gate_block_iter`（default 7，約 24 個 global block 的 1/3）後觸發，輸出 per-patch 動態 logit `g` `[B,S,P_patch]`。在 `_process_global_attention` 裡，有 `gate_logits` 時對 **camera/register query row** 於 patch key 加 `−softplus(g)` bias → camera token 結構性地只聚合靜態 patch。attention 拆兩條 path（patch↔patch 走 flash、無 bias；special-query 小 op、帶 bias），故 VRAM 成本 ≈ 0。`gate_logits_override` 是 eval-only 輸入，供 **oracle-mask** 消融用。
- **`vggt/models/vggt.py`** —— 把 `enable_gate`/`gate_block_iter` 串進去，在 `predictions` 曝出 `gate_logits`（帶 gradient）供監督，`σ(g)` 同時當免費的 dynamic mask。所有額外 head/行為都在 `enable_temporal/motion/flow/gate` flag 後面。
- **`training/loss.py`** —— `compute_gate_loss` = `BCE(σ(g), m*_patch)`，GT pixel mask 用 `adaptive_avg_pool2d` pool 到 patch grid。此處 `gate_logits` **不** detach（gradient 流回 predictor）。`oracle_gate_logits_from_mask` 直接從 GT mask 造 override。dispatcher `MultitaskLoss.forward` 只在某 loss 的 config block 存在時才加該項 → loss 純靠 yaml 開關。

gate 是否真的改善 pose 屬**未定論的研究問題**，最新進展與消融結論見 project memory 與 [docs/status.md](docs/status.md)（**全 repo 唯一的結論來源**；`method.md` 只寫機制、不下結論，CLAUDE.md 也不在此下結論）。

**操作提醒**：判斷 gate 品質一律用 **AUC/F1，不要用 BCE** —— `loss_gate`（BCE）用 soft average-pooled label，boundary patch 有不可約的 floor，val BCE 會看似 overfit 但其實與高 AUC 並存。

## 訓練階段與動態標籤

config 檔名 = `<標籤>_<成分>[_<變體>]`，**與 `exp_name` 和 `logs/<exp>/` 目錄名三者一致**（2026-08-20 統一，舊的 `dyn_vggt_v3_s{n}_*` 全部退役，對照表見 [docs/checkpoints.md](docs/checkpoints.md)）。標籤 `inst` = instance×scene-flow mask（唯一在用的動態標籤）；成分字母 `g` = gate、`t` = temporal、`s` = camera_smooth，依序疊加。所以 `inst_g` 是 gate+camera、`inst_gts` 再加 temporal 與 smooth，變體另接後綴（`inst_gts_egoflow`、`inst_g_photo`）。SCARED 那組用資料集前綴 `scared_cam_*`。

config 之間用 Hydra `defaults` 串成繼承鏈，**只寫 delta，不整份複製**：`default` → `inst_g` → `inst_gts` → 各變體；`default` → `scared_cam_b16` → `scared_cam_b2` / `scared_cam_b16_gg` → `..._smooth_temporal` → `..._smoke`。⚠️ **Hydra 對 list 是「取代」不是「合併」** —— 覆寫 `frozen_module_names` 或 `gradient_clip.configs` 時要把整份清單重寫（兩者的語意都與順序無關，可安心重排）。

動態 mask **標籤**由 `training/data/preprocess/po_*` script 離線預算：`dynmask_inst/`（instance × GT scene-flow —— PO 訓練標籤）或 `dynmask_raft/`（RAFT flow-residual —— 用於 Sintel/測試 eval）。`training/data/datasets/` 下的 dataset（`pointodyssey.py`、`tartanair.py`、`spring.py`、`waymo.py`、`co3d.py`）透過 `dynamic_source=` 載入。資料集實體在另一顆碟上，一律**透過 repo root 的 `data` symlink** 存取，並依角色分兩個 bucket：

- `data/train/` —— 訓練混合：`point_odyssey/`、`tartanair/`、`waymo_processed/`、`spring/`（PO 雖然也被 diag 拿來探測，仍歸 train —— bucket 記錄的是資料的身分，不是誰在讀它）。
- `data/eval/` —— benchmark：`sintel/`（**已扁平化**，`final/`、`depth/`、`camdata_left/` 直接在底下，不再有上游的 `training/` 那層）、`bonn/`、`scannetv2/`。

python 端走 `training/data/paths.py` 的 `data_path("train", "point_odyssey")` / `data_path("eval", "sintel")`（可用 `VGGT_DATA_ROOT` 覆寫 root），yaml 端寫相對的 `../data/train/...`（cwd = `training/`）。**不要再把 `/media/...` 掛載點寫死** —— udisks 以 filesystem label 命名 automount 並在撞名時加數字，掛載點不是穩定識別。

## Eval 與診斷

三個目錄各有明確角色，**新增檔案前先確認放哪**：

- `training/eval_utils/` —— **純 library，不放 argparse 進入點**。`vggt_infer.py`（含 `infer_sequence_chunked` 與 Sim3 拼接 `infer_sequence_stitched`）、`paths.py`、`metrics_pose.py`（全序列 ATE + AF-SfMLearner snippet ATE）、`metrics_depth.py`、`metrics_val.py`（trainer 直接依賴）、`gate_common.py`、`gate_vis.py`、`warp_psnr.py`、`media_io.py`、`ply_io.py`。
- `training/benchmark/` —— **要寫進論文的數字**。進入點契約要穩定（ckpt 進、json 出、CLI 參數別亂改名）：`eval_sintel.py`（Sintel pose+depth 主表）、`eval_scared.py`（SCARED，對齊 EndoSfM3D/AF 協定）、`eval_lesion.py` / `eval_gastric.py`（私人資料集）、`eval_monst3r_lesion.py` / `eval_monst3r_gastric.py`（MonST3R 對照）。
- `training/diag/` —— **中間探索/找問題的工具**，壞掉或砍掉不影響論文。2026-08-19 大整理後只剩 8 支，每支對應一個明確問題：
  - `gate_eval.py` —— gate 的**唯一**評估入口（合併了舊的 `gate_quality.py` + `gate_bias_ablation.py`）。`--metrics quality`（AUC/F1/calibration，跟 GT mask 比準不准）、`--metrics pose`（`off`/`predicted`/`oracle`/`predicted_xT` 的 ATE，看兌現到相機沒有）、預設兩者都跑並共用同一次 forward。`--dataset sintel|po`，Sintel 可加 `--report_dynamic_fraction`。
  - `stitch_error.py` —— Sim3 拼接長序列本身貢獻多少 ATE（拼接數字進表前必跑）。
  - `pnp_pose.py` —— 拿 point head + PnP-RANSAC 算 pose，跟 camera head 比。
  - 視覺化一律放 `diag/vis/`：`trajectory.py`（GT vs 兩個 ckpt 的軌跡形狀）、`gate_gif.py`（整段序列的 σ(g) 動畫，**絕對 0..1 色階**）、`pair_grid.py`（2×2 影片：原圖 / MonST3R / VGGT-1B / Dyn-VGGT）、`timeline.py`（重建隨時間累積）、`train_curves.py`（讀 `log.txt` 畫 loss/metric 曲線）。
  - **chunk_size 陷阱**：任何跨幀指標都必須整段一次 forward（`chunk_size=0`）。獨立 chunk 拼接會讓接縫主宰 ATE 且數字不再隨模型變化 —— 權威說明寫在 `eval_utils/vggt_infer.infer_sequence_chunked` 的 docstring。

搬家規則：diag → benchmark 的時機是**你決定那個數字要進論文的那一刻**，不是等它「看起來穩了」。

**輸出路徑**：`logs/<exp>/` 只放訓練產物（`ckpts/`、`tensorboard/`、`log.txt`、trainer 的 `pose_eval/`）；訓練之後產生的一切（benchmark json、ablation、診斷、圖）一律 `outputs/<tool>/<exp>/`，由 `eval_utils/paths.py` 的 `default_output_dir(ckpt, TOOL)` 解析，別自己拼路徑。

Sintel/flow 的共用 IO 與 mask 推導在 `training/data/`（`sintel_io.py`、`motion_mask.py`）——它們被 `data/preprocess/`、`data/datasets/`、benchmark、diag 共用，屬 data layer 而非 eval layer。

## 上游 demo（未改的 base model）

repo 根目錄的 `demo_gradio.py`、`demo_viser.py`、`demo_colmap.py` 跑 pretrained VGGT 做 point-cloud / COLMAP export；`reference/monst3r/` 是含各 dataset loader 的參考實作。
