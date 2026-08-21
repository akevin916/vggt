# Dyn-VGGT：執行手冊

> 所有指令在 `training/` 下執行，conda env `vggt-dyn`。
> 方法設計見 [method.md](method.md)；所有評測數字見 [table.md](table.md)；
> 每支 checkpoint 的身分見 [archive/checkpoints.md](archive/checkpoints.md)。
>
> 這份是 `CLAUDE.md` 的展開版：CLAUDE.md 只放 AI 每次都要知道的最小集合，完整指令與坑在這裡。

---

## 0. 環境

```bash
conda activate vggt-dyn          # py3.11 / torch 2.8.0+cu128 / RTX 5090 32 GB
cd /home/cvml-75/Desktop/vggt/training

pip show vggt | grep Editable    # 確認 vggt 指向本 repo
# 若不對：cd /home/cvml-75/Desktop/vggt && pip install -e . --no-deps
```

資料一律走 repo root 的 `data` symlink，**不要寫死 `/media/...` 掛載點**（udisks 會依 label 命名並在撞名時加數字）。python 端用 `data/paths.py` 的 `data_path("train", "scared")`，yaml 端寫相對的 `../data/train/...`。

---

## 1. 訓練

```bash
torchrun --nproc_per_node=1 launch.py --config inst_gts
```

`--config` 是 `training/config/` 裡的檔名去掉 `.yaml`。需要 `LOCAL_RANK`/`RANK`，所以用 `torchrun` 而非 `python`。

**`max_epochs` 不是「跑多久」的旋鈕**：`where = epoch / max_epochs` 驅動 LR schedule（`trainer.py:878`），改它會重塑整條 cosine 曲線 —— 10-epoch 的 run **不等於** 20-epoch run 的前 10 個 epoch。同一組比較的每個 arm 必須共用同一個值。

---

## 2. 中斷與恢復

```bash
# 中斷（優雅，會寫完 last.pt）
kill -TERM $(pgrep -f "launch.py --config <config>")

# 恢復
torchrun --nproc_per_node=1 launch.py --config <config> --resume
```

**`--resume` 不可漏**。trainer 在建模型之前就檢查：`save_dir` 下已有 `last.pt` 卻沒給 `--resume` → **拒絕啟動**（`trainer.py:131-144`），這是為了不讓你手滑覆蓋掉一個跑到一半的 run。反過來，給了 `--resume` 卻找不到 checkpoint 也會直接報錯。

resume 會還原 optimizer state 與 epoch 計數，並快轉 dataloader。`save_steps_freq` 讓 trainer 每 N 步就更新 `last.pt`，所以崩潰最多損失 N 步而不是整個 epoch。

> ⚠️ `training/checkpoints/*.pt` 是 **weights-only**（~4.7 G，沒有 optimizer state）——只能當 warm-start
> 起點，**不能 resume**。要接著練得用 `logs/<exp>/ckpts/*.pt`（~7.8 G）。size 就是判準。
>
> ⚠️ 2026-08-20 清理時，`inst_gts` 與 `scared_cam_b16` 的 `last.pt` 被刪（與 `epoch_N.pt` 逐位元
> 相同）。這兩個 run 要 resume 得先 `cp epoch_N.pt last.pt`。

`extract_weights.py` 是把訓練 checkpoint 轉成 weights-only 匯出檔用的，**不是** resume 流程的一部分：

```bash
python extract_weights.py --src logs/<exp>/ckpts/epoch_30.pt --dst checkpoints/inst_gts.pt
```

---

## 3. 評測

### 論文數字：`benchmark/`

進入點契約要穩定（ckpt 進、json 出、CLI 參數別亂改名）。

```bash
# Sintel pose + depth 主表（14 序列）
python benchmark/eval_sintel.py --ckpt logs/inst_gts/ckpts/epoch_30.pt
python benchmark/eval_sintel.py --ckpt logs/inst_gts/ckpts/epoch_30.pt --seq_list alley_2   # 單序列 smoke

# SCARED（對齊 EndoSfM3D / AF-SfMLearner 協定）
python benchmark/eval_scared.py --ckpt logs/scared_cam_b16_gg/ckpts/best_ate.pt --split val

# 私人資料集
python benchmark/eval_lesion.py  --ckpt <ckpt> --modes pair,seq
python benchmark/eval_gastric.py --ckpt <ckpt>
python benchmark/eval_monst3r_lesion.py    # MonST3R 對照
```

輸出一律 `outputs/<tool>/<exp>/`，由 `eval_utils/paths.py` 的 `default_output_dir(ckpt, TOOL)` 解析。**不要自己拼路徑** —— `<exp>` 那層讓 json 自帶 ckpt 身分，換 ckpt 重跑不會靜默覆蓋。

依賴：`evo`、`opencv-python`、`tifffile`（SCARED 的 `scene_points` 只有 tifffile 讀得對）。

### 診斷：`diag/`

```bash
# gate 的唯一評估入口：quality（AUC/F1/calibration）+ pose（off/predicted/oracle 的 ATE）
python diag/gate_eval.py --ckpt checkpoints/inst_g.pt --all_seqs
python diag/gate_eval.py --ckpt logs/inst_gts/ckpts/epoch_30.pt --metrics quality --all_seqs
python diag/gate_eval.py --ckpt checkpoints/inst_g.pt --dataset po --metrics pose

# Sim3 拼接長序列本身貢獻多少 ATE（拼接數字進表前必跑）
python diag/stitch_error.py --ckpt <ckpt>

# point head + PnP-RANSAC 算 pose，跟 camera head 比
python diag/pnp_pose.py --ckpt <ckpt>

# 視覺化
python diag/vis/gate_gif.py --ckpt <ckpt>          # 整段序列的 σ(g)，絕對 0..1 色階
python diag/vis/trajectory.py --ckpt_a <A> --ckpt_b <B>
python diag/vis/pair_grid.py                       # 2×2：原圖 / MonST3R / VGGT-1B / Dyn-VGGT
python diag/vis/timeline.py --input_video ... --clouds ...
python diag/vis/train_curves.py --log logs/<exp>/log.txt
```

**判讀 gate 品質一律用 AUC/F1，不要用 BCE** —— BCE 的目標是 average-pool 後的軟標籤，boundary patch 有不可約的 floor，val BCE 會看似 overfit 但其實與高 AUC 並存。

---

## 4. 已知坑

| 問題 | 說明 / 解法 |
|---|---|
| **`chunk_size` 陷阱** | 任何跨幀指標都必須整段一次 forward（`chunk_size=0`）。獨立 chunk 拼接會讓接縫主宰 ATE，且數字不再隨模型變化（Sintel f50 的 `temple_2` 因此變成 2.53 而非 0.057）。權威說明在 `eval_utils/vggt_infer.infer_sequence_chunked` 的 docstring |
| **Hydra 對 list 是「取代」不是「合併」** | 覆寫 `frozen_module_names` 或 `gradient_clip.configs` 時要把整份清單重寫。兩者語意都與順序無關（`freeze_modules` 用 `any(fnmatch)`、`GradientClipper` 每組獨立掃全部參數），可安心重排 |
| **VRAM 的真正決定因素是 `img_nums` 上界，不是 `max_img_per_gpu`** | `dynamic_dataloader.py:173-175` 做的是 `batch_size = max(1, floor(max_img_per_gpu / random_image_num))` —— 當 sampler 抽到的 `random_image_num` 大於 cap，floor 變 0、`max()` 抬成 1，該 batch 就帶著**超過 cap** 的影像數。最壞情況永遠是 `max(img_nums)` |
| `accum_steps > 1` 空 chunk crash | 設 `accum_steps: 1` |
| `F.binary_cross_entropy` 不允許 AMP | 已改手寫 BCE（`loss.py`） |
| TartanAir 沒有 `loss_motion` | 預期行為（純靜態集無 `motion_mask`） |
| SCARED 空深度幀 | 會靜默吃掉 camera loss，取樣時要避開當 anchor（[scared_dataset.md](scared_dataset.md) §5） |
| 主機隨機崩潰 | **未解**，跟本專案程式無關。症狀辨識與崩潰後清理見 [machine_error.md](machine_error.md) |

長時間訓練建議 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
