# P3（cross gate）：訓練崩潰的原因、修法與在別台訓練的步驟

> 2026-10-16 在 75 上查到並修好。分支 `p3-cross-gate`。

## 1. P3 是什麼

config `scared_cam_depth_tfs_p4d_lr1e3_cross` = `tfs` + PAGE-4D 式 gate + cross 範圍：

- gate 預測頭在兩層 linear 之間有 3×3 depthwise conv（`gate_head: conv`）。
- bias = −g，不截斷（`gate_bias_form: linear`）；pose loss 訓練 gate（`gate_pose_grad: true`）；gate 學習率 1e-3，其餘 3e-5。
- `gate_scope: cross`：從第 8 個 global block 起，**所有** token 看**其他幀**的 patch 時，分數加上那個 patch 的 bias；同一幀內不加。

cross 的 attention 有兩種等價寫法（`gate_cross_impl`，不存進 ckpt）：

| 寫法 | 怎麼做 | kernel |
|---|---|---|
| `augment`（預設） | q、k 各多接 S 維，內積時跨幀的那一對多出 bias；q/k 維度 64 → 72 或 80 | mem-efficient |
| `per_frame` | 每個 query 幀分開算一次 SDPA，帶 [B,1,1,N] 的 key bias；維度維持 64 | mem-efficient |

兩種都沒辦法走 flash（維度不同或帶 bias）。訓練設定本來就 `disable_flash_sdp: True`，所以訓練時和其他 run 用同一類 kernel。
推論時其他層走 flash、cross 那幾層走 mem-efficient，會比 `tfs` 慢，慢多少沒量。

## 2. 問題

2026-10-14 在 75 上 5 次 cross 訓練全部在第 41–59 步 segfault，其中 2 次整台機器當掉重開：

| run | 寫法 | 死在第幾步 |
|---|---|---|
| `..._cross_smoke_crash1` | augment | 45 |
| `..._cross_dead_reboot_1101` | augment | 41（之後機器重開） |
| `..._cross_smoke` | augment | 43（機器當掉） |
| `..._cross` | augment | 59 |
| `..._cross_pf_smoke` | per_frame | 53（機器當掉） |

核心紀錄的 segfault 全部在 `libnvrtc.so.12`（NVIDIA 的執行期編譯器），多數在反向傳播的執行緒。
同一天只限制 camera 的 `..._p4d_lr1e3_smoke` 跑滿 60 步。09-12 到 10-05 同一台、同一驅動（580.159.03）、同一 torch（2.8.0+cu128）的 18 次完整訓練都沒有 segfault。

## 3. 原因

**不是 attention**：兩種寫法都崩，而且三條 attention 路徑都走 mem-efficient kernel，這個 kernel 不會即時編譯。

**是 gate 預測頭的 depthwise conv**：

- `GatePredictor.forward` 把 `[B·S, h, w, D]` 轉置成 `[B·S, D, h, w]` 再送進 conv，這是記憶體不連續（channels-last）的 view。
- cuDNN 對這種排列的 depthwise conv 沒有現成 kernel，改用 runtime-compiled engine（`libcudnn_engines_runtime_compiled.so.9`），每遇到一種新形狀就呼叫 NVRTC 編譯一次。
- 訓練時每個 batch 的幀數和長寬比都不同，所以一直在編譯：前 55 步編了 192 次。每次編譯有小機率讓 NVRTC 自己 segfault，累積到約 50 步就會碰到。
- 以前的 run 沒有這個 conv，其他卷積不會觸發即時編譯，所以不會崩。

證據：

- 在 gdb 底下跑 P3 smoke，斷點設在 `nvrtcCompileProgram`：呼叫者全部是 cuDNN 卷積（前向、反向 input、反向 weight）；第 55 步當場抓到 segfault，堆疊是「反向傳播 → `cudnn_convolution_backward_weight` → runtime-compiled engine → `nvrtcCompileProgram`」。崩潰當下只有這一個執行緒在 NVRTC 裡。
- 單獨測同一個 conv（bf16 autocast，3 種 batch × 5 種網格，前向加反向）數 NVRTC 呼叫次數：

| conv | 輸入 | NVRTC 編譯次數 |
|---|---|---|
| depthwise 256 ch | 轉置後、不連續（原寫法） | 60 |
| depthwise 256 ch | `.contiguous()` | 0 |
| 一般 3×3 conv | 轉置後、不連續 | 0 |

## 4. 修法

`vggt/models/aggregator.py` 的 `GatePredictor.forward`：送進 `dwconv` 前加 `.contiguous()`。

- 數學相同：兩種輸入的輸出最大差 0.0156，輸出量級約 3.5（bf16 在這個量級的一個最小刻度）。
- 只影響 `gate_head: conv` 的模型；其他設定與舊 ckpt 不受影響。

驗證：修好後 P3 cross smoke（augment）不用 debugger 跑滿 200 步，沒有 segfault，結束時正常存檔。

## 5. 還沒確認的

- 只驗證了 200 步，完整 10 epoch × 1500 步還沒跑過。
- 修好後在 gdb 底下第一次重跑時，第 0 步崩在 `libcuda.so` 的 `cuStreamSynchronize`（沒有經過 NVRTC）。不用 gdb 重跑就沒事，推測和 debugger 或前一次崩潰留下的狀態有關，未證實。長跑若又在別處崩，先查這條。
- `per_frame` 在修好後沒有重跑過。
- gate 方向：上一個 session 量到第 60 步 `gate_g_median` = −0.19。bias = −g 不截斷，代表跨幀 patch 普遍被**加分**而不是壓低。這和崩潰無關，是設計上的問題，未處理。

## 6. 在別台訓練

需要的東西：

| 項目 | 來源 |
|---|---|
| 程式 | 分支 `p3-cross-gate`（含本修正的 commit） |
| config | `pipeline/training/config/` 被 gitignore，需另外複製這 4 支：`scared_depth_base.yaml`、`scared_cam_depth_tfs_p4d.yaml`、`scared_cam_depth_tfs_p4d_lr1e3.yaml`、`scared_cam_depth_tfs_p4d_lr1e3_cross.yaml`（`per_frame` 版另加 `..._cross_pf.yaml`） |
| 資料 | `data/train/scared`（repo root 的 `data` symlink 指到資料碟） |
| 權重 | `checkpoints/VGGT-1B.pt`（起點）；`flow_geom` 用的 RAFT 是 torchvision 的 `raft_large`，第一次會從網路下載到 `~/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth`，離線機器要先複製過去 |
| 環境 | conda `vggt-dyn`；75 上是 torch 2.8.0+cu128、cuDNN 9.10.2、驅動 580.159.03 |

指令（在 repo root）：

```bash
git checkout p3-cross-gate
# 先跑 smoke 確認這台不會崩（約 7 分鐘，200 步）：config 裡 limit_train_batches 是 60，要 200 步就臨時改大
torchrun --nproc_per_node=1 pipeline/training/launch.py --config scared_cam_depth_tfs_p4d_lr1e3_cross_smoke
# 正式訓練：10 epoch × 1500 步，約 10.5–11.5 小時（不含每 epoch 的 test 評估）
torchrun --nproc_per_node=1 pipeline/training/launch.py --config scared_cam_depth_tfs_p4d_lr1e3_cross
# 崩了從 last.pt 接續（scared_depth_base 已設 save_steps_freq: 1000，每 1000 步覆寫 last.pt）
torchrun --nproc_per_node=1 pipeline/training/launch.py --config scared_cam_depth_tfs_p4d_lr1e3_cross --resume
```

- 輸出在 `logs/scared_cam_depth_tfs_p4d_lr1e3_cross/`；trainer 每個 epoch 評 test 主指標（pose c64、depth 多張），存在 `logs/<exp>/pose_eval/epoch_N/`。
- 若同名 `logs/<exp>/ckpts/last.pt` 已存在，不加 `--resume` 時 trainer 會拒絕啟動。
- 開跑前先看 `df`：根目錄碟滿會弄壞 `last.pt`。
- 對照組是 `tfs`（test c64 0.0609、c5 0.0709、depth 多張 0.0413）。
