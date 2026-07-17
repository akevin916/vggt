# Execution Policy

- Always output the command first.
- Wait for my confirmation before execution.
- Do not directly run long-running programs (e.g. full training / long eval). A short smoke test is fine.

# Visualization

- Never use plt.show().
- Save every generated image inside the repository.
- Print the saved image path after saving.

# Output paths

Two sinks, split by *who writes it*:

- `training/logs/<exp>/` —— **訓練產物only**：`ckpts/`、`tensorboard/`、`log.txt`、trainer 自己的 `pose_eval/`。
- `outputs/<exp>/<tool>/` —— **訓練之後產生的一切**：benchmark 數字、ablation json、診斷、圖。

`<exp>` 由 ckpt 推導，一律走 `eval_utils/paths.py`（`default_output_dir(ckpt, TOOL)`，或 `output_dir_for_exp(exp_name, TOOL)` 給沒有 ckpt 的 caller）——**不要自己拼路徑**。`<exp>` 那層讓圖自帶 ckpt 身分，換 ckpt 重跑不會靜默覆蓋，圖和 json 也能用同一把鑰匙對起來。