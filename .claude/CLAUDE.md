# Execution Policy

- Never execute python script automatically.
- Always output the command first.
- Wait for my confirmation before execution.
- Do not directly run long-running programs (e.g. full training / long eval). A short smoke test is fine.

# Visualization

- Never use plt.show().
- Save every generated image inside the repository.
- Put all visualization output under the repo-root `outputs/`:
  - outputs/
  - outputs/debug/
- Print the saved image path after saving.