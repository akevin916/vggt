# 主機不穩定 —— 內核崩潰調查

> **狀態：未解決。** 2026-06-26 起 8 次內核 oops，**全部同一個簽名**。故障間隔從數天縮短到 23 分鐘，
> 目前無法連續跑完一個 26 小時的訓練。本檔記錄症狀、已排除的原因、以及尚未做的決定性測試。
>
> 這不是 Dyn-VGGT 的程式問題。任何在這台機器上跑的長時間工作都會受影響。

---

## 1. 症狀：看起來像「訓練卡住」，實際是主程序被殺掉

崩潰時**沒有 traceback、沒有錯誤訊息**，log 從中間直接斷掉。終端機視窗還開著，所以外觀上像是卡住。實際狀態是：

| 現象 | 原因 |
|---|---|
| log 停止更新，最後一行是正常的訓練步 | 主程序（torchrun 的子程序）被內核殺掉 |
| `torchrun` 還活著、在 `hrtimer_nanosleep` | 它的監控迴圈被孤兒 worker 的管線卡住 |
| 4 個 dataloader worker 存在但 PPID 指向不存在的程序 | 主程序死了，worker 變孤兒，卡在 `do_poll` |
| GPU 顯示 28~32 GB 佔用、**0% 使用率** | 死程序的 CUDA context 未被回收 |
| `pgrep -f launch.py` 有結果 | **會誤判**——那是孤兒 worker，不是在訓練 |

⚠️ 用 `pgrep -f "launch.py"` 判斷「訓練是否還在跑」會得到錯誤答案，理由有二：它會匹配到孤兒 worker，也會匹配到你自己那條含有 `launch.py` 字串的 `bash -c` 指令。**判斷是否真的在跑，看 log 的 mtime 與 GPU 使用率**：

```bash
stat -c '%y' logs/<exp>/log.txt      # 是否在數秒內更新
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # 0% 持續 = 死了
```

---

## 2. 崩潰簽名（8 次完全相同）

```
BUG: kernel NULL pointer dereference / unable to handle page fault
RIP: 0010:xas_init_marks+0xa/0x60        （或 +0x10，同一函式）
Call Trace:
  __filemap_remove_folio
  filemap_remove_folio
  truncate_inode_folio
  shmem_undo_range          ← 共享記憶體（/dev/shm）
  shmem_evict_inode
  evict → iput → dentry_unlink_inode → __dentry_kill → dput
  __fput → ____fput         ← 關檔案時觸發
  task_work_run → syscall_exit_to_user_mode
```

觸發點是 **`/dev/shm` 上的檔案被關閉、inode 被回收時**，內核走 page-cache 的 XArray 踩到無效指標。

`shmem` 正是 **PyTorch DataLoader 在 worker 之間傳張量的機制**——每個 batch 都在建立與銷毀共享記憶體。這解釋了為什麼訓練是最容易觸發的工作負載，但不代表訓練是原因。

### 錯誤位址：8 次全部不同

| # | 時間 | 錯誤位址 |
|---|---|---|
| 1 | 2026-06-26 01:37 | `ffffffffffffffff` |
| 2 | 2026-07-01 19:52 | `0000000000000004` |
| 3 | 2026-07-04 02:37 | `00000000ffffffd7` |
| 4 | 2026-07-10 17:21 | `000000000000002a` |
| 5 | 2026-07-25 09:10 | `0000000000000003` |
| 6 | 2026-08-11 00:22 | `0000000000000017` |
| 7 | 2026-08-11 23:41 | `00000000ffffffff` |
| 8 | 2026-08-12 10:45 | `000000000000003e` |

**同一個指令、8 個不同的垃圾值。** 純軟體的 NULL 解參考會固定落在 0 或某個結構的固定偏移；讀到互不相關的垃圾，代表**被讀的那份資料本身是壞的**——不論成因是記憶體損壞還是 use-after-free。

### 附帶事件

**2026-06-27 05:23–05:36**：`native_queued_spin_lock_slowpath` / `smp_call_function_many_cond` 的
**74 次 soft lockup + 12 次 RCU stall**，機器等同宕機。應為某次損壞之後的連鎖反應。

---

## 3. 故障率正在惡化

| 期間 | MTBF（平均故障間隔） |
|---|---|
| 2026-06-26 ~ 07-25 | 3~11 天 |
| 2026-08-11 | 23 小時 |
| 2026-08-12 | **開始訓練後 23 分鐘** |

**對照組**：8/04~8/10 的三個 egoflow run 各連續跑 26~32 小時**全部完成、零崩潰**——同一台機器、同一套 dataloader、同樣的 shm 使用模式。所以「8 月起惡化」不是曝光量偏差，是真的變差了。

若故障近似隨機發生，長度 L 的工作完成機率 ≈ `exp(−L / MTBF)`：

| MTBF | 26 小時 run 的完成機率 |
|---|---|
| 3 天 | 70% |
| 23 小時 | 32% |
| **23 分鐘** | **≈ 0%** |

---

## 4. 已排除的原因

| 假設 | 排除依據 |
|---|---|
| **記憶體 XMP / EXPO 超頻** | **一直是關的**（使用者 2026-08-13 確認）。等於這個實驗已經跑了整段歷史 |
| kernel / NVIDIA 驅動更新引入 | 6.8.0-124 自 **2026-05-26** 安裝、6/09 起使用；崩潰始於 6/26。期間 apt 只有 tailscale（8/08）與 linux-firmware 重裝（8/11，是崩潰**之後**的反應） |
| flash-attn（既有 workaround 的對象） | 那是 **userspace SIGSEGV/SIGBUS**，本案是**內核 oops**，層級不同。且崩潰時 `cuda.disable_flash_sdp: True` 是開著的 |
| 主機 RAM 耗盡（OOM killer） | 125 GB 總量、崩潰時可用 110 GB；OOM kill 會在日誌留下明確記錄，此處沒有 |
| GPU 顯存不足 | CUDA OOM 會拋 Python 例外並留下 traceback；此處 log 直接斷掉 |
| 訓練 config 或新加的 loss | 崩潰始於 6/26，早於本 repo 最早的訓練 log（7/08）；且 8 月初同樣的 config 連跑 90 小時無事 |

---

## 5. 尚存的兩種解釋

| | 支持 | 反對 |
|---|---|---|
| **記憶體 / 記憶體控制器劣化** | 故障率暴增（天 → 分鐘）；8 個互不相關的垃圾位址；6/27 的 spinlock 風暴涉及其他內核路徑 | 最近 8 次**全部**落在同一條 shmem 路徑——隨機位元翻轉應該落點分散 |
| **內核 6.8.0-124 的 shmem/XArray bug（例如 use-after-free）** | 8 次簽名完全一致；UAF 讀到的殘留資料也會是變動的垃圾值 | 為何 8/04~8/10 能連跑 90 小時無事？期間無任何更新 |

**目前無法在兩者之間定案**，這是本檔的核心未解問題。

### 一個被高估的證據（更正）

先前把 `BERT: [Hardware Error]` 當成「韌體每次崩潰都記錄到硬體錯誤」的強證據。**這個判讀是錯的**：

- 8 次記錄的**實體位址完全相同**（`0x415d8000-0x415d802f`）
- 最早出現在 **6/09**，早於第一次 oops（6/26）

同一個位址反覆出現，代表那是韌體裡**沒有被清除的舊記錄**在每次開機被重新回報，不是每次崩潰的新錯誤。這條證據應視為**中性**。

### 另一個限制

journal 只回溯到 2026-06-08，其中 6.8.0-87 只用了約一天（6/08~6/09），之後全程 6.8.0-124。
**沒有足夠長的 -87 基線可以當對照**，所以「舊 kernel 是否也會崩」目前無法從歷史紀錄回答。

---

## 6. 尚未執行的測試（依資訊量排序）

| # | 測試 | 為什麼值得做 | 成本 |
|---|---|---|---|
| 1 | **memtest86+** | 以 23 分鐘的 MTBF，若是記憶體問題應該很快就抓到。（注意：故障率低時 memtest 通過**不能**排除問題，但現在的頻率讓它有意義） | 開機 → Shift → Advanced options → memtest，一輪數小時 |
| 2 | **無 GPU 的 shm 重現腳本** | 多程序高頻建立/銷毀 shared tensor，**完全不碰 CUDA**。若 20 分鐘內崩 → 與 NVIDIA 驅動無關，純內核＋RAM；若不崩而訓練會崩 → 驅動涉入 | 需寫，測試循環約 30 分鐘 |
| 3 | **開 6.8.0-87 跑長時間工作** | 舊 kernel 仍安裝（2025-11-20），GRUB 可選。不崩 → -124 的回歸；仍崩 → 硬體 | 數小時～數天才有結論 |
| 4 | 拔掉部分記憶體 / 逐條測試 | memtest 若指出錯誤可定位到條 | 需開機殼 |

---

## 7. 崩潰之後的清理程序

死程序會卡住 GPU 顯存，孤兒 worker 會卡住 torchrun。**最乾淨的做法是重開機**；若要就地清理：

```bash
pkill -f "launch.py --config <exp>"
pkill -f "torchrun .* --config <exp>"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # 確認顯存釋放
```

若 `nvidia-smi` 仍列出已不存在的 PID 佔著顯存，只能重開機。

---

## 8. 訓練側的應對

**沒有任何軟體設定能修好這個問題**，只能降低每次崩潰的代價：

- **`checkpoint.save_steps_freq: 500`**（[trainer.py:888](../training/trainer.py#L888)）—— 每 N 步存一次 `last.pt`，崩潰最多損失約 15 分鐘而非整個 epoch。存檔會寫 `epoch_completed=False` 與 `resume_iter`，`--resume` 會快轉 dataloader 接回同一個 epoch。代價每次約 30 秒、7.8 GB。
  *（2026-08-13 使用者判斷暫不啟用：epoch 級存檔最多損失一個 epoch ≈ 1.5 小時，可接受。）*
- **`--resume`** 續訓：接自己的 `logs/<exp>/ckpts/last.pt`，而不是 config 裡的 warm start。LR 排程依絕對 epoch 計算，會正確接上。
- 規劃實驗時把 MTBF 算進去：現況下**不要設計需要連續 20 小時以上的 run**。

---

## 9. 診斷指令

```bash
# 至今的 oops 次數（數字沒增加 = 沒再犯）
journalctl 2>/dev/null | grep -cE "Oops: [0-9]+ \[#"

# 每次 oops 的時間與錯誤位址
journalctl 2>/dev/null | grep -E "kernel: (BUG: (kernel NULL|unable)|RIP: 0010:xas_init)"

# 某次崩潰的完整 call trace
journalctl --since "<YYYY-MM-DD HH:MM>" --until "<...>" | grep -A25 "Call Trace"

# 各次開機用的 kernel 與該次是否崩潰
for b in $(journalctl --list-boots | awk '{print $1}'); do
  echo "$(journalctl -b $b | grep -m1 -oE 'Linux version 6\.8\.0-[0-9]+')  oops=$(journalctl -b $b | grep -cE 'Oops: [0-9]+ \[#')"
done

# 記憶體規格（需 root）
sudo dmidecode -t 17 | grep -E "Size|Speed|Configured|Locator|Part Number"
```

---

## 10. 時間線

| 日期 | 事件 |
|---|---|
| 2026-05-26 | kernel 6.8.0-124 安裝 |
| 2026-06-08~09 | journal 起點，使用 6.8.0-87 |
| 2026-06-09 15:15 | 切換到 6.8.0-124（此後全程使用） |
| **2026-06-26 01:37** | **第一次 oops** |
| 2026-06-27 05:23 | soft lockup 風暴，機器宕機 |
| 07-01 / 07-04 / 07-10 / 07-25 | 第 2~5 次 oops，間隔 3~11 天 |
| 2026-08-04~10 | 三個 egoflow run 各跑 26~32 小時，**全部完成無崩潰** |
| 2026-08-11 00:22 | 第 6 次，殺掉 `smooth_o1` 的 epoch 4 |
| 2026-08-11 23:41 | 第 7 次，殺掉續訓的 epoch 13 |
| 2026-08-12 10:45 | 第 8 次，**開始訓練後僅 23 分鐘** |
| 2026-08-13 | 確認 XMP 一直是關的 → 排除超頻假設 |
| 2026-08-14 14:38 | 重開機，此後未再跑長時間工作，無新 oops |
