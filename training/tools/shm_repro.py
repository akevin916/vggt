#!/usr/bin/env python3
"""Reproduce the host crash WITHOUT touching CUDA -- test #2 of docs/ops/machine_error.md §6.

WHAT THIS DECIDES. The crash family in machine_error.md has surfaced in at least six places
(libcuda's autograd thread, rope forward, aggregator, cuda.synchronize, a DataLoader worker,
kswapd), which is why "it died in the DataLoader, so fix the DataLoader" keeps being the wrong
inference. The one question the history cannot answer is whether the NVIDIA stack is involved
at all. This script answers it, because it never loads a CUDA context:

    crashes within ~30 min  ->  kernel + RAM alone are sufficient. The NVIDIA driver is
                                exonerated, and the remaining suspects are the 6.8.0-124
                                shmem/XArray path and the memory subsystem (§5).
    runs clean while training crashes  ->  the driver (or the GPU under load) is required to
                                trigger it, and memtest/kernel-downgrade drop in priority.

A clean run is WEAKER evidence than a crash: it bounds the rate, it does not prove absence.
Read it as "not reproducible in N minutes at this churn rate", nothing more.

WHY SHARED-MEMORY CHURN IS THE RIGHT PROBE. The eight recorded kernel oopses all fault in
    xas_init_marks <- truncate_inode_folio <- shmem_undo_range <- __fput
i.e. the kernel evicting a /dev/shm inode as the last file descriptor closes. That is exactly
what PyTorch does between DataLoader workers: every batch allocates a shared tensor and drops
it. This script does the same thing with no model, no dataset, no GPU -- just the syscall
pattern. Note this is about the kernel walking a page-cache XArray, NOT about /dev/shm running
out of space (63 GB here, 2% used); the two are unrelated despite sharing a name.

SHARING STRATEGY MATTERS. `file_descriptor` (PyTorch's Linux default, and what the real
DataLoader uses) passes fds over a unix socket and is the faithful reproduction.
`file_system` creates and unlinks visible /dev/shm entries, which hammers the same kernel path
harder and is the more aggressive probe. Try the default first; if it survives, re-run with
--strategy file_system before concluding anything.

SAFE TO RUN ALONGSIDE TRAINING. CPU and shared memory only -- CUDA_VISIBLE_DEVICES is cleared
before torch is imported, so no context is ever created and the GPU is untouched. Bound the
footprint with --workers/--mb if the box is busy.

IT LOGS TO A FILE AND FSYNCS EVERY HEARTBEAT. A hard freeze (the 19:08 kswapd lockup left no
userspace trace at all) takes the terminal with it, so the last flushed heartbeat on disk is
the only record of how far it got.

Usage (from training/):
    python tools/shm_repro.py                          # 30 min, 8 workers, DataLoader-like
    python tools/shm_repro.py --minutes 60 --workers 12 --strategy file_system
    tail -f outputs/shm_repro/shm_repro.log            # from another shell

Exit codes:  0 = survived the full duration    1 = a worker died (the reproduction)
"""

from __future__ import annotations

# Must precede `import torch`: an empty device list means no CUDA context can be created, which
# is the entire point of this test. Do not move this below the imports.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import queue
import signal
import sys
import time
from datetime import datetime

import torch
import torch.multiprocessing as mp


def _worker(wid: int, q: mp.Queue, elems: int, stop: "mp.Event", counter) -> None:
    """Allocate a fresh shared tensor, hand it over, drop it. Repeat until told to stop.

    The tensor is filled rather than left uninitialised so the pages are actually faulted in --
    an untouched allocation would never reach the page-cache paths this test is probing.
    """
    torch.manual_seed(wid)
    while not stop.is_set():
        t = torch.empty(elems, dtype=torch.float32)
        t.fill_(float(wid))
        t.share_memory_()          # materialise the shm segment
        try:
            q.put(t, timeout=5.0)  # sending moves the handle; the parent's drop unlinks it
        except queue.Full:
            pass
        del t
        with counter.get_lock():
            counter.value += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8,
                    help="producer processes (default 8 = default_dataset.yaml's num_workers)")
    ap.add_argument("--mb", type=float, default=40.0,
                    help="MB per shared tensor (default 40 ~= one 12-frame 518px image batch)")
    ap.add_argument("--minutes", type=float, default=30.0, help="duration (default 30)")
    ap.add_argument("--strategy", default="file_descriptor",
                    choices=["file_descriptor", "file_system"],
                    help="torch sharing strategy; file_descriptor matches the real DataLoader")
    ap.add_argument("--queue-depth", type=int, default=16,
                    help="bounded so a slow consumer cannot grow the footprint without limit")
    ap.add_argument("--heartbeat", type=float, default=10.0, help="seconds between log lines")
    ap.add_argument("--out_dir", default=None,
                    help="default: outputs/shm_repro/ relative to the repo root")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir or os.path.join(repo_root, "outputs", "shm_repro")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "shm_repro.log")
    log_fh = open(log_path, "a", buffering=1)

    def log(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())   # survive a hard freeze

    mp.set_sharing_strategy(args.strategy)
    # fork, not spawn: the real DataLoader forks on Linux, and fork is what puts the parent's
    # already-mapped segments in play. Switching to spawn would test a different thing.
    ctx = mp.get_context("fork")

    elems = max(1, int(args.mb * 1024 * 1024 / 4))
    log("=" * 78)
    log(f"shm repro | workers={args.workers} mb={args.mb} strategy={args.strategy} "
        f"minutes={args.minutes} pid={os.getpid()}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"torch={torch.__version__} cuda_initialized={torch.cuda.is_initialized()}")
    log(f"log -> {log_path}")

    q: mp.Queue = ctx.Queue(maxsize=args.queue_depth)
    stop = ctx.Event()
    counter = ctx.Value("Q", 0)

    procs = [ctx.Process(target=_worker, args=(i, q, elems, stop, counter), daemon=True)
             for i in range(args.workers)]
    for p in procs:
        p.start()
    log(f"workers up: {[p.pid for p in procs]}")

    deadline = time.time() + args.minutes * 60.0
    next_beat = time.time() + args.heartbeat
    consumed = 0
    last_consumed = 0
    last_beat_t = time.time()
    verdict = 0

    def _sigint(_sig, _frm):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint)

    try:
        while time.time() < deadline:
            try:
                t = q.get(timeout=1.0)
                # Touch it so the pages are real, then drop -- the drop is what triggers the
                # inode eviction this test is about.
                _ = float(t[0]) + float(t[-1])
                del t
                consumed += 1
            except queue.Empty:
                pass

            dead = [(p.pid, p.exitcode) for p in procs if p.exitcode is not None]
            if dead:
                for pid, code in dead:
                    how = f"signal {-code} ({signal.Signals(-code).name})" if code < 0 else f"exit {code}"
                    log(f"*** WORKER DIED: pid={pid} {how}")
                log(f"*** REPRODUCED after {consumed} consumed / {counter.value} produced, "
                    f"{(time.time() - (deadline - args.minutes * 60.0)) / 60.0:.1f} min")
                log("*** No CUDA context existed in this process tree -> the NVIDIA stack is "
                    "not required to trigger the fault.")
                verdict = 1
                break

            now = time.time()
            if now >= next_beat:
                rate = (consumed - last_consumed) / max(1e-9, now - last_beat_t)
                elapsed = (now - (deadline - args.minutes * 60.0)) / 60.0
                log(f"alive {elapsed:6.1f} min | consumed={consumed:>8d} "
                    f"produced={counter.value:>8d} | {rate:7.1f} tensor/s "
                    f"({rate * args.mb / 1024:5.2f} GB/s)")
                last_consumed, last_beat_t, next_beat = consumed, now, now + args.heartbeat
    except KeyboardInterrupt:
        log("interrupted by user")
        verdict = 130
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        # Drain so the queue's feeder thread can exit without blocking at shutdown.
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass

    if verdict == 0:
        log(f"SURVIVED {args.minutes} min | consumed={consumed} produced={counter.value}")
        log("Not reproducible at this rate. This BOUNDS the failure rate; it does not prove "
            "absence. Re-run with --strategy file_system before concluding the driver is "
            "implicated (see the docstring).")
    log("=" * 78)
    log_fh.close()
    return verdict


if __name__ == "__main__":
    sys.exit(main())
