# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers
from eval_utils.metrics_val import DEPTH_KEYS, POSE_KEYS, ValMetricsAccumulator
from loss import oracle_gate_logits_from_mask

# TensorBoard tag layout: "<phase>/<group>_<key>" -- train/loss_*, validation/loss_*,
# validation/depth_*, validation/pose_*. The internal phase name is 'val'; TB spells it out.
_TB_PHASE = {"train": "train", "val": "validation"}


def _tb_metric_group(key: str) -> str:
    """Map a flat val-metric key onto its TB group ('depth' / 'pose')."""
    if key in POSE_KEYS:
        return "pose"
    if key in DEPTH_KEYS:
        return "depth"
    return "metric"


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        pose_eval: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        oracle_gate: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
            oracle_gate: If {"enabled": True, ...}, replaces the model's own gate bias with one
                built directly from batch["motion_mask"] (GT dynamic mask) on every forward pass
                (see loss.oracle_gate_logits_from_mask and dyn_vggt_v3_s1_oracle_camera_only.yaml).
                Used to test the v3 architectural bet in isolation from gate-predictor quality.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim
        self.val_metrics_conf = val_metrics or {}
        # MonST3R-style second validation channel: a deterministic full-sequence Sintel
        # pose eval that drives best.pt. Runs on its own (sparser) cadence, separate from
        # the per-epoch windowed val (which becomes monitoring-only when this is enabled).
        self.pose_eval_conf = pose_eval or {}
        self.oracle_gate_conf = oracle_gate
        # Best pose ATE seen so far (drives best.pt); restored on resume.
        self.best_ate = float("inf")

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32
            # The flash-attention SDPA kernel on torch 2.12+cu130 / RTX 5090 (Blackwell) segfaults
            # intermittently (see cuda.disable_flash_sdp note in default.yaml). Disable it so SDPA
            # falls back to the numerically-equivalent, stable mem-efficient kernel.
            if cuda_conf.get("disable_flash_sdp", False):
                torch.backends.cuda.enable_flash_sdp(False)
                logging.info(
                    "Disabled flash SDPA backend (using mem-efficient) — workaround for "
                    "torch 2.12+cu130 flash-attn segfault on RTX 5090 (see outputs/gpu_repro_*.log)."
                )

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            opt_states = checkpoint["optimizer"]
            if isinstance(opt_states, list):
                for optim, state in zip(self.optims, opt_states):
                    optim.optimizer.load_state_dict(state)
            else:
                self.optims[0].optimizer.load_state_dict(opt_states)

        # Load training progress. A checkpoint is either an END-OF-EPOCH save
        # (epoch_completed=True, or a legacy checkpoint without the flag) -> resume at
        # the NEXT epoch; or a MID-EPOCH save (epoch_completed=False) -> resume the SAME
        # epoch and fast-forward the (deterministically-seeded) dataloader to resume_iter.
        if not checkpoint.get("epoch_completed", True):
            self.epoch = int(checkpoint.get("prev_epoch", checkpoint.get("epoch", 0)))
            self._resume_iter = int(checkpoint.get("resume_iter", 0))
            logging.info(
                f"Mid-epoch resume: epoch {self.epoch}, fast-forwarding to iter {self._resume_iter}"
            )
        else:
            if "prev_epoch" in checkpoint:
                self.epoch = checkpoint["prev_epoch"] + 1
            elif "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
            self._resume_iter = 0
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)
        self.best_ate = checkpoint.get("best_ate", float("inf"))

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}
        # Iter within the current epoch to fast-forward to on a mid-epoch resume.
        # 0 = start the epoch from scratch (fresh run or end-of-epoch resume).
        self._resume_iter = 0

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Freeze individual nn.Parameters by substring match (handles params that are not
        # sub-modules, e.g. aggregator.camera_token / aggregator.register_token).
        # frozen_param_names: list of substrings; any parameter whose fully-qualified name
        # contains one of these substrings is frozen (requires_grad = False).
        if getattr(self.optim_conf, "frozen_param_names", None):
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if any(pat in name for pat in self.optim_conf.frozen_param_names):
                    param.requires_grad = False
                    frozen_count += 1
            logging.info(
                f"[Done] Freezing {frozen_count} individual params matching: "
                f"{list(self.optim_conf.frozen_param_names)} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(
        self,
        epoch: int,
        checkpoint_names: Optional[List[str]] = None,
        data_iter: Optional[int] = None,
    ):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
            data_iter: If None, this is an END-OF-EPOCH checkpoint (resume starts the next
                       epoch). If an int, this is a MID-EPOCH checkpoint taken after the
                       optimizer step for that iter -> resume re-enters this same epoch and
                       fast-forwards the dataloader to data_iter + 1.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_ate": self.best_ate,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
            # Mid-epoch resume bookkeeping (see _load_resuming_checkpoint). data_iter is None
            # for the normal end-of-epoch save -> epoch_completed=True, resume_iter unused.
            "epoch_completed": data_iter is None,
            "resume_iter": 0 if data_iter is None else int(data_iter) + 1,
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            # run_train already validates + checkpoints (last/best) every epoch,
            # including the final epoch, so no extra post-train validation is needed.
            self.run_train()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs.

        Checkpoint policy: every epoch saves a rolling ``last.pt`` (latest epoch).
        ``best.pt`` is saved whenever pose ATE improves — from the deterministic
        full-sequence pose eval (channel B, the sole validation signal) when ``pose_eval``
        is enabled, else from the legacy windowed val ATE.
        """
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)

            # Clean up training memory before validation.
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Rolling latest checkpoint.
            self.save_checkpoint(self.epoch, checkpoint_names=["last"])

            # Milestone checkpoints every checkpoint.save_freq epochs, kept permanently
            # alongside the rolling last.pt / best.pt. self.epoch is 0-indexed, so (epoch+1)
            # is the count of completed epochs -> epoch_10.pt, epoch_20.pt, ...
            save_freq = int(self.checkpoint_conf.get("save_freq", 0) or 0)
            if save_freq > 0 and (int(self.epoch) + 1) % save_freq == 0:
                self.save_checkpoint(
                    self.epoch, checkpoint_names=[f"epoch_{int(self.epoch) + 1}"]
                )

            # Two complementary validation channels run every epoch:
            #   channel A (run_val / windowed) -> the val LOSS source (validation/loss_*),
            #     for train/val overfitting monitoring. When channel B is enabled it does NOT
            #     write validation/{pose,depth}_* (B owns the metric tags -- see val_epoch).
            #   channel B (run_pose_eval / full-sequence Sintel) -> the eval METRICS source
            #     (validation/{pose,depth}_*) and the best.pt selection signal (full-seq ATE).
            #   pose_eval disabled -> best.pt falls back to channel A's windowed ATE.
            summary_a = self.run_val()
            if self.pose_eval_conf.get("enabled", False):
                every_n = int(self.pose_eval_conf.get("every_n_epochs", 1) or 1)
                is_last_epoch = int(self.epoch) + 1 >= self.max_epochs
                if (int(self.epoch) + 1) % every_n == 0 or is_last_epoch:
                    ate = self.run_pose_eval()
                else:
                    ate = None
            else:
                ate = summary_a.get("ate") if isinstance(summary_a, dict) else None
            if ate is not None and ate < self.best_ate:
                logging.info(
                    "New best pose ATE %.4f (prev %.4f) at epoch %s -> saving best.pt",
                    ate, self.best_ate, self.epoch,
                )
                self.best_ate = float(ate)
                self.save_checkpoint(self.epoch, checkpoint_names=["best"])

            self.epoch += 1

        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available.

        Returns the metric summary dict (e.g. {"ate": ..., "abs_rel": ...}),
        or an empty dict when no val dataset / metrics are configured.
        """
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return {}

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        summary = self.val_epoch(dataloader)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return summary

    @torch.no_grad()
    def run_pose_eval(self) -> Optional[float]:
        """Deterministic full-sequence Sintel pose eval on the live model (MonST3R channel B).

        Scores the in-memory model on fixed, full-length Sintel sequences (no random
        [4,16] windowing, no re-sampling across epochs) so the resulting ATE is a stable
        trend/selection signal. Returns the mean ATE (rank 0 computes; broadcast to all
        ranks so best.pt selection stays consistent). Returns None on failure.
        """
        from types import SimpleNamespace

        from benchmark.eval_sintel import evaluate as eval_sintel_evaluate

        cfg = self.pose_eval_conf
        model = self.model.module if isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        ) else self.model

        was_training = model.training
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        ate_val = torch.full((1,), float("inf"), device=self.device)
        try:
            if self.rank == 0:
                out_dir = os.path.join(
                    self.logging_conf.log_dir, "pose_eval", f"epoch_{int(self.epoch) + 1}"
                )
                args = SimpleNamespace(
                    ckpt=f"live_epoch_{int(self.epoch) + 1}",
                    sintel_root=cfg.get("sintel_root", None),
                    out_dir=out_dir,
                    seq_list=cfg.get("seq_list", None),
                    device=self.device if isinstance(self.device, str) else "cuda",
                    chunk_size=int(cfg.get("chunk_size", 0) or 0),
                    max_depth=float(cfg.get("max_depth", 80.0)),
                )
                results = eval_sintel_evaluate(args, model=model)
                ate = results.get("pose", {}).get("mean", {}).get("ate", None)
                if ate is not None and ate > 0:
                    ate_val[0] = float(ate)
                logging.info(
                    "Pose eval (full-seq Sintel) at epoch %s: ATE=%.4f", self.epoch, ate_val.item()
                )
                # Channel B is now the sole validation signal -> log its full-sequence mean
                # metrics to TensorBoard under the same validation/{pose,depth}_* tags (drop-in
                # replacing the removed windowed channel A) so the TB curves are all full-sequence.
                for group in ("pose", "depth"):
                    for key, val in results.get(group, {}).get("mean", {}).items():
                        self.tb_writer.log(f"validation/{group}_{key}", val, self.epoch)
        finally:
            if was_training:
                model.train()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Share rank-0's ATE with every rank so best.pt selection is identical everywhere.
        if is_dist_avail_and_initialized():
            dist.broadcast(ate_val, src=0)
        ate = ate_val.item()
        return ate if ate != float("inf") else None

    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        metric_meters = {}
        metrics_tracker = None
        if self.val_metrics_conf.get("enabled", False):
            metrics_tracker = ValMetricsAccumulator(
                max_depth=float(self.val_metrics_conf.get("max_depth", 80.0)),
                min_depth_pixels=int(self.val_metrics_conf.get("min_depth_pixels", 100)),
            )
            for key in ("abs_rel", "delta_1", "rmse", "ate", "rpe_trans", "rpe_rot"):
                metric_meters[f"Metric/val_{key}"] = AverageMeter(
                    f"Metric/val_{key}", self.device, ":.4f"
                )
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
                *metric_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.amp.autocast('cuda',enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.amp.autocast('cuda',
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    _, y_hat = self._step(
                        batch, self.model, phase, loss_meters
                    )
                    if metrics_tracker is not None:
                        batch_metrics = metrics_tracker.update(y_hat, batch)
                        bs = batch["extrinsics"].shape[0]
                        for key, val in batch_metrics.items():
                            meter_key = key.replace("metric_", "Metric/val_")
                            if meter_key in metric_meters:
                                metric_meters[meter_key].update(val, bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        summary = {}
        if metrics_tracker is not None and self.rank == 0:
            summary = metrics_tracker.compute()
            if summary:
                parts = []
                if "abs_rel" in summary:
                    parts.append(
                        f"depth AbsRel={summary['abs_rel']:.4f} "
                        f"d1={summary['delta_1']:.4f} RMSE={summary['rmse']:.4f} "
                        f"({int(summary.get('depth_frames', 0))} frames)"
                    )
                if "ate" in summary:
                    parts.append(
                        f"pose ATE={summary['ate']:.4f} "
                        f"RPE-t={summary['rpe_trans']:.4f} "
                        f"RPE-r={summary['rpe_rot']:.4f} "
                        f"({int(summary.get('pose_seqs', 0))} seqs)"
                    )
                logging.info("Val Epoch: [%s] metrics | %s", self.epoch, " | ".join(parts))
                # Channel A always prints its windowed metrics to the log, but only writes
                # them to TensorBoard when channel B (pose_eval) is NOT the metric source.
                # When B is enabled it owns validation/{pose,depth}_* (full-sequence); A stays
                # loss-only on TB to avoid clobbering the same tags at the same epoch step.
                if not self.pose_eval_conf.get("enabled", False):
                    for key, val in summary.items():
                        if key.endswith("_frames") or key.endswith("_seqs"):
                            continue
                        self.tb_writer.log(
                            f"validation/{_tb_metric_group(key)}_{key}", val, self.epoch
                        )

        # Return the metric summary (ATE etc.) so the training loop can drive
        # best-checkpoint selection. Empty dict when metrics are disabled.
        return summary

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        # Mid-epoch checkpointing / resume. start_iter is consumed once: only the first
        # epoch after a mid-epoch resume fast-forwards; later epochs start at 0. The epoch
        # is deterministically re-seeded in run_train (set_seeds(seed + epoch*100)), so
        # skipping to start_iter lands on the same samples the crashed run had reached.
        start_iter = getattr(self, "_resume_iter", 0)
        self._resume_iter = 0
        save_steps_freq = int(self.checkpoint_conf.get("save_steps_freq", 0) or 0)
        if start_iter > 0:
            logging.info(
                f"Fast-forwarding train dataloader to iter {start_iter} (mid-epoch resume)"
            )

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            # Skip already-completed iters on a mid-epoch resume (keep timers fresh).
            if data_iter < start_iter:
                end = time.time()
                continue

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.amp.autocast('cuda',enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            # Rolling mid-epoch checkpoint: overwrite last.pt every save_steps_freq iters so a
            # crash mid-epoch resumes from here (fast-forward to data_iter+1) instead of losing
            # the whole epoch. save_steps_freq <= 0 disables (default -> original behaviour).
            # Cost: one full ~7 GB last.pt write per trigger; robust_torch_save keeps a .bak so
            # a crash during the write can't corrupt the previous good checkpoint.
            if save_steps_freq > 0 and data_iter > 0 and data_iter % save_steps_freq == 0:
                self.save_checkpoint(
                    self.epoch, checkpoint_names=["last"], data_iter=data_iter
                )

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.amp.autocast('cuda',
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict, _ = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points. The function returns new tensors.
        # MODIFIED (B6): also normalises scene_flow_gt in lock-step when present (Dyn-VGGT).
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths, normalized_scene_flow = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
                scene_flow=batch.get("scene_flow_gt", None),
            )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths
        if normalized_scene_flow is not None:
            batch["scene_flow_gt"] = normalized_scene_flow

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        gate_override = None
        if self.oracle_gate_conf and self.oracle_gate_conf.get("enabled", False) and "motion_mask" in batch:
            gate_override = oracle_gate_logits_from_mask(
                batch["motion_mask"],
                patch_size=self.oracle_gate_conf.get("patch_size", 14),
                k=self.oracle_gate_conf.get("k", 30.0),
            )
        y_hat = model(images=batch["images"], gate_logits_override=gate_override)
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict, y_hat

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"{_TB_PHASE.get(phase, phase)}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data

