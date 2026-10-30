# 方向 3 gradient probe (planning/方向3_StereoMIS器械遮罩訓練.md §3). Every `every` optimizer steps,
# one extra forward/backward on a FIXED probe clip, never touching param .grad (torch.autograd.grad
# on chosen inputs only). The clip is built once from the dataset, no augmentation: sequence
# conf.seq (default P2_0, the one with the most instrument), n_frames rows `stride` apart, at the
# first place where every row's instrument fraction lies in conf.inst_range (default 0.15-0.25),
# so dL/db is measured on keys that actually exist. Same clip for R0/R1/R2.
#
# The probe forward is UNMASKED for every group (R0/R1/R2): it asks what the current model's pose
# loss wants, not what the masked model sees. Per global block l (0-23) it records
#   dLdb      d objective / d b_l, b_l = 0 a scalar bias on every (query, other-frame instrument
#             key) logit. < 0: the pose loss wants those keys attended LESS.
#   g_cam     mean |dL/d tokens| at the camera tokens of block l's output
#   g_D, g_C  same, mean over instrument / non-instrument patch tokens
#   logit_range_p50/p99   per query row, max logit - median logit (all keys)
#   dgap_p99  per row, max logit over other-frame instrument keys - max over other-frame
#             non-instrument keys
#   rho_cam, rho_patch    attention mass on other-frame instrument keys / mass on all other-frame
#             keys, divided by their count ratio (1 = attended in proportion to their number)
#   resid30_max/p999      mass the instrument keys would still get with -30 on their logits
#             (the eval-time soft mask of the mask grid). The training mask is boolean, so
#             its own residual mass is exactly 0 by construction and is not logged.
# Rows use all camera queries + 512 fixed random patch queries. Output: <log_dir>/grad_probe.jsonl.
import gc
import json
import logging
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from pipeline.data.instrument_mask import patch_flags


class GradProbe:
    def __init__(self, trainer, batch, conf):
        self.trainer = trainer
        self.path = os.path.join(trainer.logging_conf.log_dir, "grad_probe.jsonl")
        self.batch = self._build_clip(trainer, conf, batch["images"].device)
        clip = self.batch
        k = clip["images"].shape[1]
        self.flags = patch_flags(clip["motion_mask"], patch_size=14)          # [1, k, P_patch]
        n_patch = self.flags.shape[-1]
        g = torch.Generator().manual_seed(0)
        self.patch_q = torch.randint(0, k * n_patch, (512,), generator=g)    # (frame, patch) flat
        logging.info(f"grad_probe: clip {clip.get('seq_name')} ids {clip['ids'].tolist()}, "
                     f"instrument patch frac {self.flags.float().mean().item():.4f}")

    @staticmethod
    def _build_clip(tr, conf, device):
        import numpy as np
        from omegaconf import OmegaConf
        from torch.utils.data import default_collate
        from pipeline.data.datasets.stereomis import StereoMISDataset
        from pipeline.data.composed_dataset import ComposedDataset
        from pipeline.data.paths import data_path
        seq, n, stride = str(conf.get("seq", "P2_0")), int(conf.get("n_frames", 4)), int(conf.get("stride", 3))
        lo, hi = conf.get("inst_range", [0.15, 0.25])
        root = conf.get("root", None) or data_path("train", "stereomis")
        cc = OmegaConf.merge(tr.data_conf.val.common_config, {"img_size": 518, "patch_size": 14})
        ds = StereoMISDataset(cc, split="train", ROOT=root, seqs=[seq])
        inst = np.loadtxt(os.path.join(ds.data_store[seq]["seq_dir"], "cam_data", "inst_frac.txt"))
        rows = next(r0 + stride * np.arange(n) for r0 in range(len(inst) - stride * n)
                    if np.all((inst[r0 + stride * np.arange(n)] >= lo) & (inst[r0 + stride * np.arange(n)] <= hi)))
        # route through ComposedDataset.__getitem__'s tensor conversion, no colour aug (val mode)
        comp = ComposedDataset.__new__(ComposedDataset)
        comp.base_dataset = type("_One", (), {"__getitem__": lambda self_, _t: ds.get_data(
            seq_name=seq, ids=rows, aspect_ratio=420 / 518)})()
        comp.fixed_num_images, comp.training, comp.image_aug, comp.load_track = 0, False, None, False
        sample = comp[(0, n, 420 / 518)]
        b = tr._process_batch(default_collate([sample]))
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def run(self, step, key_mask_start=None, write=True):
        """key_mask_start: None = the unmasked probe (what training logs). An int runs the probe
        forward WITH the instrument key_mask from that global block on (offline 方向 3 analysis);
        dLdb is then exactly 0 in the masked blocks, whose biased keys are all -inf."""
        tr = self.trainer
        for optim in tr.optims:                 # step already taken; frees ~3 GB of grads
            optim.zero_grad(set_to_none=True)
        model = tr.model.module if isinstance(tr.model, torch.nn.parallel.DistributedDataParallel) else tr.model
        agg = model.aggregator
        dev = self.batch["images"].device
        n_glob = len(agg.global_blocks)
        b = [torch.zeros((), device=dev, requires_grad=True) for _ in range(n_glob)]
        probe = {"b": b, "flags": self.flags, "qk": {}, "out": {}}
        frame_pos = self.batch["ids"] if tr.temporal_frame_pos else None
        batch = dict(self.batch)
        try:
            with sdpa_kernel([SDPBackend.MATH]), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if key_mask_start is None:
                    y = model(images=batch["images"], frame_pos=frame_pos, probe=probe)
                else:
                    agg.key_mask_start = int(key_mask_start)
                    y = model(images=batch["images"], frame_pos=frame_pos, probe=probe, key_mask=self.flags)
                loss = tr.loss(y, batch)["objective"]
                outs = [probe["out"][l] for l in range(n_glob)]
                grads = torch.autograd.grad(loss, b + outs, allow_unused=True)
        except BaseException:
            # an OOM mid-forward would otherwise leave the probe/graph cycle (see cleanup below) alive
            probe["out"].clear(); probe["qk"].clear(); probe["b"] = None
            gc.collect(); torch.cuda.empty_cache()
            raise
        gb, gout = grads[:n_glob], grads[n_glob:]

        S, Pp = self.flags.shape[1:]
        ps = agg.patch_start_idx
        P = Pp + ps
        flag_tok = torch.zeros(S, P, dtype=torch.bool, device=dev)
        flag_tok[:, ps:] = self.flags[0]
        flag_tok = flag_tok.view(-1)
        frame = torch.arange(S * P, device=dev) // P
        q_idx = torch.cat([torch.arange(S, device=dev) * P,
                           (self.patch_q.to(dev) // Pp) * P + ps + self.patch_q.to(dev) % Pp])
        n_cam = S
        rec = {"step": int(step), "loss": float(loss), "per_layer": []}
        for l in range(n_glob):
            r = {"l": l, "dLdb": float(gb[l]) if gb[l] is not None else None}
            go = gout[l]
            if go is not None:
                nrm = go[0].float().norm(dim=-1)                                # [S*P]
                cam = nrm.view(S, P)[:, 0]
                pat = nrm.view(S, P)[:, ps:].reshape(-1)
                fl = self.flags[0].reshape(-1)
                r["g_cam"] = float(cam.mean())
                r["g_D"] = float(pat[fl].mean()) if fl.any() else None
                r["g_C"] = float(pat[~fl].mean())
            r.update(self._attn_stats(*probe["qk"][l], q_idx, n_cam, frame, flag_tok))
            rec["per_layer"].append(r)
        # The checkpointed block closures capture `probe`, and probe["out"] holds tensors whose
        # graph holds those closures: a cycle that keeps every probe activation alive (it left
        # ~20 GB behind and OOMed the epoch-end eval). Emptying the dict breaks it.
        probe["out"].clear(); probe["qk"].clear(); probe["b"] = None
        del y, loss, grads, probe, outs, gout, gb
        gc.collect()
        torch.cuda.empty_cache()
        rec["mem_after_gb"] = torch.cuda.memory_allocated() / 1e9   # leak check
        if write and tr.rank == 0:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    @torch.no_grad()
    def _attn_stats(self, q, k, q_idx, n_cam, frame, flag_tok):
        q, k = q[0].float(), k[0].float()                                       # [H, N, D]
        logits = (q[:, q_idx] @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5      # [H, Q, N]
        cross = frame[q_idx][:, None] != frame[None, :]                         # [Q, N]
        dmask = cross & flag_tok[None, :]
        xmask = cross & ~flag_tok[None, :]
        neg = torch.finfo(logits.dtype).min
        out = {}
        rng = logits.max(-1).values - logits.median(-1).values
        out["logit_range_p50"] = float(rng.quantile(0.5))
        out["logit_range_p99"] = float(rng.quantile(0.99))
        has_d = dmask.any(-1)
        if has_d.any():
            dmax = logits.masked_fill(~dmask, neg).max(-1).values
            xmax = logits.masked_fill(~xmask, neg).max(-1).values
            out["dgap_p99"] = float((dmax - xmax)[:, has_d].quantile(0.99))
            a = logits.softmax(-1)
            mD = (a * dmask).sum(-1)
            mX = (a * cross).sum(-1)
            ratio = (dmask.sum(-1).float() / cross.sum(-1).float().clamp(min=1))
            rho = (mD / mX.clamp(min=1e-12)) / ratio.clamp(min=1e-12)
            out["rho_cam"] = float(rho[:, :n_cam][:, has_d[:n_cam]].mean()) if has_d[:n_cam].any() else None
            out["rho_patch"] = float(rho[:, n_cam:][:, has_d[n_cam:]].mean()) if has_d[n_cam:].any() else None
            # same-frame rho: share of same-frame mass on same-frame instrument keys / their count
            # share. Rows: camera queries, and TISSUE patch queries only (instrument -> instrument
            # is not the path in question). Cross-frame masking rescales a row's same-frame mass
            # uniformly, so this ratio is the same with or without the training mask.
            same = ~cross
            sd = same & flag_tok[None, :]
            m_sd, m_s = (a * sd).sum(-1), (a * same).sum(-1)
            r_s = sd.sum(-1).float() / same.sum(-1).float().clamp(min=1)
            rho_s = (m_sd / m_s.clamp(min=1e-12)) / r_s.clamp(min=1e-12)
            ok = r_s > 0
            tissue_q = ~flag_tok[q_idx]
            cam_rows, pat_rows = ok[:n_cam], ok[n_cam:] & tissue_q[n_cam:]
            out["rho_same_cam"] = float(rho_s[:, :n_cam][:, cam_rows].mean()) if cam_rows.any() else None
            out["rho_same_tissue"] = float(rho_s[:, n_cam:][:, pat_rows].mean()) if pat_rows.any() else None
            a30 = (logits - 30.0 * dmask).softmax(-1)
            res = (a30 * dmask).sum(-1)[:, has_d].flatten()
            out["resid30_max"] = float(res.max())
            out["resid30_p999"] = float(res.quantile(0.999))
        return out
