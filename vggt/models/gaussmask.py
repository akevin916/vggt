"""Gaussian spatio-temporal key mask on global attention (contribution 3-10, ball / cone).

Query (s, p) and key (s', p') with displacement (ds, dy, dx) = key - query (frames, patch rows,
patch cols). Per global layer l >= start:

    sigma_y(ds) = max(sigma_y0 + v |ds|, 0.5)   (x alike; ball: v = 0)
    Q = ds^2 / sigma_s^2 + dy^2 / sigma_y(ds)^2 + dx^2 / sigma_x(ds)^2

Kept keys (Q <= c^2) get bias -Q/2, the rest are dropped. Always kept with bias 0: same frame,
key frame 0, special tokens (key or query). No mean token.

Budget: c^2 is solved per layer so that rho of the (query, key) pairs with ds != 0 and key frame
>= 1 in an S_max-frame contiguous chunk on the grid are kept. Pairs per displacement are
n = (S_max - |ds| - [ds < 0]) * (GH - |dy|) * (GW - |dx|); the [ds < 0] term removes the key-frame-0
pairs. So sigma only sets the SHAPE (time vs space) and the flatness inside the kept region.

Q depends on the displacement alone, so each layer's bias is one table [2*S_max-1, 2*GH-1,
2*GW-1] (~0.55 M entries) built here in plain PyTorch and looked up in flex's score_mod. That is
also what makes training affordable: gradients reach the table spread over 0.55 M addresses and
autograd takes them back to sigma. Computing Q inside the score_mod from captured scalars works
but every score's gradient then lands on the same scalar (82 s per layer at S=48 vs 4.3 s).
"""
import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DROP = -1.0e4   # finite stand-in for log 0 (same as Aggregator.CM_DROP)


class GaussMask(nn.Module):
    def __init__(self, n_layers: int = 16, start: int = 8, grid=(30, 37), S_max: int = 64,
                 cone: bool = False, rho: float = 0.25, T_start: float = 2.0, T_end: float = 0.1,
                 T_steps=(0, 1100), hard_from: int = 1100, init: Optional[dict] = None,
                 fixed_path: Optional[str] = None):
        super().__init__()
        self.start = int(start)
        self.L = int(n_layers)
        self.GH, self.GW = int(grid[0]), int(grid[1])
        self.S_max = int(S_max)
        self.MS = self.S_max - 1
        self.NS, self.NY, self.NX = 2 * self.MS + 1, 2 * self.GH - 1, 2 * self.GW - 1
        self.rho = float(rho)
        self.T_start, self.T_end = float(T_start), float(T_end)
        self.T_steps = (int(T_steps[0]), int(T_steps[1]))
        self.hard_from = int(hard_from)
        self.eval_mode = "hard"            # "hard" | "off"

        ds = torch.arange(-self.MS, self.MS + 1).float()
        dy = torch.arange(-(self.GH - 1), self.GH).float()
        dx = torch.arange(-(self.GW - 1), self.GW).float()
        n_s = (self.S_max - ds.abs() - (ds < 0).float()).clamp(min=0)
        n_s[self.MS] = 0.0                                   # ds = 0: always kept, not budgeted
        n = n_s[:, None, None] * (self.GH - dy.abs())[None, :, None] * (self.GW - dx.abs())[None, None, :]
        for name, val in (("ds", ds), ("dy", dy), ("dx", dx), ("n", n)):
            self.register_buffer(name, val, persistent=False)

        self.fixed = fixed_path is not None
        if self.fixed:
            with open(fixed_path) as f:
                spec = json.load(f)
            layers = spec["layers"]
            assert len(layers) == self.L, (len(layers), self.L)
            assert spec.get("S_max", self.S_max) == self.S_max and tuple(spec.get("grid", grid)) == tuple(grid)
            self.rho = float(spec.get("rho", self.rho))
            cone = bool(spec.get("cone", cone))
            get = lambda k: torch.tensor([float(l[k]) for l in layers])   # noqa: E731
            self.register_buffer("log_s", get("sigma_s").log(), persistent=False)
            self.register_buffer("log_y", get("sigma_y0").log(), persistent=False)
            self.register_buffer("log_x", get("sigma_x0").log(), persistent=False)
            self.register_buffer("v", get("v"), persistent=False)
            self.register_buffer("c2_fixed", get("c2"), persistent=False)
        else:
            init = dict(init or {})
            full = lambda x: torch.full((self.L,), float(x))              # noqa: E731
            self.log_s = nn.Parameter(full(math.log(init.get("sigma_s", 8.0))))
            self.log_y = nn.Parameter(full(math.log(init.get("sigma_y", 6.0))))
            self.log_x = nn.Parameter(full(math.log(init.get("sigma_x", 6.0))))
            if cone:
                self.v = nn.Parameter(full(init.get("v", 0.0)))
            else:
                self.register_buffer("v", full(0.0), persistent=True)
            self.register_buffer("steps", torch.zeros((), dtype=torch.long), persistent=True)
        self.cone = bool(cone)
        self._last_keep = None             # hard keep set at the previous stats() call

    # ---- schedule -----------------------------------------------------------------------------
    def T(self) -> float:
        if self.fixed:
            return self.T_end
        n, (a, b) = int(self.steps), self.T_steps
        if n <= a or b <= a:
            return self.T_start if n <= a else self.T_end
        return self.T_start + (self.T_end - self.T_start) * min(1.0, (n - a) / (b - a))

    def hard_on(self) -> bool:
        return self.fixed or (self.hard_from > 0 and int(self.steps) >= self.hard_from)

    @torch.no_grad()
    def post_step_(self):
        if not self.fixed:
            self.steps += 1

    # ---- the table ----------------------------------------------------------------------------
    def Q(self) -> torch.Tensor:
        """[L, NS, NY, NX] float32."""
        ad = self.ds.abs()[None, :]                                           # [1, NS]
        sy = (self.log_y.float().exp()[:, None] + self.v.float()[:, None] * ad).clamp(min=0.5)
        sx = (self.log_x.float().exp()[:, None] + self.v.float()[:, None] * ad).clamp(min=0.5)
        inv_s2 = torch.exp(-2.0 * self.log_s.float())[:, None, None, None]
        return (self.ds[None, :, None, None] ** 2 * inv_s2
                + self.dy[None, None, :, None] ** 2 / sy[:, :, None, None] ** 2
                + self.dx[None, None, None, :] ** 2 / sx[:, :, None, None] ** 2)

    @torch.no_grad()
    def solve_c2_hard(self, Q: torch.Tensor) -> torch.Tensor:
        """[L]: c^2 whose hard keep share (pair-weighted) is closest to rho. Ties in Q are kept
        or dropped together, so the share can miss rho by one tie group."""
        q = Q.detach().flatten(1)
        w = self.n.flatten()
        order = q.argsort(dim=1)
        qs = q.gather(1, order)
        cum = w[order].cumsum(1) / w.sum()
        # last index of each run of equal Q (a threshold keeps the whole run)
        last = torch.ones_like(qs, dtype=torch.bool)
        last[:, :-1] = qs[:, 1:] != qs[:, :-1]
        err = (cum - self.rho).abs().masked_fill(~last, float("inf"))
        idx = err.argmin(1, keepdim=True)
        return qs.gather(1, idx).squeeze(1)

    @torch.no_grad()
    def solve_c2_soft(self, Q: torch.Tensor, T: float) -> torch.Tensor:
        """[L]: c^2 with sum n * sigmoid((c^2 - Q) / T) = rho * sum n (bisection, no grad)."""
        q = Q.detach().flatten(1)
        w = self.n.flatten()[None]
        tot = w.sum()
        lo = q.min(1, keepdim=True).values - 30 * T
        hi = q.max(1, keepdim=True).values + 30 * T
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            m = (torch.sigmoid((mid - q) / T) * w).sum(1, keepdim=True) / tot
            lo = torch.where(m < self.rho, mid, lo)
            hi = torch.where(m < self.rho, hi, mid)
        return (0.5 * (lo + hi)).squeeze(1)

    def c2_hard(self, Q: torch.Tensor) -> torch.Tensor:
        return self.c2_fixed.to(Q.device) if self.fixed else self.solve_c2_hard(Q)

    def tables(self, training: bool) -> torch.Tensor:
        """[L, NS*NY*NX] additive bias per displacement (entries at ds = 0 are unused)."""
        Q = self.Q()
        L = Q.shape[0]
        if self.fixed or not training:
            c2 = self.c2_hard(Q).view(L, 1, 1, 1)
            return torch.where(Q <= c2, -Q / 2, DROP).flatten(1)
        T = self.T()
        c2s = self.solve_c2_soft(Q, T).view(L, 1, 1, 1)
        soft = (-Q / 2 + F.logsigmoid((c2s - Q) / T)).clamp(min=DROP)
        if not self.hard_on():
            return soft.flatten(1)
        c2h = self.solve_c2_hard(Q).view(L, 1, 1, 1)
        hard = torch.where(Q <= c2h, -Q / 2, DROP)
        return (hard + soft - soft.detach()).flatten(1)        # straight-through

    def index(self, fq, fk, yq, xq, yk, xk):
        """Flat table index of a displacement (works on ints or tensors)."""
        return ((fk - fq + self.MS) * self.NY + (yk - yq + self.GH - 1)) * self.NX + (xk - xq + self.GW - 1)

    # ---- readout ------------------------------------------------------------------------------
    @torch.no_grad()
    def layer_summary(self) -> list:
        Q = self.Q()
        c2 = self.c2_hard(Q)
        keep = Q <= c2.view(-1, 1, 1, 1)
        w = self.n
        out = []
        for l in range(Q.shape[0]):
            k = keep[l].float() * w
            tot = k.sum()
            frac = (tot / w.sum()).item()
            # extent of the kept set: pair-weighted RMS |ds|, |dy|, |dx| (shape: time vs space)
            rms = lambda g: (((g ** 2) * k).sum() / tot).sqrt().item()     # noqa: E731
            out.append(dict(
                layer=self.start + l,
                sigma_s=self.log_s[l].exp().item(), sigma_y0=self.log_y[l].exp().item(),
                sigma_x0=self.log_x[l].exp().item(), v=self.v[l].item(), c2=c2[l].item(),
                hard_keep=round(frac, 5),
                rms_ds=rms(self.ds[:, None, None]), rms_dy=rms(self.dy[None, :, None]),
                rms_dx=rms(self.dx[None, None, :])))
        return out

    @torch.no_grad()
    def stats(self) -> dict:
        Q = self.Q()
        c2 = self.c2_hard(Q)
        keep = Q <= c2.view(-1, 1, 1, 1)
        w = self.n
        kept = (keep.float() * w).flatten(1).sum(1)
        d = {"step": int(self.steps) if not self.fixed else None, "T": round(self.T(), 4),
             "hard": self.hard_on(),
             "hard_keep": [round(x, 4) for x in (kept / w.sum()).tolist()],
             "sigma_s": [round(x, 3) for x in self.log_s.exp().tolist()],
             "sigma_y0": [round(x, 3) for x in self.log_y.exp().tolist()],
             "sigma_x0": [round(x, 3) for x in self.log_x.exp().tolist()],
             "v": [round(x, 4) for x in self.v.tolist()],
             "c2": [round(x, 3) for x in c2.tolist()]}
        if not self.fixed:
            T = self.T()
            c2s = self.solve_c2_soft(Q, T)
            g = torch.sigmoid((c2s.view(-1, 1, 1, 1) - Q) / T)
            d["soft_keep"] = [round(x, 4) for x in ((g * w).flatten(1).sum(1) / w.sum()).tolist()]
        if self._last_keep is not None:
            # pair-weighted share of the kept set that changed since the previous call, per layer
            diff = ((keep ^ self._last_keep).float() * w).flatten(1).sum(1) / kept.clamp(min=1)
            d["set_change"] = [round(x, 4) for x in diff.tolist()]
        self._last_keep = keep
        return d

    @torch.no_grad()
    def export(self, path: str, note: str = ""):
        spec = dict(cone=self.cone, rho=self.rho, S_max=self.S_max, grid=[self.GH, self.GW],
                    start=self.start, note=note, layers=self.layer_summary())
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)
        return spec
