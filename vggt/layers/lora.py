"""LoRA for the pretrained linears of the aggregator (enable via VGGT(lora_rank>0)).

LoRALinear SUBCLASSES nn.Linear instead of wrapping it, so the pretrained keys keep their names
(`...attn.qkv.weight`, not `...attn.qkv.base.weight`). A wrapper would rename every target key,
and under the trainer's strict=False load VGGT-1B's weights for those layers would be silently
dropped as "missing" -- the run would train LoRA on top of random attention.

lora_B is zero-init, so at step 0 the output equals the pretrained layer exactly.

Eval builds the model from checkpoint keys and knows nothing about LoRA, so a LoRA checkpoint is
MERGED back into plain weights before loading (merge_lora_state_dict, W += scale * B @ A); the
scale is a persistent buffer so the checkpoint alone reproduces it.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from wcmatch import fnmatch


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, rank=16, alpha=32.0):
        super().__init__(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer("lora_scale", torch.tensor(float(alpha) / rank))

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float) -> "LoRALinear":
        new = cls(linear.in_features, linear.out_features, linear.bias is not None, rank, alpha)
        new.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new.bias.data.copy_(linear.bias.data)
        return new

    def forward(self, x):
        return super().forward(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scale


def apply_lora(model: nn.Module, targets, rank: int, alpha: float) -> int:
    """Replace every nn.Linear whose qualified name matches a glob in `targets`. Returns count."""
    flags = fnmatch.CASE | fnmatch.DOTMATCH | fnmatch.EXTMATCH
    hits = [(n, m) for n, m in model.named_modules()
            if type(m) is nn.Linear and any(fnmatch.fnmatch(n, p, flags=flags) for p in targets)]
    if not hits:
        raise ValueError(f"lora_targets matched no nn.Linear: {list(targets)}")
    for name, lin in hits:
        parent_name, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent_name), attr, LoRALinear.from_linear(lin, rank, alpha))
    return len(hits)


def merge_lora_state_dict(sd: dict) -> dict:
    """Fold LoRA deltas into the base weights and drop the LoRA keys (no-op without LoRA)."""
    prefixes = [k[: -len(".lora_A")] for k in sd if k.endswith(".lora_A")]
    if not prefixes:
        return sd
    sd = dict(sd)
    for p in prefixes:
        A, B, s = sd.pop(f"{p}.lora_A"), sd.pop(f"{p}.lora_B"), sd.pop(f"{p}.lora_scale")
        W = sd[f"{p}.weight"]
        sd[f"{p}.weight"] = (W.float() + s.float() * (B.float() @ A.float())).to(W.dtype)
    return sd
