import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationHead(nn.Module):
    """Decode the per-frame illumination token into a coarse LOG illumination map.

    Output is a grid x grid map of log values, not a [0,1] image: the loss that supervises it is
    scale-invariant in log space (loss.compute_illu_loss), so any sigmoid would only add a
    saturation region with nothing to gain. The grid is deliberately coarse -- one token cannot
    carry texture, and near-field shading is smooth -- and the loss is computed AT this
    resolution, so the head is never asked for high-frequency content it cannot express.
    """

    def __init__(self, dim_in: int = 2048, hidden: int = 512, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, grid * grid),
        )

    def forward(self, illu_token: torch.Tensor) -> torch.Tensor:
        """illu_token [B, S, dim_in] -> log illumination [B, S, grid, grid]."""
        B, S, _ = illu_token.shape
        return self.mlp(illu_token).view(B, S, self.grid, self.grid)

    @staticmethod
    def upsample(log_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """[B, S, g, g] -> [B, S, H, W], bilinear. Visualisation only; the loss never sees this."""
        B, S, g, _ = log_map.shape
        up = F.interpolate(log_map.reshape(B * S, 1, g, g), size=(H, W), mode="bilinear", align_corners=False)
        return up.view(B, S, H, W)
