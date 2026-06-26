import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMDecoder(nn.Module):
    """SimMIM lightweight decoder: linear-predict the raw 32x32x3 pixel block for each of the
    49 final-stage tokens, then fold into a [B, 3, 224, 224] image."""

    def __init__(self, in_dim=1024, mask_patch_size=32):
        super().__init__()
        self.p = mask_patch_size
        self.grid = 224 // mask_patch_size                 # 7
        self.head = nn.Linear(in_dim, mask_patch_size * mask_patch_size * 3)

    def forward(self, spatial):                            # [B, 49, in_dim]
        b = spatial.size(0)
        x = self.head(spatial)                             # [B, 49, p*p*3]
        x = x.view(b, self.grid, self.grid, 3, self.p, self.p)        # [B,7,7,3,32,32]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()                 # [B,3,7,32,7,32]
        return x.view(b, 3, self.grid * self.p, self.grid * self.p)  # [B,3,224,224]


def mim_loss(recon, image, cell_mask):
    """L1 over masked 32x32 cells only. cell_mask: bool [B, 7, 7]."""
    p = recon.size(-1) // cell_mask.size(-1)               # 32
    pixel_mask = cell_mask.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)  # [B,224,224]
    pixel_mask = pixel_mask.unsqueeze(1)                   # [B,1,224,224]
    diff = (recon - image).abs() * pixel_mask
    denom = pixel_mask.sum() * recon.size(1) + 1e-6        # masked pixels * channels
    return diff.sum() / denom
