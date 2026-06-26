import torch
import torch.nn as nn

from models.swin_transformer import SwinTransformer


def generate_mim_mask(batch, mask_ratio, device):
    """SimMIM mask: choose mask_ratio of the 7x7=49 cells (each = 32x32 px = 8x8 stage-0 patches).
    Returns (patch_mask [B, 56*56] bool over stage-0 patches, cell_mask [B, 7, 7] bool)."""
    n_cells = 49
    n_mask = round(mask_ratio * n_cells)
    cell_mask = torch.zeros(batch, n_cells, dtype=torch.bool, device=device)
    for b in range(batch):
        idx = torch.randperm(n_cells, device=device)[:n_mask]
        cell_mask[b, idx] = True
    cell_mask = cell_mask.view(batch, 7, 7)
    # expand each 32-cell to 8x8 stage-0 patches: 7x7 -> 56x56
    patch_grid = cell_mask.repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)  # [B,56,56]
    patch_mask = patch_grid.reshape(batch, 56 * 56)
    return patch_mask, cell_mask


class MaskedSwin(SwinTransformer):
    """SwinTransformer + an optional SimMIM mask: masked stage-0 patch tokens are replaced
    by a learnable mask_token right after patch_embed. mask=None ⇒ identical to parent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        x = self.patch_embed(x)                         # [B, 3136, 128]
        if mask is not None:
            w = mask.unsqueeze(-1).type_as(self.mask_token)   # [B, 3136, 1]
            x = x * (1 - w) + self.mask_token * w
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                                # [B, 49, 1024]
        x_cls = self.avgpool(x.transpose(1, 2))         # [B, 1024, 1]
        x = torch.cat([x_cls.transpose(1, 2), x], dim=1)
        return x
