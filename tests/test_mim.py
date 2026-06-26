import torch
from models.uit import MIMDecoder, mim_loss


def test_decoder_maps_tokens_to_image():
    dec = MIMDecoder(in_dim=1024, mask_patch_size=32)
    spatial = torch.randn(2, 49, 1024)
    recon = dec(spatial)
    assert recon.shape == (2, 3, 224, 224)


def test_mim_loss_scalar_backward_and_masked_only():
    dec = MIMDecoder(in_dim=1024, mask_patch_size=32)
    spatial = torch.randn(2, 49, 1024, requires_grad=True)
    image = torch.randn(2, 3, 224, 224)
    cell_mask = torch.zeros(2, 7, 7, dtype=torch.bool)
    cell_mask[:, 0, 0] = True                          # mask exactly one 32x32 cell
    loss = mim_loss(dec(spatial), image, cell_mask)
    assert loss.ndim == 0 and loss.item() >= 0
    loss.backward()
    # all-unmasked -> zero loss contribution
    zero = mim_loss(dec(spatial.detach()), image, torch.zeros(2, 7, 7, dtype=torch.bool))
    assert zero.item() == 0.0
