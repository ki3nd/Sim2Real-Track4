import torch
import torch.nn.functional as F
from models.uit import contrastive_loss


def test_contrastive_loss_single_process_equals_symmetric_ce():
    torch.manual_seed(0)
    i = F.normalize(torch.randn(4, 16), dim=-1)
    t = F.normalize(torch.randn(4, 16), dim=-1)
    temp = 0.07
    got = contrastive_loss(i, t, temp)
    logits = i @ t.t() / temp
    labels = torch.arange(4)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    assert torch.allclose(got, expected, atol=1e-6)
    got.backward if False else None  # smoke: it is differentiable
    contrastive_loss(i.requires_grad_(), t.requires_grad_(), temp).backward()
