import random
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation


class LHPTransform:
    """Stochastically pick local (random-resized-crop) / masked (full resize + patch
    attention-mask) / global (full resize) view, then ToTensor + Inception normalize.
    Returns (tensor, view_name, patch_mask) where patch_mask is a bool [num_patches]
    (all-False unless view == 'masked')."""

    def __init__(self, resolution=384, crop_scale=(0.5, 0.8), local_prob=0.5,
                 masked_prob=0.0, mask_ratio_range=(0.4, 0.6)):
        self.local_prob = local_prob
        self.masked_prob = masked_prob
        self.mask_ratio_range = mask_ratio_range
        self.num_patches = (resolution // 16) ** 2
        self._local = transforms.Compose([
            RandomResizedCropAndInterpolation(resolution, scale=crop_scale, interpolation="bicubic"),
            transforms.RandomHorizontalFlip(),
        ])
        self._global = transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC)
        self._finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

    def _make_patch_mask(self, masked):
        patch_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        if masked:
            ratio = random.uniform(*self.mask_ratio_range)
            n_mask = int(round(self.num_patches * ratio))
            idx = torch.randperm(self.num_patches)[:n_mask]
            patch_mask[idx] = True
        return patch_mask

    def __call__(self, image):
        r = random.random()
        if r < self.local_prob:
            view, img = "local", self._local(image)
        elif r < self.local_prob + self.masked_prob:
            view, img = "masked", self._global(image)   # full image; patches blocked at model
        else:
            view, img = "global", self._global(image)
        return self._finalize(img), view, self._make_patch_mask(view == "masked")
