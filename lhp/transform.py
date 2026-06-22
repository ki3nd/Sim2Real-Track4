import random
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation


class LHPTransform:
    """Stochastically pick a local (random-resized-crop) or global (full resize) view,
    then ToTensor + Inception normalize. Returns (tensor, view_name)."""

    def __init__(self, resolution=384, crop_scale=(0.5, 0.8), local_prob=0.5):
        self.local_prob = local_prob
        self._local = transforms.Compose([
            RandomResizedCropAndInterpolation(resolution, scale=crop_scale, interpolation="bicubic"),
            transforms.RandomHorizontalFlip(),
        ])
        self._global = transforms.Resize((resolution, resolution), interpolation=3)  # BICUBIC
        self._finalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

    def __call__(self, image):
        if random.random() < self.local_prob:
            view, img = "local", self._local(image)
        else:
            view, img = "global", self._global(image)
        return self._finalize(img), view
