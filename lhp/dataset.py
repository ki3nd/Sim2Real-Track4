import os
import random
from PIL import Image
from torch.utils.data import Dataset

from dataset.utils import pre_caption, read_json_to_list  # reused from CMP (I/O only)


class LHPDataset(Dataset):
    def __init__(self, ann_files, image_root, transform, max_words=56, eda=False, eda_p=0.5):
        self.image_root = image_root
        self.transform = transform
        self.max_words = max_words
        self.eda = eda
        self.eda_p = eda_p
        self.ann = []
        for f in ann_files:
            self.ann.extend(read_json_to_list(f))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        for _ in range(len(self.ann)):
            ann = self.ann[index]
            image_path = os.path.join(self.image_root, ann["image"])
            image_path = image_path.replace(".jpg", ".webp")
            try:
                image = Image.open(image_path).convert("RGB")
                image, _view, patch_mask = self.transform(image)
                caption = pre_caption(ann["caption"], self.max_words, self.eda, self.eda_p)
                return image, caption, ann["image"], patch_mask
            except Exception:
                index = random.randint(0, len(self.ann) - 1)
        raise RuntimeError("LHPDataset: no loadable images found in dataset")
