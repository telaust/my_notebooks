import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra: str = root_zebra
        self.root_horse: str = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(self.root_zebra)
        self.horse_images = os.listdir(self.root_horse)

        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

        self.len_dataset = max(self.zebra_len, self.horse_len)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        zebra_img_path = self.zebra_images[index % self.zebra_len]
        horse_img_path = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img_path)
        horse_path = os.path.join(self.root_horse, horse_img_path)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augs = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augs["image"]
            horse_img = augs["image0"]

        return zebra_img, horse_img




