import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, subset_indices=None, limit=None):
        df = pd.read_csv(csv_path)

        if subset_indices is not None:
            df = df.iloc[subset_indices].reset_index(drop=True)

        if limit:
            df = df.iloc[:limit]

        self.images = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        self.labels = df.iloc[:, 0].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.transpose(img, (1, 0))

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        label = torch.tensor(self.labels[idx])
        return img, label


def transforms(is_training=False):
    if is_training:
        return A.Compose([
            A.Affine(translate_percent={"x": 0.06, "y": 0.06}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.Normalize(mean=(0.1307,), std=(0.3081,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.1307,), std=(0.3081,)),
            ToTensorV2(),
        ])
