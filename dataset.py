'''Custom CIFAR10 Dataset for Kaggle competition data.'''
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10Kaggle(Dataset):
    """Dataset for the Kaggle CIFAR-10 competition format.

    Expects:
      - A CSV with columns: id, filename, label
      - An image directory containing the PNG files referenced in the CSV
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = int(row['label'])
        return image, label


class CIFAR10KaggleTest(Dataset):
    """Dataset for the Kaggle CIFAR-10 test set (no labels).

    Expects:
      - A CSV with columns: id, filename
      - An image directory containing the PNG files referenced in the CSV
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, int(row['id'])

