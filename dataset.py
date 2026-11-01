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
        print("==> Preloading images into RAM (takes ~5 seconds, uses only ~150MB)...")
        all_paths = [os.path.join(img_dir, fn) for fn in self.df['filename']]
        self.images = [Image.open(path).convert('RGB') for path in all_paths]
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]  
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

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
        print("==> Preloading images into RAM (takes ~5 seconds, uses only ~150MB)...")
        all_paths = [os.path.join(img_dir, fn) for fn in self.df['filename']]
        self.images = [Image.open(path).convert('RGB') for path in all_paths]
        self.ids = self.df['id'].tolist()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]  
        if self.transform:
            image = self.transform(image)
        label = self.ids[idx]


        return image, label

