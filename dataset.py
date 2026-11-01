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
        # Fast Python lists for filenames and labels (no pandas overhead in __getitem__)
        self.filenames = self.df['filename'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')

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
        self.filenames = self.df['filename'].tolist()
        self.ids = self.df['id'].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.ids[idx]