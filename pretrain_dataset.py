import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class RotationDataset(Dataset):
    """Self-supervised dataset for Rotation Prediction (RotNet).
    
    Loads images from train_images and test_images (52,000 images total).
    For each image, it randomly applies one rotation:
      0:   0 degrees (original upright)
      1:  90 degrees
      2: 180 degrees
      3: 270 degrees
    Returns: (rotated_image, rotation_label)
    """

    def __init__(self, data_dir, include_test=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []

        # 1. Collect all train images (50,000) via train_labels.csv (instant load)
        train_csv = os.path.join(data_dir, 'train_labels.csv')
        train_img_dir = os.path.join(data_dir, 'train_images')
        if os.path.isfile(train_csv):
            train_df = pd.read_csv(train_csv)
            self.image_paths.extend([
                os.path.join(train_img_dir, fn) for fn in train_df['filename']
            ])
        elif os.path.isdir(train_img_dir):
            import glob
            self.image_paths.extend(sorted(glob.glob(os.path.join(train_img_dir, '*.png'))))

        # 2. Collect all test images (2,000) via test.csv
        if include_test:
            test_csv = os.path.join(data_dir, 'test.csv')
            test_img_dir = os.path.join(data_dir, 'test_images')
            if os.path.isfile(test_csv):
                test_df = pd.read_csv(test_csv)
                self.image_paths.extend([
                    os.path.join(test_img_dir, fn) for fn in test_df['filename']
                ])
            elif os.path.isdir(test_img_dir):
                import glob
                self.image_paths.extend(sorted(glob.glob(os.path.join(test_img_dir, '*.png'))))

        print(f"==> RotationDataset initialized with {len(self.image_paths):,} total unlabeled images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # 1. Pick a random rotation: 0, 1, 2, or 3
        rotation_label = random.randint(0, 3)
        angle = rotation_label * 90  # 0, 90, 180, or 270 degrees

        # 2. Rotate the image
        if angle > 0:
            image = image.rotate(angle, expand=True)

        # 3. Apply normalization / tensor transform
        if self.transform is not None:
            image = self.transform(image)

        # 4. Return standard 1 image and 1 label
        return image, rotation_label



