import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset

class HandballDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        unique_classes = self.annotations['class'].unique()
        self.classes = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}

    def __len__(self):
        return len(self.annotations['filename'].unique())

    def __getitem__(self, idx):
        img_name = self.annotations['filename'].unique()[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # Get the labels and bounding boxes for the current image
        boxes, labels = [], []
        img_annotations = self.annotations[self.annotations['filename'] == img_name]
        for _, row in img_annotations.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes[row['class']])  # Map class to integer ID

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img = self.transform(img)

        return img, target
