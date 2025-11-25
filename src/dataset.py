import os
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")


class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for loading chest X-ray images.
    It automatically reads images from /data/{split}/{class_name}/.
    
    Args:
        split: 'train', 'val', or 'test'
        transform: Optional transformations to apply
        use_augmentation: Whether to use data augmentation (for training)
    """
    def __init__(self, split="train", transform=None, use_augmentation=False):
        self.image_paths = []
        self.labels = []
        self.split = split
        self.transform = transform
        self.use_augmentation = use_augmentation
        
       
        self.mean = 0.5  # Grayscale mean
        self.std = 0.5   # Grayscale std

    
        if not DATA_DIR.exists():
            raise RuntimeError(f"Data directory not found: {DATA_DIR}")

        for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            folder = DATA_DIR / split / class_name
            if not folder.exists():
                raise RuntimeError(f"Class folder not found: {folder}")
            
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = folder / file
                    self.image_paths.append(str(path))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"  - NORMAL: {sum(1 for l in self.labels if l == 0)}")
        print(f"  - PNEUMONIA: {sum(1 for l in self.labels if l == 1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Basic augmentation for training (simple approach without albumentations)
        if self.use_augmentation and self.split == 'train':
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random brightness adjustment
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)
        
        # Convert to tensor and normalize
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        
        # Normalize with mean and std
        img = (img - self.mean) / self.std
        
        return img, torch.tensor(label, dtype=torch.long)
