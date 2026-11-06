
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.h5')])
        
        print(f"✅ Loaded {len(self.img_files)} samples from {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = img_name.replace("image_", "mask_")
        
        # Load Image
        with h5py.File(os.path.join(self.img_dir, img_name), 'r') as f:
            image = np.array(f['img'])   # (128, 128, 14)
        
        # Load Mask (only if mask_dir is provided)
        if self.mask_dir is not None:
            with h5py.File(os.path.join(self.mask_dir, mask_name), 'r') as f:
                mask = np.array(f['mask'])
        else:
            mask = np.zeros((128, 128), dtype=np.int64)  # dummy mask (won't be used)
        
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # (14, 128, 128)
        mask = torch.from_numpy(mask).long()
        
        return image, mask
