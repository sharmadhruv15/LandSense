
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import LandslideDataset

def get_dataloaders(batch_size=16, num_workers=2, val_split=0.2):
    # Full training dataset with masks
    full_train_dataset = LandslideDataset(
        img_dir='/content/landslide4sense/TrainData/img/',
        mask_dir='/content/landslide4sense/TrainData/mask/'
    )
    
    # Split into train and validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # For validation, we still need masks → so we create a new dataset for val part
    # (random_split keeps indices, but we need to handle it carefully)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    print(f"Train samples: {train_size} | Validation samples: {val_size}")
    
    return train_loader, val_loader
