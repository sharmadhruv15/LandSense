
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import sys

sys.path.append('/content/Himalayan_Landslide_Detection/src')

# Direct import
from unet import UNet

def visualize_predictions(num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(n_channels=14, n_classes=2).to(device)
    model.load_state_dict(torch.load('/content/Himalayan_Landslide_Detection/outputs/checkpoints/best_unet.pth', 
                                   map_location=device, weights_only=True))
    model.eval()
    
    img_dir = '/content/landslide4sense/TrainData/img/'
    mask_dir = '/content/landslide4sense/TrainData/mask/'
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.h5')])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    fig.suptitle('Landslide Detection Results (Best Dice Score: 0.6477)', fontsize=18)
    
    for i in range(num_samples):
        idx = i * 250 + 100
        img_name = img_files[idx]
        mask_name = img_name.replace("image_", "mask_")
        
        with h5py.File(os.path.join(img_dir, img_name), 'r') as f:
            image = np.array(f['img'])
        with h5py.File(os.path.join(mask_dir, mask_name), 'r') as f:
            mask = np.array(f['mask'])
        
        # Prediction
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # RGB
        rgb = np.stack([image[:,:,3], image[:,:,2], image[:,:,1]], axis=-1)
        rgb = np.clip(rgb / rgb.max(), 0, 1)
        
        axes[i,0].imshow(rgb)
        axes[i,0].set_title('RGB Image')
        axes[i,0].axis('off')
        
        axes[i,1].imshow(mask, cmap='gray')
        axes[i,1].set_title('Ground Truth')
        axes[i,1].axis('off')
        
        axes[i,2].imshow(pred_mask, cmap='gray')
        axes[i,2].set_title('Prediction')
        axes[i,2].axis('off')
        
        overlay = rgb.copy()
        overlay[pred_mask == 1] = [1, 0, 0]
        axes[i,3].imshow(overlay)
        axes[i,3].set_title('Prediction Overlay')
        axes[i,3].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/content/Himalayan_Landslide_Detection/outputs/visualizations/final_predictions.png', dpi=300)
    plt.show()
    
    print("✅ Visualization completed and saved!")

print("✅ Visualization script ready!")
