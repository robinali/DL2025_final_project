import os
import torch
import lpips
from PIL import Image
from torchvision import transforms
import numpy as np

# LPIPS model (use 'alex', 'vgg', or 'squeeze')
loss_fn = lpips.LPIPS(net='alex')

# Paths
real_dir = "./org"
fake_dir = "./fake"

# Get matching filenames
real_files = [f for f in os.listdir(real_dir) if f.endswith(".png")]
fake_files = [f for f in os.listdir(fake_dir) if f.endswith(".png")]

# Ensure only common files are processed
common_files = list(set(real_files) & set(fake_files))
common_files.sort()  # optional: sort to ensure consistent order

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # adjust if needed
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # for 3-channel LPIPS normalization
])

# LPIPS distances
distances = []

for fname in common_files:
    real_path = os.path.join(real_dir, fname)
    fake_path = os.path.join(fake_dir, fname)

    real_img = transform(Image.open(real_path).convert('RGB')).unsqueeze(0)
    fake_img = transform(Image.open(fake_path).convert('RGB')).unsqueeze(0)

    dist = loss_fn(real_img, fake_img)
    distances.append(dist.item())

    print(f"LPIPS between real and fake {fname}: {dist.item():.4f}")

# Average LPIPS
avg_distance = np.mean(distances)
print(f"\nAverage LPIPS distance over {len(distances)} images: {avg_distance:.4f}")

