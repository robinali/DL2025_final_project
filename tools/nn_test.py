import os
import torch
import lpips
from PIL import Image
from torchvision import transforms
import numpy as np

# LPIPS model (use 'alex', 'vgg', or 'squeeze')
loss_fn = lpips.LPIPS(net='alex')

# Paths
img_dir = "../models/Paint-CUT/experiments/jli3158/test"
real_images = [f for f in os.listdir(img_dir) if "_X_" in f]
gen_images = [f for f in os.listdir(img_dir) if "_Y_fake_" in f]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # adjust to your image size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # for LPIPS normalization
])

# Load all real images into memory
real_imgs = []
real_names = []
for fname in real_images:
    img = transform(Image.open(os.path.join(img_dir, fname)).convert('RGB')).unsqueeze(0)
    real_imgs.append(img)
    real_names.append(fname)

real_imgs = torch.cat(real_imgs, dim=0)

# Loop over generated images
for gen_name in gen_images:
    gen_img = transform(Image.open(os.path.join(img_dir, gen_name)).convert('RGB')).unsqueeze(0)

    # Compute LPIPS distance to all real images
    distances = []
    for real_img in real_imgs:
        dist = loss_fn(gen_img, real_img.unsqueeze(0))
        distances.append(dist.item())

    # Find best match
    min_idx = int(np.argmin(distances))
    print(f"Nearest neighbor to {gen_name} is {real_names[min_idx]} with distance {distances[min_idx]:.4f}")

