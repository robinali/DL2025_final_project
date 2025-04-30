import os
import shutil
from cleanfid import fid

# Original dir containing all images
img_dir = "./test"
real_dir = "temp_real"
fake_dir = "temp_fake"

# Create temp folders
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Separate the files
for fname in os.listdir(img_dir):
    if fname.endswith(".png"):
        if "_X_" in fname:
            shutil.copy(os.path.join(img_dir, fname), os.path.join(real_dir, fname))
        elif "_Y_fake_" in fname:
            shutil.copy(os.path.join(img_dir, fname), os.path.join(fake_dir, fname))

# Compute FID
fid_score = fid.compute_fid(real_dir, fake_dir)
print("FID score:", fid_score)

# Optional: Clean up temp folders
shutil.rmtree(real_dir)
shutil.rmtree(fake_dir)

