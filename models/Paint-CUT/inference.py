import os
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.cutsavggedge import ContrastiveModel
#from models.cutyuan import ContrastiveModel
from utils.dataset import XYDataset
from utils.util import read_yaml_config, test_transforms, reverse_image_normalize
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import yaml
from yaml.loader import SafeLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

test_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

def read_yaml_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def reverse_image_normalize(img, mean=0.5, std=0.5):
    return img * std + mean
class XYDataset(Dataset):
    def __init__(self, root_X, root_Y, transform=None):
        self.root_X = root_X
        self.root_Y = root_Y
        self.transform = transform

        self.X_images = self.listdir(root_X)
        self.Y_images = self.listdir(root_Y)
        self.length_dataset = max(len(self.X_images), len(self.Y_images))
        self.X_len = len(self.X_images)
        self.Y_len = len(self.Y_images)
    
    def listdir(self, path):
        files = []
        for f in os.listdir(path):
            if not f.startswith('.'): # to ignore .gitkeep
                files.append(f)
        return files

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        X_img = self.X_images[index % self.X_len]
        Y_img = self.Y_images[index % self.Y_len]

        X_path = os.path.join(self.root_X, X_img)
        Y_path = os.path.join(self.root_Y, Y_img)

        X_img = np.array(Image.open(X_path).convert("RGB"))
        Y_img = np.array(Image.open(Y_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=X_img, image0=Y_img)
            X_img = augmentations["image"]
            Y_img = augmentations["image0"]

        return X_img, Y_img

class XInferenceDataset(Dataset):
    def __init__(self, root_X, transform=None):
        self.root_X = root_X
        self.transform = transform

        self.X_images = os.listdir(root_X)
        self.length_dataset = len(self.X_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        X_img = self.X_images[index]

        X_path = os.path.join(self.root_X, X_img)

        X_img = np.array(Image.open(X_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=X_img)
            X_img = augmentations["image"]

        return X_img, X_path

def main():
    config = read_yaml_config("./config.yaml")

    model = ContrastiveModel(config)

    val_dataset = XInferenceDataset(root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"], transform=test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    for idx, data in enumerate(val_loader):
        print(f"Processing {idx}", end="\r")

        X, X_path = data
        Y_fake = model.inference(X)

        if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
            save_image(reverse_image_normalize(X), os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", f"{Path(X_path[0]).stem}_X_{idx}.png"))
        
        save_image(reverse_image_normalize(Y_fake), os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", f"{Path(X_path[0]).stem}_Y_fake_{idx}.png"))

if __name__ == "__main__":
    main()
