import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .dino_struct import DinoStructureLoss
from .training_utils import build_transform

# Try to import tqdm, but make it optional
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Define a simple replacement for tqdm that does nothing
    def tqdm(iterable, *args, **kwargs):
        return iterable

class DinoScoreCalculator:
    """
    用于计算两个图像文件夹之间的DINO-Struct分数的类
    """
    
    def __init__(self, device=None, val_img_prep="resize_256"):
        """
        初始化DINO分数计算器
        
        Args:
            device: 使用的设备 ("cuda" 或 "cpu")，如果为None则自动检测
            val_img_prep: 验证图像预处理方式
        """
        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"DinoScoreCalculator using device: {self.device}")
        
        try:
            self.net_dino = DinoStructureLoss(device=self.device)
            print("DinoStructureLoss initialized successfully")
        except Exception as e:
            print(f"Error initializing DinoStructureLoss: {str(e)}")
            raise
        
        # 设置图像预处理，直接使用training_utils中的build_transform函数
        self.T_val = build_transform(val_img_prep)
        print(f"Using image preprocessing: {val_img_prep}")
            
        self.image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    def get_image_files(self, folder_path):
        """
        获取文件夹中的所有图像文件
        
        Args:
            folder_path: 图像文件夹路径
            
        Returns:
            排序后的图像文件路径列表
        """
        files = []
        for ext in self.image_extensions:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        return sorted([f for f in files if not os.path.basename(f).startswith('.')])
    
    def calculate_dino_score(self, img_a_path, img_b_path):
        """
        计算两个图像之间的DINO-Struct分数
        
        Args:
            img_a_path: 第一个图像的路径
            img_b_path: 第二个图像的路径
            
        Returns:
            DINO-Struct分数或None（如果出错）
        """
        try:
            with torch.no_grad():
                # 打开并预处理第一个图像
                try:
                    # 使用更安全的方式加载图像
                    with open(img_a_path, 'rb') as f:
                        img_a_pil = Image.open(f)
                        img_a_pil = img_a_pil.convert("RGB")
                    # 使用与train_cyclegan_turbo.py相同的预处理方式
                    input_img_a = self.T_val(img_a_pil)
                    a = self.net_dino.preprocess(input_img_a).unsqueeze(0).to(self.device)
                except Exception as e:
                    print(f"Error processing image {img_a_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None
                
                # 打开并预处理第二个图像
                try:
                    # 使用更安全的方式加载图像
                    with open(img_b_path, 'rb') as f:
                        img_b_pil = Image.open(f)
                        img_b_pil = img_b_pil.convert("RGB")
                    # 使用与train_cyclegan_turbo.py相同的预处理方式
                    input_img_b = self.T_val(img_b_pil)
                    b = self.net_dino.preprocess(input_img_b).unsqueeze(0).to(self.device)
                except Exception as e:
                    print(f"Error processing image {img_b_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None
                
                # 计算DINO分数
                dino_ssim = self.net_dino.calculate_global_ssim_loss(a, b).item()
                return dino_ssim
        except Exception as e:
            print(f"Error calculating DINO score between {img_a_path} and {img_b_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_folder_scores(self, folder_a, folder_b, use_tqdm=True):
        """
        计算两个文件夹之间的DINO-Struct分数
        
        Args:
            folder_a: 第一个文件夹路径
            folder_b: 第二个文件夹路径
            use_tqdm: 是否使用tqdm进度条
            
        Returns:
            dino_score_a2b: 从A到B的平均DINO-Struct分数
            dino_score_b2a: 从B到A的平均DINO-Struct分数
            scores_a2b: 从A到B的所有分数列表
            scores_b2a: 从B到A的所有分数列表
        """
        # 获取图像文件
        images_a = self.get_image_files(folder_a)
        images_b = self.get_image_files(folder_b)
        
        print(f"Found {len(images_a)} images in folder A: {os.path.basename(folder_a)}")
        print(f"Found {len(images_b)} images in folder B: {os.path.basename(folder_b)}")
        
        # 计算A→B的DINO分数
        print(f"\nCalculating DINO scores from {os.path.basename(folder_a)} to {os.path.basename(folder_b)}...")
        scores_a2b = []
        valid_scores_a2b = []
        
        image_iter = tqdm(images_a) if use_tqdm and TQDM_AVAILABLE else images_a
        for idx, img_path_a in enumerate(image_iter):
            if idx >= len(images_b):
                break
                
            img_path_b = images_b[idx]
            dino_ssim = self.calculate_dino_score(img_path_a, img_path_b)
            scores_a2b.append(dino_ssim)
            if dino_ssim is not None:
                valid_scores_a2b.append(dino_ssim)
        
        if valid_scores_a2b:
            dino_score_a2b = np.mean(valid_scores_a2b)
            print(f"Successfully processed {len(valid_scores_a2b)}/{len(scores_a2b)} image pairs for A→B")
        else:
            dino_score_a2b = None
            print("Warning: No valid scores for A→B")
        
        # 计算B→A的DINO分数
        print(f"\nCalculating DINO scores from {os.path.basename(folder_b)} to {os.path.basename(folder_a)}...")
        scores_b2a = []
        valid_scores_b2a = []
        
        image_iter = tqdm(images_b) if use_tqdm and TQDM_AVAILABLE else images_b
        for idx, img_path_b in enumerate(image_iter):
            if idx >= len(images_a):
                break
                
            img_path_a = images_a[idx]
            dino_ssim = self.calculate_dino_score(img_path_b, img_path_a)
            scores_b2a.append(dino_ssim)
            if dino_ssim is not None:
                valid_scores_b2a.append(dino_ssim)
        
        if valid_scores_b2a:
            dino_score_b2a = np.mean(valid_scores_b2a)
            print(f"Successfully processed {len(valid_scores_b2a)}/{len(scores_b2a)} image pairs for B→A")
        else:
            dino_score_b2a = None
            print("Warning: No valid scores for B→A")
        
        # 打印结果
        print(f"\nResults:")
        if dino_score_a2b is not None:
            print(f"DINO-Struct score ({os.path.basename(folder_a)} → {os.path.basename(folder_b)}): {dino_score_a2b:.4f}")
        else:
            print(f"DINO-Struct score ({os.path.basename(folder_a)} → {os.path.basename(folder_b)}): None")
            
        if dino_score_b2a is not None:
            print(f"DINO-Struct score ({os.path.basename(folder_b)} → {os.path.basename(folder_a)}): {dino_score_b2a:.4f}")
        else:
            print(f"DINO-Struct score ({os.path.basename(folder_b)} → {os.path.basename(folder_a)}): None")
        
        return dino_score_a2b, dino_score_b2a, scores_a2b, scores_b2a
