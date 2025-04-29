#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DINO-Struct分数计算器 - 用于计算两个图像文件夹之间的结构相似度
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import time
import matplotlib.pyplot as plt
import io


def build_transform(image_prep):
    """
    构建图像转换，与training_utils.py中的build_transform完全一致
    
    Args:
        image_prep: 图像预处理方式
        
    Returns:
        转换函数
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


def safe_open_image(image_path):
    """
    安全地打开图像文件，处理各种可能的错误
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL.Image或None（如果出错）
    """
    try:
        # 首先检查文件是否存在
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return None
            
        # 检查文件大小
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print(f"文件大小为0: {image_path}")
            return None
            
        # 尝试读取文件内容到内存
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        # 从内存中加载图像
        image = Image.open(io.BytesIO(image_data))
        
        # 确保图像可以转换为RGB
        if image.mode not in ['RGB', 'RGBA', 'L']:
            print(f"不支持的图像模式 {image.mode}: {image_path}")
            return None
            
        # 转换为RGB
        image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"加载图像时出错 {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_image_loading(image_path, T_val):
    """
    测试图像加载和转换
    
    Args:
        image_path: 图像路径
        T_val: 转换函数
        
    Returns:
        成功与否
    """
    try:
        print(f"\n测试图像加载: {image_path}")
        img = safe_open_image(image_path)
        
        if img is not None:
            print(f"原始图像大小: {img.size}, 模式: {img.mode}")
            
            # 应用T_val转换
            img_transformed = T_val(img)
            print(f"转换后图像大小: {img_transformed.size}, 模式: {img_transformed.mode}")
            
            # 显示图像
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("原始图像")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img_transformed)
            plt.title("转换后图像")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            return True
        else:
            print("无法加载测试图像")
            return False
    except Exception as e:
        print(f"测试图像加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def calculate_dino_scores(folder_a, folder_b, device=None, val_img_prep="resize_256", show_test_image=True, save_csv=True):
    """
    计算两个文件夹之间的DINO-Struct分数
    
    Args:
        folder_a: 第一个文件夹路径
        folder_b: 第二个文件夹路径
        device: 使用的设备 ("cuda" 或 "cpu")，如果为None则自动检测
        val_img_prep: 验证图像预处理方式
        show_test_image: 是否显示测试图像
        save_csv: 是否保存CSV文件
        
    Returns:
        dino_score_a2b: 从A到B的平均DINO-Struct分数
        dino_score_b2a: 从B到A的平均DINO-Struct分数
        scores_a2b: 从A到B的所有分数列表
        scores_b2a: 从B到A的所有分数列表
    """
    # 确保可以导入my_utils
    UTILS_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if UTILS_DIR_PATH not in sys.path:
        sys.path.insert(0, UTILS_DIR_PATH)
        print(f"Added '{UTILS_DIR_PATH}' to sys.path")
    
    # 检查CUDA是否可用
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_a):
        print(f"警告: 文件夹A不存在: {folder_a}")
        return None, None, [], []
    if not os.path.exists(folder_b):
        print(f"警告: 文件夹B不存在: {folder_b}")
        return None, None, [], []
    
    # 计算DINO分数
    try:
        # 导入DinoStructureLoss
        try:
            print("\n导入DinoStructureLoss...")
            from my_utils.dino_struct import DinoStructureLoss
            print("成功导入DinoStructureLoss")
        except ImportError:
            print("无法导入DinoStructureLoss，尝试直接从当前目录导入...")
            try:
                from src.my_utils.dino_struct import DinoStructureLoss
                print("成功从src.my_utils导入DinoStructureLoss")
            except ImportError:
                print("无法导入DinoStructureLoss，请确保文件存在并且路径正确")
                return None, None, [], []
        
        print("\n初始化DinoStructureLoss...")
        start_time = time.time()
        
        # 初始化DINO-Struct损失
        net_dino = DinoStructureLoss(device=device)
        print(f"DinoStructureLoss initialized on {device}")
        
        # 构建图像转换
        T_val = build_transform(val_img_prep)
        print(f"Using image preprocessing: {val_img_prep}")
        
        print(f"初始化耗时: {time.time() - start_time:.2f}秒")
        
        # 获取图像文件
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        
        # 获取文件夹A中的图像
        images_a = []
        for ext in image_extensions:
            images_a.extend(glob.glob(os.path.join(folder_a, ext)))
        images_a = sorted([f for f in images_a if not os.path.basename(f).startswith('.')])
        
        # 获取文件夹B中的图像
        images_b = []
        for ext in image_extensions:
            images_b.extend(glob.glob(os.path.join(folder_b, ext)))
        images_b = sorted([f for f in images_b if not os.path.basename(f).startswith('.')])
        
        print(f"Found {len(images_a)} images in folder A: {os.path.basename(folder_a)}")
        print(f"Found {len(images_b)} images in folder B: {os.path.basename(folder_b)}")
        
        # 打印文件列表以便调试
        print("\n文件夹A中的前5个文件:")
        for i, f in enumerate(images_a[:5]):
            print(f"  {i+1}. {os.path.basename(f)} ({os.path.getsize(f)} bytes)")
        
        print("\n文件夹B中的前5个文件:")
        for i, f in enumerate(images_b[:5]):
            print(f"  {i+1}. {os.path.basename(f)} ({os.path.getsize(f)} bytes)")
        
        # 测试加载一个图像
        if show_test_image and images_a:
            test_image_loading(images_a[0], T_val)
        
        # 计算A→B的DINO分数
        print(f"\n计算DINO分数: {os.path.basename(folder_a)} → {os.path.basename(folder_b)}...")
        start_time = time.time()
        l_dino_scores_a2b = []
        
        for idx, input_img_path in enumerate(images_a):
            if idx >= len(images_b):
                break
                
            print(f"处理图像对 {idx+1}/{min(len(images_a), len(images_b))}", end="\r")
            
            try:
                with torch.no_grad():
                    # 使用安全的图像加载函数
                    input_img_pil = safe_open_image(input_img_path)
                    if input_img_pil is None:
                        print(f"\n无法加载图像A: {input_img_path}")
                        continue
                    
                    img_b_path = images_b[idx]
                    img_b_pil = safe_open_image(img_b_path)
                    if img_b_pil is None:
                        print(f"\n无法加载图像B: {img_b_path}")
                        continue
                    
                    # 应用转换
                    input_img = T_val(input_img_pil)
                    img_b = T_val(img_b_pil)
                    
                    # 使用DinoStructureLoss计算分数
                    a = net_dino.preprocess(input_img).unsqueeze(0).to(device)
                    b = net_dino.preprocess(img_b).unsqueeze(0).to(device)
                    dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                    l_dino_scores_a2b.append(dino_ssim)
            except Exception as e:
                print(f"\nError processing image pair {input_img_path} and {img_b_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 计算平均分数
        if l_dino_scores_a2b:
            dino_score_a2b = np.mean(l_dino_scores_a2b)
            print(f"\nSuccessfully processed {len(l_dino_scores_a2b)} image pairs for A→B")
        else:
            dino_score_a2b = None
            print("\nWarning: No valid scores for A→B")
        
        # 计算B→A的DINO分数
        print(f"\n计算DINO分数: {os.path.basename(folder_b)} → {os.path.basename(folder_a)}...")
        l_dino_scores_b2a = []
        
        for idx, input_img_path in enumerate(images_b):
            if idx >= len(images_a):
                break
                
            print(f"处理图像对 {idx+1}/{min(len(images_b), len(images_a))}", end="\r")
            
            try:
                with torch.no_grad():
                    # 使用安全的图像加载函数
                    input_img_pil = safe_open_image(input_img_path)
                    if input_img_pil is None:
                        print(f"\n无法加载图像B: {input_img_path}")
                        continue
                    
                    img_a_path = images_a[idx]
                    img_a_pil = safe_open_image(img_a_path)
                    if img_a_pil is None:
                        print(f"\n无法加载图像A: {img_a_path}")
                        continue
                    
                    # 应用转换
                    input_img = T_val(input_img_pil)
                    img_a = T_val(img_a_pil)
                    
                    # 使用DinoStructureLoss计算分数
                    a = net_dino.preprocess(input_img).unsqueeze(0).to(device)
                    b = net_dino.preprocess(img_a).unsqueeze(0).to(device)
                    dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                    l_dino_scores_b2a.append(dino_ssim)
            except Exception as e:
                print(f"\nError processing image pair {input_img_path} and {img_a_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 计算平均分数
        if l_dino_scores_b2a:
            dino_score_b2a = np.mean(l_dino_scores_b2a)
            print(f"\nSuccessfully processed {len(l_dino_scores_b2a)} image pairs for B→A")
        else:
            dino_score_b2a = None
            print("\nWarning: No valid scores for B→A")
        
        print(f"\n计算DINO分数耗时: {time.time() - start_time:.2f}秒")
        
        # 打印结果
        print(f"\n结果:")
        if dino_score_a2b is not None:
            print(f"DINO-Struct score ({os.path.basename(folder_a)} → {os.path.basename(folder_b)}): {dino_score_a2b:.4f}")
        else:
            print(f"DINO-Struct score ({os.path.basename(folder_a)} → {os.path.basename(folder_b)}): None")
            
        if dino_score_b2a is not None:
            print(f"DINO-Struct score ({os.path.basename(folder_b)} → {os.path.basename(folder_a)}): {dino_score_b2a:.4f}")
        else:
            print(f"DINO-Struct score ({os.path.basename(folder_b)} → {os.path.basename(folder_a)}): None")
        
        # 可视化分数分布
        if l_dino_scores_a2b and l_dino_scores_b2a:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(l_dino_scores_a2b, bins=20, alpha=0.7)
            plt.title(f'DINO Scores: {os.path.basename(folder_a)} → {os.path.basename(folder_b)}')
            plt.xlabel('Score')
            plt.ylabel('Count')
            
            plt.subplot(1, 2, 2)
            plt.hist(l_dino_scores_b2a, bins=20, alpha=0.7)
            plt.title(f'DINO Scores: {os.path.basename(folder_b)} → {os.path.basename(folder_a)}')
            plt.xlabel('Score')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.show()
        else:
            print("没有足够的有效分数来生成直方图")
        
        # 保存详细分数到CSV文件
        if save_csv:
            try:
                import pandas as pd
                # 将None值替换为NaN，以便在CSV中正确表示
                scores_a2b_for_csv = [np.nan if score is None else score for score in l_dino_scores_a2b]
                scores_b2a_for_csv = [np.nan if score is None else score for score in l_dino_scores_b2a]
                
                # 创建DataFrame
                scores_df = pd.DataFrame({
                    f"{os.path.basename(folder_a)}→{os.path.basename(folder_b)}": scores_a2b_for_csv,
                    f"{os.path.basename(folder_b)}→{os.path.basename(folder_a)}": scores_b2a_for_csv
                })
                
                # 添加图像文件名
                image_files_a = [os.path.basename(f) for f in images_a[:len(scores_a2b_for_csv)]]
                scores_df['Image A'] = image_files_a
                
                # 保存CSV
                csv_path = "dino_scores.csv"
                scores_df.to_csv(csv_path)
                print(f"详细分数已保存到 {csv_path}")
                
                # 显示前几行
                print("\nCSV文件内容预览:")
                print(scores_df.head())
            except Exception as e:
                print(f"保存CSV时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n计算完成!")
        return dino_score_a2b, dino_score_b2a, l_dino_scores_a2b, l_dino_scores_b2a
            
    except Exception as e:
        print(f"计算DINO分数时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, [], []


def main():
    """
    主函数，用于命令行调用
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="计算两个图像文件夹之间的DINO-Struct分数")
    parser.add_argument("--folder_a", type=str, required=True, help="第一个文件夹路径")
    parser.add_argument("--folder_b", type=str, required=True, help="第二个文件夹路径")
    parser.add_argument("--device", type=str, default=None, help="使用的设备 (cuda 或 cpu)")
    parser.add_argument("--val_img_prep", type=str, default="resize_256", help="验证图像预处理方式")
    parser.add_argument("--no_test_image", action="store_true", help="不显示测试图像")
    parser.add_argument("--no_csv", action="store_true", help="不保存CSV文件")
    
    args = parser.parse_args()
    
    calculate_dino_scores(
        args.folder_a, 
        args.folder_b, 
        device=args.device, 
        val_img_prep=args.val_img_prep,
        show_test_image=not args.no_test_image,
        save_csv=not args.no_csv
    )


if __name__ == "__main__":
    main()
