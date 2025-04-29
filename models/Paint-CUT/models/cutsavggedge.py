import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # 实现各种优化算法的包
from skimage import filters
from torch.optim import lr_scheduler  # 提供了一些根据epoch训练次数来调整学习率的方法，一般情况下会设置随着epoch的增大而减小学习率从而达更好的训练效果
from torch.utils.data import DataLoader  # 数据装载器

from models.base import BaseModel
from models.discriminator import Discriminator
from models.generatorshuffle import Generator
from models.projector import Head  # MLP
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16
from scipy.ndimage import filters

##化简前
# 计算特征提取模块的感知损失
# def vgg16_loss(feature_module, loss_func, y, y_):
#     out = feature_module(y)
#     out_ = feature_module(y_)
#     loss = loss_func(out, out_)
#     return loss
#
# # 化简前
# # 获取指定的特征提取模块
# def get_feature_module(layer_index, device=None):
#     vgg = vgg16(pretrained=True, progress=True).features
#     vgg.eval()
#
#     # 冻结参数
#     for parm in vgg.parameters():
#         parm.requires_grad = False
#
#     feature_module = vgg[0:layer_index + 1]
#     feature_module.to(device)
#     return feature_module
#
#
# # 计算指定的组合模块的感知损失
# class PerceptualLoss(nn.Module):
#     def __init__(self, loss_func, layer_indexs=None, device=None):
#         super(PerceptualLoss, self).__init__()
#         self.creation = loss_func
#         self.layer_indexs = layer_indexs
#         self.device = device
#
#     def forward(self, y, y_):
#         loss = 0
#         for index in self.layer_indexs:
#             feature_module = get_feature_module(index, self.device)
#             loss += vgg16_loss(feature_module, self.creation, y, y_)
#         return loss
#
#
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.ones((1, 3, 256, 256))
#     y = torch.zeros((1, 3, 256, 256))
#     x,y=x.to(device),y.to(device)
#
#     layer_indexs = [3, 8, 15, 22]
#     # 基础损失函数：确定使用那种方式构成感知损失，比如MSE、MAE
#     loss_func = nn.MSELoss().to(device)
#     # 感知损失
#     creation = PerceptualLoss(loss_func, layer_indexs, device)
#     perceptual_loss=creation(x,y)
#     print(perceptual_loss)

#化简后
#计算特征提取模块的感知损失
# def vgg16_loss(feature_module, loss_func, y, y_):
#     out = feature_module(y)
#     out_ = feature_module(y_)
#     loss = loss_func(out, out_)
#     return loss
#
# def get_feature_module(device=None):
#     vgg = vgg16(pretrained=True, progress=True).features.to(device)
#     vgg.eval()
#
#     # 冻结参数
#     for parm in vgg.parameters():
#         parm.requires_grad = False
#     return vgg
#
#
# # 1. 定义辅助函数获得中间特征
# def get_features(x, model, layer_indexs):
#     features = []
#     for index2, layer in enumerate(model):
#         for index1 in layer_indexs:
#             if index1 == index2:
#                 temp = layer(x)
#                 features.append(temp)
#     return features
#
#
# # 计算指定的组合模块的感知损失
# # 计算指定的组合模块的感知损失
# def PerceptualLoss(y,y_,layer_weights=None, layer_indexs=None, device=None):
#         mse = nn.MSELoss()
#         loss = 0
#         features1 = get_features(y, get_feature_module(device),layer_indexs)
#         features2 = get_features(y_, get_feature_module(device), layer_indexs)
#         for i in range(len(features1)):
#             loss =loss+ layer_weights[i] * mse(features1[i], features2[i])
#         return loss
# 获取指定的特征提取模块，并冻结参数
def get_feature_module(layer_index):
    vgg = vgg16(pretrained=True, progress=True).features
    vgg.eval()

    for param in vgg.parameters():
        param.requires_grad = False

    feature_module = vgg[0:layer_index + 1]

    return feature_module


# 计算指定的组合模块的感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, loss_func, device=None):
        super(PerceptualLoss, self).__init__()
        self.creation = loss_func
        self.device = device

        # 获取四个特征提取模块并存储在实例变量中
        self.feature_modules = nn.ModuleList([get_feature_module(i) for i in [3,8]])
        for module in self.feature_modules:
            module.to(device)

    def forward(self, y, y_):
        loss = 0
        for feature_module in self.feature_modules:
            out = feature_module(y)
            out_ = feature_module(y_)
            loss += self.creation(out, out_)
        return loss


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.ones((1, 3, 256, 256))
#     y = torch.zeros((1, 3, 256, 256))
#     x,y=x.to(device),y.to(device)
#
#     layer_indexs = [3, 8, 15, 22]
#     # 基础损失函数：确定使用那种方式构成感知损失，比如MSE、MAE
#     loss_func = nn.MSELoss().to(device)
#     # 感知损失
#     creation = PerceptualLoss(loss_func, layer_indexs, device)
#     perceptual_loss=creation(x,y)
#     print(perceptual_loss)

class ContrastiveModel(BaseModel):  # 对比学习模型
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.model_names = ['D_Y', 'G', 'H']
        self.loss_names = ['G_adv', 'D_Y', 'G', 'NCE']
        self.visual_names = ['X', 'Y', 'Y_fake']
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['Y_idt']

        self.D_Y = Discriminator().to(self.device)  # 将模型加载到指定的设备上
        self.G = Generator().to(self.device)
        self.H = Head().to(self.device)

        self.opt_D_Y = optim.Adam(self.D_Y.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"],
                                  betas=(0.5, 0.999), )  # adam优化算法
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"],
                                betas=(0.5, 0.999), )  # 前一个表示梯度运行平均值系数和梯度平方的系数，后一个是梯度平方前的系数
        self.opt_H = optim.Adam(self.H.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"],
                                betas=(0.5, 0.999), )

        self.l1 = nn.L1Loss()  # l1损失
        self.mse = nn.MSELoss()  # 均方误差损失
        loss_func = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')  # 交叉熵损失函数
        self.perceptual_loss = PerceptualLoss(loss_func, self.device)

        if self.config["TRAINING_SETTING"]["LOAD_MODEL"]:
            self.load_networks(self.config["TRAINING_SETTING"]["LOAD_EPOCH"])

        lambda_lr = lambda epoch: 1.0 - max(0, epoch - self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2) / (
                    self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2)
        # 一个自定义的函数，这个函数以训练epoch为输入，学习率倍率系数为输出
        self.scheduler_disc = lr_scheduler.LambdaLR(self.opt_D_Y, lr_lambda=lambda_lr)
        self.scheduler_gen = lr_scheduler.LambdaLR(self.opt_G, lr_lambda=lambda_lr)
        self.scheduler_mlp = lr_scheduler.LambdaLR(self.opt_H, lr_lambda=lambda_lr)

    def set_input(self, input):
        self.X, self.Y = input

    def forward(self):
        self.Y = self.Y.to(self.device)
        self.X = self.X.to(self.device)
        self.Y_fake = self.G(self.X)
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.Y_idt = self.G(self.Y)

    def inference(self, X):
        self.eval()
        with torch.no_grad():  # 表明当前计算不需要反向传播
            X = X.to(self.device)
            Y_fake = self.G(X)
        return Y_fake

    def optimize_parameters(self):  # 优化参数
        # forward
        with torch.autograd.set_detect_anomaly(True):
            self.forward()

        # update G and H
            self.set_requires_grad(self.D_Y, False)
            self.opt_G.zero_grad()
            self.opt_H.zero_grad()
            self.loss_G = self.compute_G_loss()
            self.loss_G.backward()
            self.opt_G.step()
            self.opt_H.step()
        # update D
            self.set_requires_grad(self.D_Y, True)
            self.opt_D_Y.zero_grad()  # 将梯度清零
            self.loss_D_Y = self.compute_D_loss()  # 计算模型训练的损失
            self.loss_D_Y.backward()  # 反向传播
            self.opt_D_Y.step()  # .step()更新参数？？？

    def scheduler_step(self):
        self.scheduler_disc.step()  # 学习率更新
        self.scheduler_gen.step()
        self.scheduler_mlp.step()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 计算特征提取模块的感知损失

    def compute_D_loss(self):
        # Fake
        fake = self.Y_fake.detach()  # detach()返回一个新的tensor，并且这个tensor是从当前的计算图中分离出来的
        pred_fake = self.D_Y(fake)  # D_Y判别器
        self.loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))  # 均方误差 torch.zeros_like生成和括号内变量维度一致的全是零的内容
        # Real
        self.pred_real = self.D_Y(self.Y)
        self.loss_D_real = self.mse(self.pred_real, torch.ones_like(self.pred_real))

        self.loss_D_Y = (self.loss_D_fake + self.loss_D_real) / 2
        return self.loss_D_Y


    def compute_G_loss(self):
        fake = self.Y_fake
        pred_fake = self.D_Y(fake)
        self.loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))
        # 感知损失
        #perceptual_loss=self.perceptual_loss(self.Y,self.Y_fake)
        #perceptual_loss=self.perceptual_loss(self.X,self.Y_fake)
        perceptual_loss=self.perceptual_loss(self.Y,self.Y_fake)*0.6+self.perceptual_loss(self.X,self.Y_fake)*0.4
        self.loss_NCE = self.calculate_NCE_loss(self.X, self.Y_fake)# NCE损失？？？
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.Y, self.Y_idt)
            self.loss_NCE = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        self.edge_loss = self.calculate_edge_loss(self.X, self.Y_fake)

        #计算总损失
        self.loss_G = self.loss_G_adv + self.loss_NCE + perceptual_loss+self.edge_loss

        # self.loss_G = self.loss_G_adv + self.loss_NCE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):  # 计算NCE损失
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):  # 将可迭代的对象作为参数，讲对象中对应的元素打包成一个元组，然后返回由这些元组组成的对象
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07  # 矩阵相乘，针对二维矩阵
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=self.device))
        return loss
    #计算边缘损失
    def calculate_edge_loss(self, input, fake):
        # 1.tensor转彩色图片
        x = input.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy()
        # 将彩色图片转灰度图像
        # x= cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        # x=Image.fromarray(x)
        x = Image.fromarray(x.astype('uint8')).convert('L')
        x = np.array(x)
        y = fake.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().detach().numpy()
        # y = y.convert('L')
        y = Image.fromarray(y.astype('uint8')).convert('L')
        y = np.array(y)
        # 高斯滤波
        x = filters.gaussian_filter(x, 1)
        y = filters.gaussian_filter(y, 1)
        # 对比度抑制自适应直方图均衡

        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        # planes = cv2.split(x)  # 将图片分为三个单通道，
        # for i in range(0, 3):
        #     # 可能是因为读取到的图片是三通到，而cv2.createCLAHE只能对单通道图片处理
        #     # 所有用cv2.split()将图片变为三个单通道，然后在应用cv2.createCLAHE处理
        #     planes[i] = clahe.apply(planes[i])
        # x = cv2.merge(planes)
        # plane = cv2.split(y)  # 将图片分为三个单通道，
        # for i in range(0, 3):
        #     # 可能是因为读取到的图片是三通到，而cv2.createCLAHE只能对单通道图片处理
        #     # 所有用cv2.split()将图片变为三个单通道，然后在应用cv2.createCLAHE处理
        #     planes[i] = clahe.apply(plane[i])
        # y = cv2.merge(plane)
        x = clahe.apply(x)
        y = clahe.apply(y)
        # Canny算子检测边缘
        x = cv2.Canny(x, 100, 200, 5)
        y = cv2.Canny(y, 100, 200, 5)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        edge_loss = self.mse(x, y)
        return edge_loss
