import torch
import torch.nn as nn
from torch.nn import functional as F

#最邻近差值
class up_conv(nn.Module):
    def __init__(self,ch_in):
        super(up_conv,self).__init__()
        self.layer=nn.Conv2d(ch_in,ch_in,1,1)
    def forward(self,x):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return out


#转置卷积
# class Upsample(nn.Module):
#     def __init__(self, features):
#         super().__init__()
#         layers = [
#             nn.ReplicationPad2d(1),
#             nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=3)
#         ]
#         self.model = nn.Sequential(*layers)    #*将输入迭代器拆成一个个元素，可以有多个layer？？
#
#     def forward(self, input):
#         return self.model(input)
