import torch
import torch.nn as nn
from models.downsample import Downsample
from models.upsample import Upsample
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

class ResnetBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features),
            nn.ReLU(True),
            ShuffleAttention(features),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x):
        return x + self.model(x)




class ShuffleAttention(nn.Module):

    def __init__(self, features=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(features // (2 * G), features // (2 * G))
        self.cweight = Parameter(torch.zeros(1, features // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, features // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, features // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, features // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


# if __name__ == '__main__':
#     input=torch.randn(50,512,7,7)
#     se = ShuffleAttention(channel=512,G=8)
#     output=se(input)
#     print(output.shape)

class GeneratorBasicBlock(nn.Module):
    def __init__(self, in_features, out_features, do_upsample=False, do_downsample=False):
        super().__init__()
        
        self.do_upsample = do_upsample
        self.do_downsample = do_downsample

        if self.do_upsample:
            self.upsample = Upsample(in_features)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.instancenorm = nn.InstanceNorm2d(out_features)
        self.relu = nn.ReLU(True)
        if self.do_downsample:
            self.downsample = Downsample(out_features)
        
    def forward(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x
    
    def fordward_hook(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        x_hook = self.conv(x)
        x = self.instancenorm(x_hook)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x_hook, x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, residuals=9):
        super().__init__()
        self.residuals = residuals

        self.reflectionpad = nn.ReflectionPad2d(3)
        self.block1 = nn.Sequential(
                        nn.Conv2d(in_channels, features, kernel_size=7),
                        nn.InstanceNorm2d(features),
                        nn.ReLU(True)
                        )

        self.downsampleblock2 = GeneratorBasicBlock(features, features * 2, do_upsample=False, do_downsample=True)
        self.downsampleblock3 = GeneratorBasicBlock(features * 2, features * 4, do_upsample=False, do_downsample=True)

        self.resnetblocks4 = nn.Sequential(*[ResnetBlock(features * 4) for _ in range(residuals)])

        self.upsampleblock5 = GeneratorBasicBlock(features * 4, features * 2, do_upsample=True, do_downsample=False)
        self.upsampleblock6 = GeneratorBasicBlock(features * 2, features, do_upsample=True, do_downsample=False)

        self.block7 = nn.Sequential(
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(features, in_channels, kernel_size=7),
                        nn.Tanh(),
                        )

    def append_sample_feature(self, feature, return_ids, return_feats, mlp_id=0, num_patches=256, patch_ids=None):
        B, H, W = feature.shape[0], feature.shape[2], feature.shape[3]
        feature_reshape = feature.permute(0, 2, 3, 1).flatten(1, 2) # B, F, C
        if patch_ids is not None:
            patch_id = patch_ids[mlp_id]
        else:
            patch_id = torch.randperm(feature_reshape.shape[1])
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
        x_sample = feature_reshape[:, patch_id, :].flatten(0, 1)

        return_ids.append(patch_id)
        return_feats.append(x_sample)

    def forward(self, x, encode_only=False, num_patches=256, patch_ids=None):
        if not encode_only:
            x = self.reflectionpad(x)
            x = self.block1(x)
            x = self.downsampleblock2(x)
            x = self.downsampleblock3(x)
            x = self.resnetblocks4(x)
            x = self.upsampleblock5(x)
            x = self.upsampleblock6(x)
            x = self.block7(x)
            return x

        else:
            return_ids = []
            return_feats = []
            mlp_id = 0

            x = self.reflectionpad(x)
            self.append_sample_feature(x, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
            mlp_id += 1

            x = self.block1(x)
            
            x_hook, x = self.downsampleblock2.fordward_hook(x)
            self.append_sample_feature(x_hook, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
            mlp_id += 1

            x_hook, x = self.downsampleblock3.fordward_hook(x)
            self.append_sample_feature(x_hook, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
            mlp_id += 1

            for resnet_layer_id, resnet_layer in enumerate(self.resnetblocks4):
                x = resnet_layer(x)
                if resnet_layer_id in [0, 4]:
                    self.append_sample_feature(x, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
                    mlp_id += 1

            return return_feats, return_ids


if __name__ == "__main__":

    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    G = Generator()
    x = G(x)
    print(x.shape)

    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    G = Generator()
    feat_k_pool, sample_ids = G(x, encode_only=True, patch_ids=None)
    feat_q_pool, _ = G(x, encode_only=True, patch_ids=sample_ids)
    print(len(feat_k_pool))