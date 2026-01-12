'''
Author: xiaoniu
Date: 2026-01-06 16:35:51
LastEditors: xiaoniu
LastEditTime: 2026-01-06 17:01:58
Description: toy model for unet
'''
import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block1(nn.Module):
    #卷积一次的模块，在最高语义和解码器中使用
    def __init__(self,in_channel,out_channel):
        super(Conv_Block1, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3, stride=1, padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1)
            )
    def forward(self,x):
        return self.layer(x)
    
class Conv_Block2(nn.Module):
    #卷积两次的模块，在编码器中使用
    def __init__(self,in_channel,out_channel):
        super(Conv_Block2, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3, stride=1, padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),
            nn.Conv2d(out_channel,out_channel,kernel_size=3, stride=1, padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),
            )
    def forward(self,x):
        return self.layer(x)

class Conv_Block3(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block3, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3, stride=1, padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),
            nn.Conv2d(out_channel,1,kernel_size=3, stride=1, padding=1,padding_mode='reflect'),
            )
    def forward(self,x):
        return self.layer(x)
    
class DownSample(nn.Module):
    #下采样模块，使用最大池化
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    #上采样模块，使用线性插值
    def __init__(self):
        super(UpSample,self).__init__()
    
    def forward(self,x):
        up_features=F.interpolate(x,scale_factor=2,mode='bilinear')
        return up_features
class UpSample2(nn.Module):
    #上采样模块，使用线性插值
    def __init__(self):
        super(UpSample2,self).__init__()
    
    def forward(self,x):
        up_features=F.interpolate(x,scale_factor=4,mode='bilinear')
        return up_features
    
class Connect(nn.Module):
    #跳跃连接模块
    def __init__(self):
        super(Connect, self).__init__()
    def forward(self,x,feature_map):
        connected_features = torch.cat((x, feature_map), dim=1)
        return connected_features


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.d=DownSample()
        self.u=UpSample()
        self.u2=UpSample2()
        self.Connect=Connect()
        #编码器
        self.c1=Conv_Block2(1,64)
        self.c2=Conv_Block2(64,128)
        self.c3=Conv_Block2(128,256)
        self.c4=Conv_Block2(256,512)
        #最高语义
        self.c5_1=Conv_Block1(512,1024)
        self.c5_2=Conv_Block1(1024,512)
        #解码器
        self.c6_1=Conv_Block1(1024,512)
        self.c6_2=Conv_Block1(512,256)
        self.c7_1=Conv_Block1(512,256)
        self.c7_2=Conv_Block1(256,128)
        self.c8_1=Conv_Block1(256,128)
        self.c8_2=Conv_Block1(128,64)
        self.c9_1=Conv_Block1(128,64)
        self.c9_2=Conv_Block1(64,32)
        self.c10=Conv_Block3(33,16)

        
    def forward(self,x):
        #编码器
        R1 = self.c1(x)
        R2 = self.c2(self.d(R1))
        R3 = self.c3(self.d(R2))
        R4 = self.c4(self.d(R3))
        R5 = self.c5_2(self.c5_1(self.d(R4)))
        #解码器
        O1=self.c6_2(self.c6_1(self.Connect(R4,self.u(R5))))
        O2=self.c7_2(self.c7_1(self.Connect(R3,self.u(O1))))
        O3=self.c8_2(self.c8_1(self.Connect(R2,self.u(O2))))
        O4=self.c9_2(self.c9_1(self.Connect(R1,self.u(O3))))
        O5=self.c10(self.u2(self.Connect(x,O4)))

        return O5

if __name__ == '__main__':
    x=torch.randn(1,1,32,64)
    net=UNet()
    print(net(x).shape)
    
    
    
    
    
    
    
    
    
    
    
    