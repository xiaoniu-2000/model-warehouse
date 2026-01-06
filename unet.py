'''
Author: xiaoniu
Date: 2026-01-06 16:28:04
LastEditors: xiaoniu
LastEditTime: 2026-01-06 16:32:17
Description: unet with residual block
'''
import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1, batch_norm=True, activation=None):
        super(ConvolutionalBlock, self).__init__()
        layers = list()
        
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))
        #是否BN层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        #激活函数选择层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmod':
            layers.append(nn.Sigmoid())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input) 
        return output

class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, in_channel=64,mid_channel=None,out_channel=64):
        super(ResidualBlock, self).__init__()
        if mid_channel is None:
            mid_channel=out_channel
        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=in_channel, out_channels=mid_channel, kernel_size=kernel_size,
                                              batch_norm=True, activation='leakyrelu')
        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=mid_channel, out_channels=out_channel, kernel_size=kernel_size,
                                              batch_norm=True)
        self.residualconv = ConvolutionalBlock(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                                              batch_norm=False)


    def forward(self, input):
        #能residual就residual
        
        residual = input
        residual = self.residualconv(input)
        output = self.conv_block1(input) 
        output = self.conv_block2(output) 
        output = output + residual  
        return output
    



class Upsample(nn.Module):
    def __init__(self, dim,type='pixel_shuffle',scaling_factor=2,kernel_size=3):
        super().__init__()
        #子像素卷积上采样
        if type=='pixel_shuffle':
            self.up=nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim * (scaling_factor ** 2),
                                  kernel_size=kernel_size, padding=kernel_size // 2),
                                  nn.PixelShuffle(upscale_factor=scaling_factor),
                                  nn.PReLU())
        #反卷积上采样
        elif type=='convT':
            self.up=nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2)
        #插值上采样nearest | linear | bilinear | bicubic | trilinear | area |
        else:
            self.up = nn.Upsample(scale_factor=scaling_factor, mode=type)

    def forward(self, x):
        return self.up(x)

class Downsample(nn.Module):
    def __init__(self, dim,type='conv'):
        super().__init__()
        #卷积下采样
        if type=='conv':
        #最大池化下采样
            self.down = nn.Conv2d(dim, dim, 3, 2, 1)
        elif type=='max':
        #平均池化下采样
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        elif type=='mean':
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
            

    def forward(self, x):
        return self.down(x)

    
class Unet(nn.Module):
    def __init__(self,inc=4,outc=1,l=2):
        #inc模型输入通道数，outc模型输出通道数，l模型深度，层数（包括level0）
        super(Unet, self).__init__()
        down_filters=[inc,64,128,256,512,1024,2048,4096,8192]
        up_filters=[8192,4096,2048,1024,512,256,128,64,64]
        if l!=0:
            mid_filters=down_filters[l+1]
            down_filters=down_filters[0:l+1]            
            up_filters=up_filters[::-1][:l+2][::-1]
  
        #编码器
        Encoderlayers=list()
        for k in range(len(down_filters)-1):
            Encoderlayers.append(ResidualBlock(in_channel=down_filters[k],
                                               out_channel=down_filters[k+1]))
            Encoderlayers.append(Downsample(dim=down_filters[k+1],type='conv'))
        self.Encoder = nn.Sequential(*Encoderlayers)
        #最高语义层
        self.Bridge=ResidualBlock(in_channel=down_filters[-1],
                                  mid_channel=mid_filters,
                                  out_channel=up_filters[1])
        #解码器
        Decoderlayers=list()
        for k in range(len(up_filters)-2):
            Decoderlayers.append(Upsample(dim=up_filters[k+1],type='pixel_shuffle'))
            Decoderlayers.append(ResidualBlock(in_channel=up_filters[k],
                                               mid_channel=up_filters[k+1],
                                               out_channel=up_filters[k+2]))

        self.Decoder = nn.Sequential(*Decoderlayers)
        #激活层
        self.last=ConvolutionalBlock(in_channels=up_filters[-1], 
                                out_channels=outc, 
                                batch_norm=False, 
                                activation=None)

    def forward(self,x):
        feats=[x]
        #编码器
        for layer in self.Encoder:
            if isinstance(layer, ResidualBlock):
                x = layer(x)
                feats.append(x)
            else:
                x = layer(x)
        #最高语义层
        x=self.Bridge(x)
        
        #解码器
        for layer in self.Decoder:
            if isinstance(layer, ResidualBlock):

                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)
        #激活层
        x=self.last(x)
        
        return x
if __name__ == '__main__':
    x=torch.randn(1,16,424,624)
    _,_,w,h=x.shape
    l=0
    while w%2==0 and h%2==0:
        l=l+1
        w=w/2
        h=h/2
    print('level:',l)
    net=Unet(inc=16,outc=1,l=3)
    print("Num params: ", sum(p.numel() for p in net.parameters()))
    print(net(x).shape)
