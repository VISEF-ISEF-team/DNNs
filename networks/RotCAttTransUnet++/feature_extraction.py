import torch 
import torch.nn as nn

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
            
    def forward(self, x):
        return self.act(self.batch_norm(self.conv(x)))
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv):
        super().__init__()
        layers = []
        self.maxpool = nn.MaxPool2d(2)
        layers.append(ConvBatchNorm(in_channels, out_channels))
        for _ in range(nb_Conv - 1): 
            layers.append(ConvBatchNorm(out_channels, out_channels))
        self.nConvs = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.nConvs(self.maxpool(x))
        
class DownSampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.in_channels
        channel_nums = config.channel_nums
        self.convert = ConvBatchNorm(in_channels, channel_nums[0])
        self.down1 = DownBlock(channel_nums[0], channel_nums[1], nb_Conv=2)
        self.down2 = DownBlock(channel_nums[1], channel_nums[2], nb_Conv=2)
        self.down3 = DownBlock(channel_nums[2], channel_nums[3], nb_Conv=2)
        self.down4 = DownBlock(channel_nums[3], channel_nums[3], nb_Conv=2)
                    
    def forward(self, x):
        x = x.float()
        x1 = self.convert(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1,x2,x3,x4,x5