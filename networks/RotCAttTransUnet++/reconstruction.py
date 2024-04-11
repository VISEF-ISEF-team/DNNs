import numpy as np
import torch
import torch.nn as nn

class ReconBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        if kernel_size == 3: padding = 1
        else: padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.act(out)
        return out

class Reconstruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_nums = config.channel_nums
        patches_size = config.patches_size
        self.reconstruct_1 = ReconBlock(channel_nums[0], channel_nums[0], kernel_size=1,
                                        scale_factor=(patches_size[0], patches_size[0]))
        self.reconstruct_2 = ReconBlock(channel_nums[1], channel_nums[1], kernel_size=1,
                                        scale_factor=(patches_size[1], patches_size[1]))
        self.reconstruct_3 = ReconBlock(channel_nums[2], channel_nums[2], kernel_size=1,
                                        scale_factor=(patches_size[2], patches_size[2]))
        self.reconstruct_4 = ReconBlock(channel_nums[3], channel_nums[3], kernel_size=1, 
                                        scale_factor=(patches_size[3], patches_size[3]))
        
    def forward(self, enc1, enc2, enc3, enc4):
        r1 = self.reconstruct_1(enc1)
        r2 = self.reconstruct_2(enc2)
        r3 = self.reconstruct_3(enc3)
        r4 = self.reconstruct_4(enc4)
        return r1, r2, r3, r4
