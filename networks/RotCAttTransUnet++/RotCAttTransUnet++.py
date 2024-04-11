from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import get_config
from feature_extraction import *
from encoder_transformer import *
from reconstruction import *
from decoder_attention import *
from rotatory_attention import *


class RotCAttTransUNetDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_nums = config.channel_nums
        num_classes = config.num_classes
        
        self.down_sampling = DownSampling(config)
        self.channel_transformer = ChannelTransformer(config)
        self.reconstruction = Reconstruction(config)
        self.up_sampling = UpSampling(config)
        self.out = nn.Conv2d(channel_nums[0], num_classes, kernel_size=(1,1), stride=(1,1))
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.down_sampling(x)
        enc1, enc2, enc3, enc4, a_weights = self.channel_transformer(x1, x2, x3, x4)
        r1, r2, r3, r4 = self.reconstruction(enc1, enc2, enc3, enc4)

        y = self.up_sampling(r1, r2, r3, r4, x5)
        y = self.out(y)
        return y, a_weights
        
config = get_config()
model = RotCAttTransUNetDense(config=config).cuda()
input = torch.rand(3, 1, 256, 256).cuda()
logits, _ = model(input)
print(logits.size())