from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import ml_collections
from configs import get_config
from vit_resnet_skip import ResNetV2


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__(self)
        
    def forward(self, x):
        pass

class Embedding(nn.Module):
    def __init__(self, config, input_channels):
        super().__init__()
        
        if config.patches.get("grid") is not None: 
            grid_size = config.patches["grid"]
            patch_size = (config.width // 16 // grid_size[0], config.height // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            num_patches = (config.width // patch_size_real[0]) * (config.height // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            num_patches = (config.width // patch_size[0]) * (config.height // patch_size[1])
            self.hybrid = False
            
        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            input_channels = self.hybrid_model.width * 16
        
        self.patch_embeddings = nn.Conv2d (
            in_channels=input_channels,
            out_channels=config.d_f,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.d_f))
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):    
        if self.hybrid: 
            x, features = self.hybrid_model(x)
        else: features = None
        
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features
    
class ExtractKeyVal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_K = nn.Linear(config.d_f, config.d_k)
        self.W_V = nn.Linear(config.d_f, config.d_v)
        
    def forward(self, x):
        K = self.W_K(x)
        V = self.W_V(x)
        return K, V
        
class ActivatedGeneral(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = nn.Linear(config.d_k, config.d_f)
        self.act = nn.Tanh()
        self.align = nn.Softmax(dim=0)
    
    def forward(self, r, K):
        e = self.act(self.W(K) @ r)
        a = self.align(e)
        return a
                    
class SingleAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.extract_key_val = ExtractKeyVal(config)
        self.activated_general = ActivatedGeneral(config)
        
    def forward(self, r, x):
        K, V = self.extract_key_val(x)
        a = self.activated_general(r, K)
        r_x = torch.matmul(V.transpose(-1,-2), a)
        return r_x
    
class RotatoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.single_attention = SingleAttention(config)
        
    def forward(self, x):
        left = x[0]
        target = x[1]
        right = x[2]
        
        r_t = torch.mean(target, dim=0)
        r_l = self.single_attention(r_t, left)
        r_r = self.single_attention(r_t, right)
        
        r_l_t = self.single_attention(r_l, target)
        r_r_t = self.single_attention(r_r, target)
        
        r = torch.concat([r_l, r_r, r_l_t, r_r_t])
        return r
        
class MultiheadAttention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.n_heads = config.transformer.num_heads
        self.d_f = config.d_f
        self.d_h = int(self.d_f / self.n_heads)
        
        self.W_Q = nn.Linear(config.d_f, config.d_q)
        self.W_K = nn.Linear(config.d_f, config.d_k)
        self.W_V = nn.Linear(config.d_f, config.d_v)
        self.W_O = nn.Linear(config.d_f, config.d_f)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config.transformer.att_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.att_dropout_rate)
        
    def _decompose(self, x):
        new_shape = x.size()[:-1] + (self.n_heads, self.d_h)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def _compose(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.d_f, )
        return x.view(*new_shape)
    
    def forward(self, x):
        
        Q = self._decompose(self.W_Q(x))
        K = self._decompose(self.W_K(x))
        V = self._decompose(self.W_V(x))
    
        e = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_h)
        a = self.softmax(e)
        a_weights = a if self.vis else None
        
        a = self.attn_dropout(a)
        r = self._compose(torch.matmul(a, V))
        r = self.proj_dropout(r)
        return r, a_weights
        
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_f, config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim, config.d_f)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.transformer.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x      

class EncoderBlock(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.d_f = config.d_f
        self.attention_norm = nn.LayerNorm(config.d_f, eps=1e-6)
        self.mulithead_attention = MultiheadAttention(config, vis)
        self.ffn_norm = nn.LayerNorm(config.d_f, eps=1e-6)
        self.ffn = MLP(config)
        
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.mulithead_attention(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.layers = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.d_f, eps=1e-6)
        
        for _ in range(config.transformer.num_layers):
            layer = EncoderBlock(config, vis)
            self.layers.append(copy.deepcopy(layer))
            
    def forward(self, embeddings):
        a_weight_layers = []
        for layer_block in self.layers:
            embeddings, a_weights = layer_block(embeddings)
            if self.vis: a_weight_layers.append(a_weights)
            
        encoded = self.encoder_norm(embeddings)
        return encoded, a_weight_layers
    
class Transformer(nn.Module):
    def __init__(self, config, input_channels, vis):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config, input_channels)
        self.encoder = Encoder(config, vis)
        self.rot_attenttion = RotatoryAttention(config)
        self.W_r = nn.Linear(config.d_f*4, config.d_f)
    
    def forward(self, x):
        embeddings, features = self.embedding(x)
        encoded, a_weights = self.encoder(embeddings)
        interslice = self.W_r(self.rot_attenttion(embeddings)).view(1,1,self.config.d_f)
        
        inter_encoded = encoded + interslice
        return encoded, a_weights, features, inter_encoded
    
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 padding=0, stride=1, use_batchnorm=True):
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None: x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.d_f, head_channels,
            kernel_size=3, padding=1, use_batchnorm=True
        )
        
        self.decoder_channels = config.decoder_channels
        self.in_channels = [head_channels] + list(self.decoder_channels[:-1])
        self.out_channels = self.decoder_channels
        
        if self.config.n_skip != 0:
            self.skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip): 
                self.skip_channels[3 - i] = 0
                
        else: self.skip_channels=[0,0,0,0]
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) \
            for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None

            x = decoder_block(x, skip=skip)
        return x
    
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, upsampling=1):
        
        conv2d = nn.Conv2d(in_channels, out_channels, 
                           kernel_size=kernel_size, 
                           padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class RotAttTransUNet(nn.Module):
    def __init__(self, config, num_classes, input_channels=1, vis=True):
        super().__init__()
        self.num_classes = num_classes
        self.transformer = Transformer(config, input_channels, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.num_classes,
            kernel_size=3,
        )
        
    def forward(self, x):
        encoded, a_weights, features, inter_encoded= self.transformer(x)
        decoded = self.decoder(inter_encoded, features)
        logits = self.segmentation_head(decoded)
        return logits