import torch
import torch.nn as nn
import ml_collections


class Embedding(nn.Module):
    def __init__(self, config, input_channels):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = (config.width // self.patch_size) * (config.height // self.patch_size)
        self.patch_embeddings = nn.Conv2d (
            in_channels=input_channels,
            out_channels=config.d_f,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, config.d_f))
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
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
        
    
class RotAttTransUNet(nn.Module):
    def __init__(self, num_classes, config, input_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = Embedding(config, input_channels)
        self.rotatory_attention = RotatoryAttention(config)
        
    def forward(self, x):
        x = self.embedding(x)
        r = self.rotatory_attention(x)
        
        
    
def get_config():
    config = ml_collections.ConfigDict()
    config.width=128
    config.height=128
    config.patch_size = 16
    config.d_f = 768
    config.d_k = 768
    config.d_v = 768
    
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 8
    config.activation = 'softmax'
    return config

config = get_config()
input = torch.randn((3,1,config.width,config.height))
model = RotAttTransUNet(num_classes=8, config=config, input_channels=1)
output = model(input)