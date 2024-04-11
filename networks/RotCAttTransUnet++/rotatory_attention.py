import torch
import torch.nn as nn

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