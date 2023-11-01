import torch
import numpy as np
import torch.nn as nn
from model.MIRNet.Downsampling import DownsamplingModule

from model.MIRNet.DualAttentionUnit import DualAttentionUnit
from model.MIRNet.SelectiveKernelFeatureFusion import SelectiveKernelFeatureFusion
from model.MIRNet.Upsampling import UpsamplingModule


class MultiScaleResidualBlock(nn.Module):
    """
    Three parallel convolutional streams at different resolutions. Information is exchanged through residual connexions. 
    """
    
    def __init__(self, num_features, height, width, stride, bias):
        super().__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        features = [int((stride**i) * num_features) for i in range(height)]
        scale = [2**i for i in range(1, height)]
        
        self.dual_attention_units = nn.ModuleList([nn.ModuleList([DualAttentionUnit(int(num_features*stride**i))] * width) for i in range(height)])
        self.last_up = nn.ModuleDict()
        for i in range(1, height):
            self.last_up.update({f'{i}': UpsamplingModule(in_channels = int(num_features*stride**i), scaling_factor = 2**i, stride = stride)})
            
        self.down = nn.ModuleDict()
        i = 0
        scale.reverse()
        for f in features:
            for s in scale[i:]:
                self.down.update({f'{f}_{s}': DownsamplingModule(f, s, stride)})
            i+=1
            
        self.up = nn.ModuleDict()
        i = 0
        features.reverse()
        for f in features:
            for s in scale[i:]:
                self.up.update({f'{f}_{s}': UpsamplingModule(f, s, stride)})
            i+=1
            
        self.out_conv = nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1, bias = bias)
        self.skff_blocks = nn.ModuleList([SelectiveKernelFeatureFusion(num_features*stride**i, height) for i in range(height)])
        
    
    def forward(self, x):
        inp = x.clone()
        out = []
        
        for j in range(self.height):
            if j==0:
                inp = self.dual_attention_units[j][0](inp)
            else:
                inp = self.dual_attention_units[j][0](self.down[f'{inp.size(1)}_{2}'](inp))
            out.append(inp)
        
        for i in range(1, self.width):
            
            if True:
                temp = []
                for j in range(self.height):
                    TENSOR = []
                    nfeats = (2**j)*self.num_features
                    for k in range(self.height):
                        TENSOR.append(self.select_up_down(out[k], j, k))
                    
                    skff = self.skff_blocks[j](TENSOR)
                    temp.append(skff)
                    
            else:
                
                temp = out
                
            for j in range(self.height):
                
                out[j] = self.dual_attention_units[j][i](temp[j])
                
        output = []
        for k in range(self.height):
            
            output.append(self.select_last_up(out[k], k))
            
        output = self.skff_blocks[0](output)
        output = self.out_conv(output)
        output = output + x
        return output

    def select_up_down(self, tensor, j, k):    
        if j == k:
            return tensor
        else:
            diff = 2 ** np.abs(j-k)
            if j < k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)
            
    def select_last_up(self, tensor, k):
        if k == 0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)
