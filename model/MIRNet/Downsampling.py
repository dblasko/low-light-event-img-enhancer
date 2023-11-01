import torch
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np


class DownsamplingBlock(nn.Module):
    """
    Downsamples the input to halve the dimensions while doubling the channels through two parallel conv + antialiased downsampling branches.
    
    In: HxWxC
    Out: H/2xW/2x2C
    """
    
    def __init__(self, in_channels, bias=False):
        super().__init__()
        self.branch1 = nn.Sequential(  # 1x1 conv + PReLU -> 3x3 conv + PReLU -> AD -> 1x1 conv
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=bias), 
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias),
            nn.PReLU(),
            DownSample(channels=in_channels, filter_size=3, stride=2),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, padding=0, bias=bias)
        )
        self.branch2 = nn.Sequential(
            DownSample(channels=in_channels, filter_size=3, stride=2),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, padding=0, bias=bias)
        )
    
    def forward(self, x):
        return self.branch1(x) + self.branch2(x) # H/2xW/2x2C
        
        

class DownsamplingModule(nn.Module):
    """
    Downsampling module of the network composed of (scaling factor) DownsamplingBlocks.
    
    In: HxWxC
    Out: H/2^(scaling factor) x W/2^(scaling factor) x C^2(scaling factor)
    """
    
    def __init__(self, in_channels, scaling_factor, stride=2):
        super().__init__()
        self.scaling_factor = int(np.log2(scaling_factor))
        
        blocks = []
        for i in range(self.scaling_factor):
            blocks.append(DownsamplingBlock(in_channels))
            in_channels = int(in_channels * stride)
        self.blocks = nn.Sequential(*blocks)
            
    
    def forward(self, x):
        x = self.blocks(x)
        return x # H/2^(scaling factor) x W/2^(scaling factor) x C^2(scaling factor)



class DownSample(nn.Module):
    """
    Antialiased downsampling module using the blur-pooling method.
    
    From Adobe's implementation available here: https://github.com/yilundu/improved_contrastive_divergence/blob/master/downsample.py 
    """
    
    def __init__(self, pad_type = 'reflect', filter_size = 3, stride = 2, channels = None, pad_off = 0):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.pad_off = pad_off
        self.channels = channels
        self.pad_sizes = [int(1.0 * (filter_size - 1) / 2),
                          int(np.ceil(1.0 * (filter_size - 1) / 2)),
                          int(1.0 * (filter_size - 1) / 2),
                          int(np.ceil(1.0 * (filter_size - 1) / 2))]
    
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.off = int((self.stride - 1) / 2.0)
        
        if self.filter_size == 1:
            a = np.array([1.0])
        elif self.filter_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filter_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filter_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filter_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filter_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filter_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
            
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        
    def forward(self, x):
        
        if self.filter_size == 1:
            if self.pad_off == 0:
                return x[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(x)[:, :, ::self.stride, ::self.stride]
        
        else:
            return fun.conv2d(self.pad(x), self.filt, stride = self.stride, groups = x.shape[1])
        

def get_pad_layer(pad_type):
    
    if pad_type == 'reflect':
        pad_layer = nn.ReflectionPad2d
    elif pad_type == 'replication':
        pad_layer = nn.ReplicationPad2d
    else:
        print('Pad Type [%s] not recognized' % pad_type)
    
    return pad_layer
