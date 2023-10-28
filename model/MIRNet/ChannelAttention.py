import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Squeezes down the input to 1x1xC, applies the excitation operation and restores the C channels through a 1x1 convolution. 
    
    In: HxWxC
    Out: HxWxC (original channels are restored by multiplying the output with the original input)
    """
    
    def __init__(self, in_channels, reduction_ratio=8, bias=True):
        super().__init__()
        self.squeezing = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, padding=0, bias=bias),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        squeezed_x = self.squeeze(x) # 1x1xC
        excitation = self.excitation(squeezed_x) # 1x1x(C/r)
        return excitation * x # HxWxC restored through the mult. with the original input
        
