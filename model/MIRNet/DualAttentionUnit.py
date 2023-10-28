import torch
import torch.nn as nn
from model.MIRNet.ChannelAttention import ChannelAttention

from model.MIRNet.SpatialAttention import SpatialAttention


class DualAttentionUnit(nn.Module):
    """
    Combines the ChannelAttention and SpatialAttention modules.
    (conv, PReLU, conv -> concat. SA & CA output -> conv -> skip connection from input)
    
    In: HxWxC
    Out: HxWxC (original channels are restored by multiplying the output with the original input)
    """
    
    def __init__(self, in_channels, kernel_size=3, reduction_ratio=8, bias=False):
        super().__init__()
        self.initial_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, bias=bias)
        )
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, bias)
        self.spatial_attention = SpatialAttention()
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        initial_convs = self.initial_convs(x) # HxWxC
        channel_attention = self.channel_attention(initial_convs) # HxWxC
        spatial_attention = self.spatial_attention(initial_convs) # HxWxC
        attention = torch.cat((spatial_attention, channel_attention), dim=1) # HxWx2C
        block_output = self.final_conv(attention) # HxWxC - the 1x1 conv. restores the C channels for the skip connection
        return x + block_output # the addition is the skip connection from input
        
