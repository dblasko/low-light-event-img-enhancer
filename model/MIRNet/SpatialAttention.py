import torch
import torch.nn as nn

from model.MIRNet.ChannelCompression import ChannelCompression

class SpatialAttention(nn.Module):
    """
    Reduces the input to 2 channel with the ChannelCompression module and applies a 2D convolution with 1 output channel.
    
    In: HxWxC
    Out: HxWxC (original channels are restored by multiplying the output with the original input)
    """
    
    def __init__(self):
        super().__init__()
        self.channel_compression = ChannelCompression()
        self.conv = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2)
    
    def forward(self, x):
        x_compressed = self.channel_compression(x) # HxWx2
        x_conv = self.conv(x_compressed) # HxWx1
        scaling_factor = torch.sigmoid(x_conv)
        return x * scaling_factor # HxWxC
        
