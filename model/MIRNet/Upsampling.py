import torch
import torch.nn as nn


class UpsamplingBlock(nn.Module):
    """
    Upsamples the input to double the dimensions while halving the channels through two parallel conv + bilinear upsampling branches.
    
    In: HxWxC
    Out: 2Hx2WxC/2
    """
    
    def __init__(self, in_channels, bias=False):
        super().__init__()
        self.branch1 = nn.Sequential(  # 1x1 conv + PReLU -> 3x3 conv + PReLU -> BU -> 1x1 conv
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=bias), 
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0, bias=bias)
        )
        self.branch2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0, bias=bias)
        )
    
    def forward(self, x):
        return self.branch1(x) + self.branch2(x) # 2Hx2WxC/2
        
        

class UpsamplingModule(nn.Module):
    """
    Upsampling module of the network composed of (scaling factor) UpsamplingBlocks.
    
    In: HxWxC
    Out: 2^(scaling factor)H x 2^(scaling factor)W x C/2^(scaling factor)
    """
    
    def __init__(self, in_channels, scaling_factor, stride=2):
        super().__init__()
        self.scaling_factor = scaling_factor
        
        blocks = []
        for i in range(scaling_factor):
            blocks.append(UpsamplingBlock(in_channels))
            in_channels = int(in_channels // 2)
        self.blocks = nn.Sequential(*blocks)
            
    
    def forward(self, x):
        return self.blocks(x) # 2^(scaling factor)H x 2^(scaling factor)W x C/2^(scaling factor)


