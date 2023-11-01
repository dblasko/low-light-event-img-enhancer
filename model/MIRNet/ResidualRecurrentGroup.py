import torch
import torch.nn as nn

from model.MIRNet.MultiScaleResidualBlock import MultiScaleResidualBlock

class ResidualRecurrentGroup(nn.Module):
    """
    Group of multi-scale residual blocks followed by a convolutional layer. The output is what is added to the input image for restoration.
    """
    
    def __init__(self, num_features, number_msrb_blocks, height, width, stride, bias=False):
        super().__init__()
        blocks = [MultiScaleResidualBlock(num_features, height, width, stride, bias) for _ in range(number_msrb_blocks)]
        blocks.append(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=1, bias=bias)
        )
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        output = self.blocks(x)
        return x + output # restored image, HxWxC
