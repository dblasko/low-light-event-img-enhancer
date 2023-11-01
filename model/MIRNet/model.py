import torch
import torch.nn as nn

from model.MIRNet.ResidualRecurrentGroup import ResidualRecurrentGroup

class MIRNet(nn.Module):
    """
    Low-level features are extracted through convolution and passed to n residual recurrent groups that operate at different resolutions.
    Their output is added to the input image for restoration.
    
    Please refer to the documentation of the different blocks of the model in this folder for detailed explanations.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=64, kernel_size=3, stride=2, number_msrb=2, number_rrg=3, height=3, width=2, bias=False):
        super().__init__()
        self.conv_start = nn.Conv2d(in_channels, num_features, kernel_size, padding=1, bias=bias)
        msrb_blocks = [ResidualRecurrentGroup(num_features, number_msrb, height, width, stride, bias) for _ in range(number_rrg)]
        self.msrb_blocks = nn.Sequential(*msrb_blocks)
        self.conv_end = nn.Conv2d(num_features, out_channels, kernel_size, padding=1, bias=bias)
    
    def forward(self, x):
        output = self.conv_start(x)
        output = self.msrb_blocks(output)
        output = self.conv_end(output)
        return x + output # restored image, HxWxC

