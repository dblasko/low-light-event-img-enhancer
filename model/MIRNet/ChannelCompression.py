import torch
import torch.nn as nn


class ChannelCompression(nn.Module):
    """
    Reduces the input to 2 channels by concatenating the global average pooling and global max pooling outputs.

    In: HxWxC
    Out: HxWx2
    """

    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )
