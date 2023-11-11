import torch
import torch.nn as nn


class SelectiveKernelFeatureFusion(nn.Module):
    """
    Merges outputs of the three different resolutions through self-attention.

    All three inputs are summed -> global average pooling -> downscaling -> the signal is passed through 3 different convs to have three descriptors,
    softmax is applied to each descriptor to get 3 attention activations used to recalibrate the three input feature maps.
    """

    def __init__(self, in_channels, reduction_ratio, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        conv_out_channels = max(int(in_channels / reduction_ratio), 4)
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels, conv_out_channels, kernel_size=1, padding=0, bias=bias
            ),
            nn.PReLU(),
        )

        self.attention_convs = nn.ModuleList([])
        for i in range(3):
            self.attention_convs.append(
                nn.Conv2d(
                    conv_out_channels, in_channels, kernel_size=1, stride=1, bias=bias
                )
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x[0].shape[0]
        n_features = x[0].shape[1]

        x = torch.cat(
            x, dim=1
        )  # the three outputs of diff. res. are concatenated along the channel dimension
        x = x.view(
            batch_size, 3, n_features, x.shape[2], x.shape[3]
        )  # batch_size x 3 x n_features x H x W

        z = torch.sum(x, dim=1)  # batch_size x n_features x H x W
        z = self.avg_pool(z)  # batch_size x n_features x 1 x 1
        z = self.convolution(z)  # batch_size x n_features/8 x 1 x 1

        attention_activations = [
            atn(z) for atn in self.attention_convs
        ]  # 3 x (batch_size x n_features x 1 x 1)
        attention_activations = torch.cat(
            attention_activations, dim=1
        )  # batch_size x 3*n_features x 1 x 1
        attention_activations = attention_activations.view(
            batch_size, 3, n_features, 1, 1
        )  # batch_size x 3 x n_features x 1 x 1

        attention_activations = self.softmax(
            attention_activations
        )  # batch_size x 3 x n_features x 1 x 1

        return torch.sum(
            x * attention_activations, dim=1
        )  # batch_size x n_features x H x W (the three feature maps are recalibrated and summed
