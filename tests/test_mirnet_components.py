import unittest
import torch

import sys

sys.path.append(".")
from model.MIRNet.model import MIRNet
import model.MIRNet as MIRNetComponents


class TestMIRNetModel(unittest.TestCase):
    """
    Tensor-dimension sanity checks for the MIRNet model and its components.
    """

    def test_forward_pass(self):
        """
        The forward pass of the MIRNet should output a tensor of the same dimension as the input.
        """
        input_tensor = torch.rand(1, 3, 256, 256)  # mock image tensor

        model = MIRNet()
        output = model(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)

    def test_channel_attention(self):
        """
        The channel attention block should output a tensor of the same dimension as the input.
        """
        input_tensor = torch.rand(1, 64, 256, 256)
        channelAttention = MIRNetComponents.ChannelAttention.ChannelAttention(64)
        output = channelAttention(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_channel_compression(self):
        """
        The channel compression block should output a tensor of the same dimension as the input, except for the channels that should always be reduced to 2.
        """
        input_tensor = torch.rand(1, 64, 256, 256)
        channelCompression = MIRNetComponents.ChannelCompression.ChannelCompression()
        output = channelCompression(input_tensor)
        self.assertEqual(output.shape[1], 2)
        self.assertEqual(output.shape[0], input_tensor.shape[0])
        self.assertEqual(output.shape[2], input_tensor.shape[2])
        self.assertEqual(output.shape[3], input_tensor.shape[3])

    def test_downsampling(self):
        """
        The downsampling module divides the height & width by the scaling factor, and multiplies the channels by it.
        """
        input_tensor = torch.rand(1, 64, 256, 256)
        scaling_factor = 8
        downsamplingBlock = MIRNetComponents.Downsampling.DownsamplingModule(
            64, scaling_factor
        )
        output = downsamplingBlock(input_tensor)
        self.assertEqual(output.shape[0], input_tensor.shape[0])
        self.assertEqual(output.shape[1], input_tensor.shape[1] * scaling_factor)
        self.assertEqual(output.shape[2], input_tensor.shape[2] // scaling_factor)
        self.assertEqual(output.shape[3], input_tensor.shape[3] // scaling_factor)

    def test_upsampling(self):
        """
        The upsampling module multiples the height & width by the scaling factor, and divides the channels by it.
        """
        input_tensor = torch.rand(1, 64, 256, 256)
        scaling_factor = 8
        upsamplingBlock = MIRNetComponents.Upsampling.UpsamplingModule(
            64, scaling_factor
        )
        output = upsamplingBlock(input_tensor)
        self.assertEqual(output.shape[0], input_tensor.shape[0])
        self.assertEqual(output.shape[1], input_tensor.shape[1] // scaling_factor)
        self.assertEqual(output.shape[2], input_tensor.shape[2] * scaling_factor)
        self.assertEqual(output.shape[3], input_tensor.shape[3] * scaling_factor)

    def test_dual_attention_unit(self):
        """
        The dual attention unit should output a tensor of the same dimension as the input.
        """
        input_tensor = torch.rand(1, 64, 256, 256)
        dau = MIRNetComponents.DualAttentionUnit.DualAttentionUnit(64)
        output = dau(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_residual_recurrent_group(self):
        """
        The residual recurrent group should output a tensor of the same dimension as the input.
        """
        input_tensor = torch.rand(1, 64, 256, 256)
        rrg = MIRNetComponents.ResidualRecurrentGroup.ResidualRecurrentGroup(
            64, 2, 3, 2, 2, False
        )
        output = rrg(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_spatial_attention(self):
        """
        The spatial attention block should output a tensor of the same dimension as the input.
        """
        input_tensor = torch.rand(1, 64, 256, 256)
        spatialAttention = MIRNetComponents.SpatialAttention.SpatialAttention()
        output = spatialAttention(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)


if __name__ == "__main__":
    unittest.main()
