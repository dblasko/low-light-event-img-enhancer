import unittest
import torch

import sys
sys.path.append(".")
from model.MIRNet.model import MIRNet

class TestMIRNetModel(unittest.TestCase):
    def test_forward_pass(self):
        """
        The forward pass of the MIRNet should output a tensor of the same dimension as the input.
        """
        input_tensor = torch.rand(1, 3, 256, 256)  # mock image tensor
        
        model = MIRNet()
        output = model(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)

if __name__ == '__main__':
    unittest.main()
