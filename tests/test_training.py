import unittest
import torch
import torch.nn as nn
import warnings

import sys
sys.path.append(".")
from model.MIRNet.model import MIRNet

class TestOptimizer(unittest.TestCase):
    def test_optimizer_updates_weights(self):
        """
        The optimizer should update the model weights
        """
        model = MIRNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        initial_weights = torch.clone(model.conv_start.weight.data)
        
        inputs = torch.randn(1, 3, 64, 64)
        targets = torch.randn(1, 3, 64, 64)
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        updated_weights = model.conv_start.weight.data
        self.assertFalse(torch.equal(initial_weights, updated_weights),
                         "The optimizer did not update the weights")
