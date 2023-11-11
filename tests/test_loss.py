import unittest
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append(".")

from training.training_utils.CharbonnierLoss import CharbonnierLoss


class TestLoss(unittest.TestCase):
    """
    Tests associated to the loss function.
    """

    def test_zero_loss(self):
        """
        Loss should be zero when prediction equals target
        """
        criterion = CharbonnierLoss()
        input = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0, 0.0])
        loss = criterion(input, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_positive_loss(self):
        """
        Loss should be positive when prediction does not equal target.
        """
        criterion = CharbonnierLoss()
        input = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0, 0.0])
        loss = criterion(input, target)
        self.assertGreater(loss.item(), 0.0)

    def test_epsilon_effect(self):
        """
        Smaller epsilon should result in a loss closer to L1.
        """
        small_epsilon_criterion = CharbonnierLoss(epsilon=1e-12)
        large_epsilon_criterion = CharbonnierLoss(epsilon=1e-3)
        input = torch.tensor([1.0, -1.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0])
        small_epsilon_loss = small_epsilon_criterion(input, target)
        large_epsilon_loss = large_epsilon_criterion(input, target)
        self.assertLess(small_epsilon_loss.item(), large_epsilon_loss.item())

    def test_backward(self):
        """
        The function should be differentiable and the backward pass should run.
        """
        criterion = CharbonnierLoss()
        input = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0, 0.0])
        loss = criterion(input, target)
        loss.backward()
        self.assertIsNotNone(input.grad)
