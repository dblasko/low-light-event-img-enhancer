import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """
    Implementation of Charbonnier loss for image restoration.
    Based on J.T. Barron, 2019: https://arxiv.org/pdf/1701.03077.pdf 
    """
    
    def __init__(self, epsilon = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predicted, target):
        return torch.mean(torch.sqrt((predicted - target).pow(2) + self.epsilon**2))
    
