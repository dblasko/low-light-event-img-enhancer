import unittest
import torch
import torch.nn as nn
import warnings
import random
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
from model.MIRNet.model import MIRNet
from training.train import train, validate
from training.training_utils.CharbonnierLoss import CharbonnierLoss
from dataset_generation.PretrainingDataset import PretrainingDataset

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


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        # We use a subset of the data to be able to run on the CI server
        self.batch_size = 2
        self.img_size = 128
        self.train_dataset = PretrainingDataset('tests/data_examples/pretraining/train/imgs', 'tests/data_examples/pretraining/train/targets', img_size=self.img_size)
        self.train_data = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        self.model = MIRNet(num_features=64, number_msrb=2, number_rrg=2) # smaller model for CI
        
        self.criterion = CharbonnierLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 3, 0.00001) 

    def test_training_loop_runs(self):
        """
        Ensure the training loop runs without raising errors.
        """
        device = torch.device("cpu")  # CI environment constraint
        self.model.to(device)
        
        initial_loss = None
        for epoch in range(3):
            epoch_loss, _ = train(self.train_data, self.model, self.criterion, self.optimizer, epoch, device)
            if initial_loss is None:
                initial_loss = epoch_loss
            else:
                self.assertLessEqual(epoch_loss, initial_loss, "Loss did not decrease or remained the same; something might be wrong in the training loop.")
                initial_loss = epoch_loss  

    def test_loss_decreases(self):
        """
        Ensure that the loss decreases over multiple epochs.
        """
        device = torch.device("cpu")
        self.model.to(device)

        losses = []
        for epoch in range(3):
            epoch_loss, _ = train(self.train_data, self.model, self.criterion, self.optimizer, epoch, device)
            losses.append(epoch_loss)

        for i in range(1, len(losses)):
            self.assertLess(losses[i], losses[i-1], "Loss did not decrease after an epoch; training might not be functioning correctly.")



if __name__ == '__main__':
    unittest.main()
