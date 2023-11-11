import unittest
import torch
import torch.nn as nn
import warnings
import math
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
        self.assertFalse(
            torch.equal(initial_weights, updated_weights),
            "The optimizer did not update the weights",
        )


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        # We use a subset of the data to be able to run on the CI server
        self.batch_size = 2
        self.img_size = 128
        self.train_dataset = PretrainingDataset(
            "tests/data_examples/pretraining/train/imgs",
            "tests/data_examples/pretraining/train/targets",
            img_size=self.img_size,
        )
        self.train_data = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

        self.model = MIRNet(
            num_features=64, number_msrb=2, number_rrg=2
        )  # smaller model for CI

        self.criterion = CharbonnierLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 3, 0.00001
        )

    def test_training_loop_runs(self):
        """
        Ensure the training loop runs without raising errors.
        """
        device = torch.device("cpu")  # CI environment constraint
        self.model.to(device)

        initial_loss = None
        for epoch in range(3):
            epoch_loss, _ = train(
                self.train_data,
                self.model,
                self.criterion,
                self.optimizer,
                epoch,
                device,
            )
            if initial_loss is None:
                initial_loss = epoch_loss
            else:
                self.assertLessEqual(
                    epoch_loss,
                    initial_loss,
                    "Loss did not decrease or remained the same; something might be wrong in the training loop.",
                )
                initial_loss = epoch_loss

    def test_loss_decreases_no_nan_or_inf_params(self):
        """
        Ensure that the loss decreases over multiple epochs, and that no weights of the model become NaN or Inf.
        Both are done in a simple test function to limit the number of times the training loop has to be run on the free CI server. This is more of an integration test.
        """
        device = torch.device("cpu")
        self.model.to(device)

        losses = []
        for epoch in range(3):
            epoch_loss, _ = train(
                self.train_data,
                self.model,
                self.criterion,
                self.optimizer,
                epoch,
                device,
            )
            losses.append(epoch_loss)
            for param in self.model.parameters():
                self.assertFalse(
                    torch.isnan(param).any(),
                    f"NaNs found in model parameters after {epoch+1} epochs",
                )
                self.assertFalse(
                    torch.isinf(param).any(),
                    f"Infs found in model parameters after {epoch+1} epochs",
                )

        for i in range(1, len(losses)):
            self.assertLess(
                losses[i],
                losses[i - 1],
                "Loss did not decrease after an epoch; training might not be functioning correctly.",
            )


class TestValidationLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        # We use a subset of the data to be able to run on the CI server
        self.batch_size = 2
        self.img_size = 128
        self.train_dataset = PretrainingDataset(
            "tests/data_examples/pretraining/test/imgs",
            "tests/data_examples/pretraining/test/targets",
            img_size=self.img_size,
        )
        self.train_data = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

        self.model = MIRNet(
            num_features=64, number_msrb=2, number_rrg=2
        )  # smaller model for CI

        self.criterion = CharbonnierLoss()

    def test_validation_integration(self):
        """
        Integration test for the validation loop.
        Ensures that the validation loop does not modify the model's weights,
            that the loss and PSNR calculation does not lead to error or create NaN/Inf values,
            and that the PSNR calculation creates realistic values.
        """
        device = torch.device("cpu")
        self.model.to(device)

        initial_state_dict = self.model.state_dict()
        try:
            validation_loss, validation_psnr = validate(
                self.train_data, self.model, self.criterion, device
            )
        except Exception as e:
            self.fail(f"Validation loss computation raised an exception: {e}")
        post_validation_state_dict = self.model.state_dict()

        self.assertFalse(math.isnan(float(validation_loss)), "Validation loss is NaN")
        self.assertFalse(math.isinf(validation_loss), "Validation loss is Inf")
        self.assertFalse(math.isnan(validation_psnr), "Validation psnr is NaN")
        self.assertFalse(math.isinf(validation_psnr), "Validation psnr is Inf")
        self.assertGreater(validation_psnr, 0, "Validation PSNR should be positive")
        self.assertLess(
            validation_psnr, 100, "Validation PSNR should be less than 100 in practice"
        )

        for param_before, param_after in zip(
            initial_state_dict.values(), post_validation_state_dict.values()
        ):
            self.assertTrue(
                torch.equal(param_before, param_after),
                "Model weights changed during validation",
            )


if __name__ == "__main__":
    unittest.main()
