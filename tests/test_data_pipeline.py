import unittest
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

from dataset_generation.PretrainingDataset import PretrainingDataset

class TestDataPipeline(unittest.TestCase):
    """
    Tests associated to the entire data pipeline for model training and inference (loaders, transforms, etc.)
    """
    
    def test_data_loader_train(self):
        """
        The training data loader should perform image_size / 2 x image_size / 2 crops in the image.
        """
        batch_size = 2
        img_size = 128
        train_dataset = PretrainingDataset('tests/data_examples/pretraining/train/imgs', 'tests/data_examples/pretraining/train/targets', img_size=img_size)
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        images, _ = next(iter(train_data))
        self.assertEqual(images.size(), (batch_size, 3, img_size // 2, img_size // 2))
        

    def test_data_loader_val(self):
        """
        The validation/test data loader does not perform any cropping - the smallest dimension of the image is resized to image_size.
        """
        batch_size = 2
        img_size = 128
        train_dataset = PretrainingDataset('tests/data_examples/pretraining/test/imgs', 'tests/data_examples/pretraining/test/targets', img_size=img_size, train=False)
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        images, _ = next(iter(train_data))
        self.assertEqual(images.size()[0], batch_size)
        self.assertEqual(images.size()[1], 3)
        self.assertTrue(images.size()[2] == img_size or images.size()[3] == img_size)
