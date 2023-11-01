from datasets import load_dataset
import os
from PIL import Image
import numpy as np
import torch

if __name__ == "__main__":
    np.random.seed(42)
    
    for split in ['train', 'test', 'val']:
        os.makedirs(f'data/pretraining/{split}', exist_ok=True)
        os.makedirs(f'data/pretraining/{split}/imgs', exist_ok=True)
        os.makedirs(f'data/pretraining/{split}/targets', exist_ok=True)

    dataset = load_dataset("huggan/night2day")
    # dataset = load_dataset("geekyrakshit/LoL-Dataset")
    
    train_size = int(0.85 * len(dataset['train']))
    val_size = int(0.10 * len(dataset['train']))
    test_size = len(dataset['train']) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset['train'], [train_size, val_size, test_size])
    
    for dataset, dataset_name in [(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')]:
        for i in range(len(dataset)):
            imageA = dataset[i]['imageA']
            imageB = dataset[i]['imageB']
            
            imageA.save(f'data/pretraining/{dataset_name}/imgs/train_img_{i}.png')
            imageB.save(f'data/pretraining/{dataset_name}/targets/train_target_{i}.png')
