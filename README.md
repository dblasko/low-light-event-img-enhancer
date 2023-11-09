![Python](https://img.shields.io/badge/python-3.10-green.svg)
![Lint & test](https://github.com/dblasko/low-light-event-img-enhancer/actions/workflows/continuous-integration.yml/badge.svg)
[![HitCount](https://hits.dwyl.com/dblasko/low-light-event-img-enhancer.svg)](https://hits.dwyl.com/dblasko/low-light-event-img-enhancer)

# low-light-event-img-enhancer
Deep-learning-based low-light image enhancer specialized on restoring dark images from events (concerts, parties, clubs...).

## Applied Deep Learning course
The deliverables for *Assignment 1 - Initiate* are located in `/initiate.md`. Those for *Assignment 2 - Hacking* are located in `/hacking.md`.

# Usage
## Requirements
In a Python 3.10 environment, install the requirements with `pip install -r requirements.txt`.

## Training the model
Training is generally done with the `training/train.py` script. Before running it, the training run must be configured in `training/config.yaml`. The configuration file contains the following parameters:
***TODO***

Then, you can run the training script while pointing to your configuration file with `python training/train.py --config training/config.yaml`.
***TODO: document will be tracked in WANDB...***

### Pre-training
***TODO: document pre-training pass & commands/configs used, alternatively download weights here***

### Fine-tuning 
***TODO: document fine-tuning pass & commands/configs, alternatively download weights here***

## Running the model for inference 

## Running tests


## Generating the datasets
### Pre-training dataset


### Fine-tuning dataset

