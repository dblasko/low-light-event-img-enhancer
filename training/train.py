# TODO: documentation, then prepare finetuning -> try fine-tune on larger resolutions on colab
# TODO: (! huggan night2day not darkened - use some LoL) + all data augs from albumentations etc?
# TODO: sanity tests + GitHub CI?
# TODO: setup colab or Gradient?
# TODO: verify batch psnr makes sense
# TODO: design pretrain run + document it (config files, what run, how dataset was created, wandb + resuming... add release of weights.)
# TODO: after trained version, add tests & clean up -> prepare fine-tune dataset, make fine-tuned version too -> eval, compare impact, package dataset - check expectations and the rest, if time compare other model from lib that I trained (cf onglet WANDB)
# TODO: then prepare fine tune data, design fine tune run, test non-fine tuned model on the data too...
# TODO: other model from lib OR TRY SCALING UP THE MODEL???

import sys, os, argparse, yaml, torch, wandb
sys.path.append(".")

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.MIRNet.model import MIRNet
from dataset_generation.PretrainingDataset import PretrainingDataset
from training.training_utils.CharbonnierLoss import CharbonnierLoss
from training.training_utils.MixUp import MixUp
from training.training_utils.Psnr import batch_psnr


def train(train_data, model, criterion, optimizer, epoch, device, enable_mixup=True):
    losses, psnrs = [], []
    batches = tqdm(train_data, position=0, leave=True)
    model.train()
    mixup = MixUp()
    
    for idx, (img, target) in enumerate(batches):
        assert not np.any(np.isnan(img.numpy())) # TODO: try from epoch 0, if persists check if None in loss list, print the epoch loss list....
        img = img.to(device)
        target = target.to(device)
        
        if epoch > 5 and enable_mixup:
            target, img = mixup.augment(target, img)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        losses.append(loss.item())
        psnrs.append(batch_psnr(target, output, 1))
        loss.backward()
        optimizer.step()
    
    model.eval()
    epoch_psnr = sum(psnrs) / len(psnrs)
    epoch_loss = sum(losses) / len(losses)
    torch.cuda.empty_cache()
    return epoch_loss, epoch_psnr
    

def validate(val_data, model, criterion, device): 
    losses, psnrs = [], []
    batches = tqdm(val_data, position=0, leave=True)
    model.eval()
    
    for idx, (img, target) in enumerate(batches):
        img = img.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, target)
            losses.append(loss.item())
            psnrs.append(batch_psnr(target, output, 1))
        
    epoch_psnr = sum(psnrs) / len(psnrs)
    epoch_loss = sum(losses) / len(losses)
    return epoch_loss, epoch_psnr
    

if __name__ == "__main__":
    # Setup and verify training configuration:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the training run config file", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    required_keys = ['epochs', 'train_dataset_path', 'val_dataset_path', 'test_dataset_path', 'batch_size', 'learning_rate', 'early_stopping', 'image_size']
    if not all(key in config for key in required_keys):
        raise ValueError(f"Config file must contain properties {','.join(required_keys)}.")
    
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"-> {device.type} device detected.")
    
    wandb.init(project="low-light-event-img-enhancer")#, id='nanygdnw', resumme='must')
    wandb.config.update(config)

    # Prepare training objects:
    model = MIRNet(num_features=config['num_features'] if 'num_features' in config else 64).to(device)
    wandb.watch(model)
    criterion = CharbonnierLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9,0.999), weight_decay=1e-8, eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], 5e-5, verbose=True)
    
    train_dataset = PretrainingDataset(config['train_dataset_path'] + '/imgs', config['train_dataset_path'] + '/targets', img_size=config['image_size'])
    val_dataset = PretrainingDataset(config['val_dataset_path'] + '/imgs', config['val_dataset_path'] + '/targets', img_size=config['image_size'], train=False)
    test_dataset = PretrainingDataset(config['test_dataset_path'] + '/imgs', config['test_dataset_path'] + '/targets', img_size=config['image_size'], train=False)
    train_data = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'] if 'workers' in config else 8, pin_memory=True)
    val_data = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Restoring training from checkpoint if available and 'resume_epoch' specified in config:
    start_epoch = 0
    if 'resume_epoch' in config and os.path.isfile(f'model/weights/Mirnet_enhance{config["resume_epoch"]}.pth'):
        # resume training
        checkpoint = torch.load(f'model/weights/Mirnet_enhance{config["resume_epoch"]}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"-> Resuming training from epoch {start_epoch}")
    
    # Training loop:
    best_val_loss = float('inf')
    best_last_6_val_loss = float('inf')  # Best validation loss over the last 5 epochs
    epochs_no_improve = 0
    max_epochs_stop = 6  # Number of epochs to stop training if no improvement
    last_6_checkpoints = [] 

    for epoch in range(start_epoch, config['epochs']):
        epoch_loss, epoch_psnr = train(train_data, model, criterion, optimizer, epoch, device)
        print(f"***Epoch {epoch}***\n\tTraining loss: {epoch_loss} - Training PSNR: {epoch_psnr}")
        wandb.log({"Training Loss": epoch_loss, "Training PSNR": epoch_psnr})
        lr_scheduler.step()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
        }
        checkpoint_path = f'model/weights/Mirnet_enhance{epoch}.pth'
        torch.save(checkpoint, open(checkpoint_path, 'wb'))
        # wandb.save(checkpoint_path)
        epoch_loss, epoch_psnr = validate(val_data, model, criterion, device)
        print(f"\tValidation loss: {epoch_loss} - Validation PSNR: {epoch_psnr}")
        wandb.log({"Validation Loss": epoch_loss, "Validation PSNR": epoch_psnr})
        
        # Early stopping
        if config['early_stopping']:
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= max_epochs_stop:
                    print(f'-> Early stopping, no val improvement for 6 epochs! Best validation loss: {best_val_loss}')
                    break

        # Remove old checkpoints if performance improved
        last_6_checkpoints.append(checkpoint_path)
        if len(last_6_checkpoints) > 6:
            old_checkpoint = last_6_checkpoints.pop(0)
            if epoch_loss < best_last_6_val_loss:
                best_last_6_val_loss = epoch_loss
                for old_checkpoint in last_6_checkpoints:
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                last_6_checkpoints = []
                
    # Testing:
    test_loss, test_psnr = validate(test_data, model, criterion, device)
    print(f"***Testing***\n\tTest loss: {test_loss} - Test PSNR: {test_psnr}")
    wandb.log({"Test Loss": test_loss, "Test PSNR": test_psnr})
    