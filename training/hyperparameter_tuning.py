import sys, torch, wandb

sys.path.append(".")

from torch.utils.data import DataLoader
from model.MIRNet.model import MIRNet
from dataset_generation.PretrainingDataset import PretrainingDataset
from training.training_utils.CharbonnierLoss import CharbonnierLoss

from training.train import train, validate


def main():
    """Main function run for every hyperparameter combination."""
    # Setup and verify training configuration:
    with wandb.init() as run:
        config = run.config

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"-> {device.type} device detected.")

        # Prepare training objects:
        model = MIRNet(
            num_features=64,
            number_msrb=config.num_msrb,
        ).to(device)
        wandb.watch(model)
        criterion = CharbonnierLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-8,
            eps=1e-8,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.epochs, config.learning_rate * 5 / 10, verbose=True
        )

        train_dataset = PretrainingDataset(
            "data/pretraining/train/imgs",
            "data/pretraining/train/targets",
            img_size=128,
        )
        val_dataset = PretrainingDataset(
            "data/pretraining/val/imgs",
            "data/pretraining/val/targets",
            img_size=128,
            train=False,
        )
        test_dataset = PretrainingDataset(
            "data/pretraining/test/imgs",
            "data/pretraining/test/targets",
            img_size=128,
            train=False,
        )
        train_data = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        val_data = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_data = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Training loop:
        for epoch in range(config.epochs):
            epoch_loss, epoch_psnr = train(
                train_data,
                model,
                criterion,
                optimizer,
                epoch,
                device,
                False,
            )
            print(
                f"|| Sweep for LR {config.learning_rate}, num_msrb {config.num_msrb} || ***Epoch {epoch}***\n\tTraining loss: {epoch_loss} - Training PSNR: {epoch_psnr}"
            )

            # Logging to WANDB
            wandb.log(
                {
                    "Training Loss": epoch_loss,
                    "Training PSNR": epoch_psnr,
                    "Epoch": epoch,
                },
                step=epoch,
            )

            lr_scheduler.step()
            epoch_loss, epoch_psnr = validate(val_data, model, criterion, device)
            print(f"\tValidation loss: {epoch_loss} - Validation PSNR: {epoch_psnr}")
            wandb.log(
                {
                    "Validation Loss": epoch_loss,
                    "Validation PSNR": epoch_psnr,
                    "Epoch": epoch,
                },
                step=epoch,
            )


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "metric": {"name": "Validation PSNR", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [1e-4, 1e-3, 1e-5, 1e-6]},
            "num_msrb": {"values": [1, 2, 3]},
            "epochs": {"value": 15},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="low-light-event-img-enhancer")
    wandb.agent(sweep_id, main)
