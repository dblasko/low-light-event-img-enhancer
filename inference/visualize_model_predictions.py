import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import argparse
from torch.utils.data import DataLoader

sys.path.append(".")

from dataset_generation.PretrainingDataset import PretrainingDataset
from training.training_utils.CharbonnierLoss import CharbonnierLoss
from model.MIRNet.model import MIRNet
from training.train import validate

"""
Run this script to visualize & evaluate a trained model's results on a test dataset.
The script expects a positional argument with the folder that contains both "imgs" and "targets".
The constants below can be changed to visualize the results of different models.
The output image is saved in 'inference/results', and it is a grid where each row contains the original image, the inference result and the ground truth image.
"""

IMG_SIZE = 400
NUM_FEATURES = 64
MODEL_PATH = "model/weights/Mirnet_enhance_finetune-35-early-stopped_64x64.pth"  # f"model/weights/Mirnet_enhance{99}_64x64.pth"  #'model/weights/Mirnet_enhance_finetune-35-early-stopped_64x64.pth'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="Path to the folder containing the images")
    args = parser.parse_args()
    root_dir = args.image_folder

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"-> {device.type} device detected.")
    model = MIRNet(num_features=NUM_FEATURES).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        # Directory containing the images
        img_dir = f"{root_dir}/imgs"
        target_dir = f"{root_dir}/targets"
        img_files = [
            file for file in os.listdir(img_dir) if not file.startswith(".DS_Store")
        ]

        # List to store the original, predicted and ground truth images
        images = []
        filenames = []

        # Maximal image dimensions are computed to later pad images to the same size to draw a grid
        for img_file in img_files:
            img = Image.open(os.path.join(img_dir, img_file))
            img_tensor = T.Compose(
                [
                    T.Resize(IMG_SIZE),
                    T.ToTensor(),
                    T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                ]
            )(img).unsqueeze(0)
            images.append(img_tensor)

        max_height = max([img.shape[2] for img in images])
        max_width = max([img.shape[3] for img in images])
        images = []

        for img_file in img_files:
            # Load and preprocess the image
            img = Image.open(os.path.join(img_dir, img_file))
            img_tensor = T.Compose(
                [
                    T.Resize(IMG_SIZE),
                    T.ToTensor(),
                    T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                ]
            )(img).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            padded_img_tensor = T.Pad(
                (
                    (max_width - img_tensor.shape[3]) // 2,
                    (max_height - img_tensor.shape[2]) // 2,
                ),
                fill=0,
            )(img_tensor)

            # Run inference
            output = model(img_tensor)
            padded_output = T.Pad(
                (
                    (max_width - img_tensor.shape[3]) // 2,
                    (max_height - img_tensor.shape[2]) // 2,
                ),
                fill=0,
            )(output)

            # Load and preprocess the ground truth image
            target = Image.open(os.path.join(target_dir, img_file))
            target_tensor = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])(
                target
            ).unsqueeze(0)
            target_tensor = T.Pad(
                (
                    (max_width - target_tensor.shape[3]) // 2,
                    (max_height - target_tensor.shape[2]) // 2,
                ),
                fill=0,
            )(target_tensor)
            target_tensor = target_tensor.to(device)

            # Append the original, predicted and ground truth images to the list
            images.extend(
                [
                    padded_img_tensor.squeeze(),
                    padded_output.squeeze(),
                    target_tensor.squeeze(),
                ]
            )
            filenames.append(img_file)

        # Create a grid of images
        grid = make_grid(images, nrow=3)
        vutils.save_image(grid, open(f"inference/results/model_testing_grid.png", "wb"))
        # Read 'model_testing_grid.png' as a PIL image
        grid = Image.open("inference/results/model_testing_grid.png")

        # Draw filenames on the grid
        draw = ImageDraw.Draw(grid)
        font_cats = ImageFont.truetype("utils/assets/Roboto-Medium.ttf", 21)
        font_files = ImageFont.truetype("utils/assets/Roboto-Medium.ttf", 21)
        for i, filename in enumerate(filenames):
            draw.text(
                (0, i * max_height + max_height // 2),
                filename,
                fill="white",
                font=font_files,
            )

        for i in range(3):
            draw.text(
                (i * max_width + (max_width // 2) - 20, 0),
                ["Original", "Predicted", "Ground Truth"][i],
                fill="white",
                font=font_cats,
            )

        # Save the grid to disk
        grid.save("inference/results/model_testing_grid.png")

    img_dir = f"{root_dir}/imgs"
    target_dir = f"{root_dir}/targets"
    test_dataset = PretrainingDataset(img_dir, target_dir, img_size=128, train=False)
    test_data = DataLoader(test_dataset, batch_size=8, shuffle=False)
    criterion = CharbonnierLoss()
    test_loss, test_psnr = validate(test_data, model, criterion, device)
    print(
        f"***Performance on the dataset:***\n\tTest loss: {test_loss} - Test PSNR: {test_psnr}"
    )
