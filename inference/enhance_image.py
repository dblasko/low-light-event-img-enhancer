import argparse

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
Run this script to run model inference on a specified image and write the enhanced image to an output folder.
Usage: python inference/enhance_image.py -i <path_to_input_image> [-o <path_to_output_folder> -m <path_to_model>]
    or python inference/enhance_image.py --input_image_path <path_to_input_image> [--output_folder_path <path_to_output_folder> --model_path <path_to_model>]
If the output folder is not specified, the enhanced image is written to the directory the script is run from.
If the model path is not specified, the default model defined in MODEL_PATH is used.
"""

IMG_SIZE = 400
NUM_FEATURES = 64
MODEL_PATH = "model/weights/Mirnet_enhance_finetune-35-early-stopped_64x64.pth"  # f"model/weights/Mirnet_enhance{99}_64x64.pth"


def run_inference(input_image_path, output_folder_path, device, model_path=MODEL_PATH):
    model = MIRNet(num_features=NUM_FEATURES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        try:
            img = Image.open(input_image_path)
            img_tensor = T.Compose(
                [
                    T.Resize(IMG_SIZE),
                    T.ToTensor(),
                    T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                ]
            )(img).unsqueeze(0)
            img_tensor = img_tensor.to(device)

            output = model(img_tensor)
        except:
            print("Could not open image - verify the provided path.")
            return
        try:
            output_image = T.ToPILImage()(output.squeeze().cpu())
            out_path = (
                output_folder_path
                if output_folder_path[-1] == "/"
                else output_folder_path
                + "/"
                + input_image_path.split("/")[-1].split(".")[0]
                + "_enhanced.png"
            )
            output_image.save(out_path)
            print('-> Enhanced image saved to "' + out_path + '".')
        except:
            print("Error: Could not save image - verify the provided path.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image_path",
        "-i",
        help="Path to the input image to enhance.",
        required=True,
    )
    parser.add_argument(
        "--output_folder_path",
        "-o",
        help="Path to the output folder to save the enhanced image to the MODEL_PATH constant specified in the script.",
        default=".",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        help="Path to model weights to use. Defaults to the ",
        default=MODEL_PATH,
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"-> {device.type} device detected.")

    run_inference(
        args.input_image_path, args.output_folder_path, device, args.model_path
    )
