import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import os

from model.MIRNet.model import MIRNet

"""
Run this script to visualize a trained model's results on the test dataset. The constants below can be changed to visualize the results of different models.
The output image is saved in 'inference/results', and it is a grid where each row contains the original image, the inference result and the ground truth image.
"""

IMG_SIZE = 400
NUM_FEATURES = 64
MODEL_PATH = f'model/weights/Mirnet_enhance{99}_64x64.pth'


if __name__ == '__main__':
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
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        # Directory containing the images
        img_dir = 'data/pretraining/test/imgs'
        target_dir = 'data/pretraining/test/targets'
        img_files = [file for file in os.listdir(img_dir) if not file.startswith('.DS_Store')]

        # List to store the original, predicted and ground truth images
        images = []
        filenames = []

        for img_file in img_files:
            # Load and preprocess the image
            img = Image.open(os.path.join(img_dir, img_file))
            img_tensor = T.Compose([T.Resize(IMG_SIZE),  T.ToTensor(), T.Normalize([0.0,0.0,0.0], [1.0,1.0,1.0])])(img).unsqueeze(0)
            img_tensor = img_tensor.to(device)

            # Run inference
            output = model(img_tensor)

            # Load and preprocess the ground truth image
            target = Image.open(os.path.join(target_dir, img_file))
            target_tensor = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])(target).unsqueeze(0)
            target_tensor = target_tensor.to(device)

            # Append the original, predicted and ground truth images to the list
            images.extend([img_tensor.squeeze(), output.squeeze(), target_tensor.squeeze()])
            filenames.append(img_file)

        # Create a grid of images
        grid = make_grid(images, nrow=3)
        vutils.save_image(grid, open(f'inference/results/model_testing_grid.png', 'wb'))
        # Read 'model_testing_grid.png' as a PIL image
        grid = Image.open('model_testing_grid.png')

        # Draw filenames on the grid
        draw = ImageDraw.Draw(grid)
        font = ImageFont.truetype("utils/assets/Roboto-Medium.ttf", 18)
        for i, filename in enumerate(filenames):
            draw.text((0, i * IMG_SIZE + IMG_SIZE // 2), filename, fill='white', font=font)
        
        for i in range(3):
            draw.text((i * 1.5 * IMG_SIZE + (1.5 * IMG_SIZE // 2), 0), ['Original', 'Predicted', 'Ground Truth'][i], fill='white', font=font)

        # Save the grid to disk
        grid.save('model_testing_grid.png')

        #fixed_ip_train = T.Compose([T.Resize(IMG_SIZE),  T.ToTensor(), T.Normalize([0.0,0.0,0.0], [1.0,1.0,1.0])])(Image.open('data/pretraining/test/imgs/493.png')).unsqueeze(0)
        #fixed_ip_train = fixed_ip_train.to(device)
        #output = model(fixed_ip_train)
        #vutils.save_image(images[7], open(f'inference/results/predicted_image.png', 'wb'))

