from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import io
import os
import sys

sys.path.append(".")
from model.MIRNet.model import MIRNet

"""
Example use of the API with curl:
    - Inference on a single image: curl -X POST -F "image=@./img_102.png" http://localhost:5000/enhance --output ./enhanced_image.png
    - Inference on a batch of images: curl -X POST -F "images=@./image1.jpg" -F "images=@./image2.jpg" http://localhost:5000/batch_enhance -o batch_enhanced.zip
"""

app = Flask(__name__)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MIRNet(num_features=64).to(device)
    checkpoint = torch.load(
        "model/weights/Mirnet_enhance_finetune-35-early-stopped_64x64.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


model, device = load_model()


def run_model(input_image):
    img_tensor = (
        T.Compose(
            [T.Resize(400), T.ToTensor(), T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])]
        )(input_image)
        .unsqueeze(0)
        .to(device)
    )

    # Adjusting tensor size
    if img_tensor.shape[2] % 8 != 0:
        img_tensor = img_tensor[:, :, : -(img_tensor.shape[2] % 8), :]
    if img_tensor.shape[3] % 8 != 0:
        img_tensor = img_tensor[:, :, :, : -(img_tensor.shape[3] % 8)]

    with torch.no_grad():
        output = model(img_tensor)
    vutils.save_image(output, open(f"temp.png", "wb"))
    output_image = Image.open("temp.png")
    os.remove("temp.png")
    return output_image


# Endpoint for single image
@app.route("/enhance", methods=["POST"])
def enhance_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    input_image = Image.open(file.stream).convert("RGB")
    output = run_model(input_image)

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="image/png",
        as_attachment=True,
        download_name="enhanced_image.png",
    )


# Endpoint for batch images
@app.route("/batch_enhance", methods=["POST"])
def batch_enhance_images():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    enhanced_images = []
    for file in files:
        input_image = Image.open(file.stream).convert("RGB")
        output = run_model(input_image)
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        buffer.seek(0)
        enhanced_images.append(buffer.getvalue())

    return jsonify({"enhanced_images": enhanced_images})


if __name__ == "__main__":
    app.run(debug=True)
