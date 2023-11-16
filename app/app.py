import streamlit as st
from PIL import Image
import io
import sys, os
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import base64

sys.path.append(".")
from model.MIRNet.model import MIRNet


def run_model(input_image):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"-> {device.type} device detected.")

    model = MIRNet(num_features=64).to(device)
    checkpoint = torch.load(
        "model/weights/Mirnet_enhance_finetune-35-early-stopped_64x64.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        img = input_image
        img_tensor = T.Compose(
            [
                T.Resize(400),
                T.ToTensor(),
                T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ]
        )(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        if img_tensor.shape[2] % 8 != 0:
            img_tensor = img_tensor[:, :, : -(img_tensor.shape[2] % 8), :]
        if img_tensor.shape[3] % 8 != 0:
            img_tensor = img_tensor[:, :, :, : -(img_tensor.shape[3] % 8)]

        output = model(img_tensor)

        vutils.save_image(output, open(f"temp.png", "wb"))
        output_image = Image.open("temp.png")
        os.remove("temp.png")
        return output_image


def get_base64_font(font_path):
    with open(font_path, "rb") as font_file:
        return base64.b64encode(font_file.read()).decode()


st.set_page_config(layout="wide")

font_name = "Gloock"
gloock_b64 = get_base64_font("utils/assets/Gloock-Regular.ttf")
font_name_text = "Merriweather sans"
merri_b64 = get_base64_font("utils/assets/MerriweatherSans-Regular.ttf")
hide_streamlit_style = f"""
            <style>
            #MainMenu {'{visibility: hidden;}'}
            footer {'{visibility: hidden;}'}
            
            @font-face {{
                font-family: '{font_name}';
                src: url(data:font/ttf;base64,{gloock_b64}) format('truetype');
            }}
            @font-face {{
                font-family: '{font_name_text}';
                src: url(data:font/ttf;base64,{merri_b64}) format('truetype');
            }}
            span {{
                font-family: '{font_name_text}';  
            }}
            .e1nzilvr1, .st-emotion-cache-10trblm {{
                font-family: '{font_name}';
                font-size: 65px;
            }}
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Low-light event-image enhancement with MIRNet.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

    # Create two columns for images
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width="always")

        # Button to enhance image
        if st.button("Enhance Image"):
            with col2:
                # Assume your model has a function 'enhance' to enhance the image
                enhanced_image = run_model(image)
                st.image(
                    enhanced_image, caption="Enhanced Image", use_column_width="always"
                )

                # Download button
                buf = io.BytesIO()
                enhanced_image.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="enhanced_image.jpg",
                    mime="image/jpeg",
                )
