FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y gcc curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY . .

# Download model weights from GitHub release
ARG MODEL_URL=https://github.com/dblasko/low-light-event-img-enhancer/releases/download/mirnet-finetuned-1.0.0-100epochs/mirnet_lowlight-enhance_finetuned_100-epochs_early-stopped_64x64.pth
RUN curl -L -o model/weights/Mirnet_enhance_finetune-35-early-stopped_64x64.pth ${MODEL_URL}

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app/api.py"]
