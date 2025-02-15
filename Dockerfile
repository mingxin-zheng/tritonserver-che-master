FROM nvcr.io/nvidia/tritonserver:25.01-pyt-python-py3

# Install lower version of torch t avoid model loading error
RUN pip install \
    "torch<2.6" \
    torchvision \
    transformers \
    pillow \
    pyyaml \
    numpy \
    pandas \
    tensorboard \
    timm \
    einops

# # Copy model files and code
# RUN mkdir -p /opt/chestxray

# COPY configs/ /opt/chestxray/configs/
# COPY factory/ /opt/chestxray/factory/
# COPY models/ /opt/chestxray/models/
# COPY pretrained_bert_weights/ /opt/chestxray/pretrained_bert_weights/
# COPY test_example.py /opt/chestxray/test_example.py

RUN apt-get update && apt-get install -y git-lfs && git lfs install

RUN git clone https://huggingface.co/Elfenreigen/CHE-Master /opt/chestxray && \
    cd /opt/chestxray && \
    git lfs pull

# Create model repository directory
RUN mkdir -p /models

# Copy Triton model configuration
COPY triton_model_repository/ /models/

# docker build -t chest-xray-triton .
# docker run --gpus "device=2" -p 8030:8000 -p 8031:8001 -p 8032:8002 chest-xray-triton tritonserver --model-repository=/models