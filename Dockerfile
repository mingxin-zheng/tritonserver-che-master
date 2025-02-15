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


RUN apt-get update && apt-get install -y git-lfs && git lfs install

RUN git clone https://huggingface.co/Elfenreigen/CHE-Master /opt/chestxray && \
    cd /opt/chestxray && \
    git lfs pull

# Create model repository directory
RUN mkdir -p /models

# Copy Triton model configuration
COPY triton_model_repository/ /models/
