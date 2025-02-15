# Chest X-Ray Classification Triton Server

This repository demonstrates how to serve a chest X-ray classification model from HuggingFace using NVIDIA Triton Inference Server. The model can detect 14 different medical conditions from chest X-ray images.

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Python 3.8+
- Git LFS

## Quick Start

### 1. Build the Docker Container

```bash
docker build -t chest-xray-triton .
```

### 2. Run the Docker Container

```bash
docker run -d --gpus "device=0" -p 8000:8000 8001:8001 -p 8002:8002 chest-xray-triton tritonserver --model-repository=/models
```

### 3. Test the Model

```bash
# You may need to create a new virtual environment
pip install -r requirements.txt
python client.py
```


## Scaling the server to enable multiple instances on a single GPU

Please refer to the [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#multiple-model-instances)

You may change the `count` in the instance group section of the [config.pbtxt](triton_model_repository/chest_xray/config.pbtxt) file to enable multiple instances on a single GPU.
