import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from PIL import Image
import requests
import io

# Initialize the client
client = grpcclient.InferenceServerClient(url="localhost:8031")

# Define image preprocessing function
def preprocess_image(image):
    # Resize
    image = image.resize((224, 224), Image.BICUBIC)
    
    # Convert to numpy array and normalize to 0-1
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Transpose from (H, W, C) to (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    return img_array.astype(np.float32)

# Download and preprocess the image
image_url = "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_00026451_030.jpg"
response = requests.get(image_url)
image = Image.open(io.BytesIO(response.content)).convert('RGB')
image_tensor = preprocess_image(image)

# Add batch dimension and rearrange dimensions to match expected shape [batch, height, width, channels]
image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension  # Rearrange to [batch, height, width, channels]

# Create input tensor
input_tensor = grpcclient.InferInput(
    name="image",
    shape=image_tensor.shape,
    datatype=np_to_triton_dtype(image_tensor.dtype)
)
input_tensor.set_data_from_numpy(image_tensor)

# Send request
response = client.infer(
    model_name="chest_xray",
    inputs=[input_tensor]
)

# Get predictions
output = response.as_numpy("predictions")
print("Predictions:", output)

# Optional: Print predictions with labels
labels = ["atelectasis", "cardiomegaly", "pleural effusion", "infiltration", 
          "lung mass", "lung nodule", "pneumonia", "pneumothorax", "consolidation", 
          "edema", "emphysema", "fibrosis", "pleural thicken", "hernia"]

for label, pred in zip(labels, output[0]):
    print(f"{label}: {pred:.4f}")
