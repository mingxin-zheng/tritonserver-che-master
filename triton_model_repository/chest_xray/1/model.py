import io
import sys
from collections import OrderedDict

sys.path.append('/opt/chestxray')

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from torchvision import transforms
from models.clip_tqn import CLP_clinical, ModelRes, TQN_Model
from transformers import AutoTokenizer

def handle_state_dict(source_state_dict, target_model):
    """
    Handles module prefix differences between source and target state dictionaries.
    
    Args:
        source_state_dict: The source state dictionary to load from
        target_model: The model to align the state dict with
        
    Returns:
        OrderedDict: Properly aligned state dictionary
    """
    new_state_dict = OrderedDict()
    if 'module.' in list(target_model.state_dict().keys())[0] and 'module.' not in list(source_state_dict.keys())[0]:
        for k, v in source_state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
    elif 'module.' not in list(target_model.state_dict().keys())[0] and 'module.' in list(source_state_dict.keys())[0]:
        for k, v in source_state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
    else:
        new_state_dict = source_state_dict
    return new_state_dict


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model components
        self.image_encoder = ModelRes('resnet152').to(self.device)
        try:
            self.text_encoder = CLP_clinical(bert_model_name='/opt/chestxray/pretrained_bert_weights/UMLSBert_ENG/').to(self.device)
        except Exception as e:
            print(f"Error initializing text encoder: {e}")
        self.model = TQN_Model(class_num=1).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load('/opt/chestxray/configs/base_checkpoint.pt', map_location='cpu')
        image_state_dict = checkpoint['image_encoder']
        
        new_image_state_dict = handle_state_dict(image_state_dict, self.image_encoder)
        self.image_encoder.load_state_dict(new_image_state_dict, strict=False)

        state_dict = checkpoint['model']
        new_state_dict = handle_state_dict(state_dict, self.model)
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # Set to evaluation mode
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('/opt/chestxray/pretrained_bert_weights/UMLSBert_ENG/', do_lower_case=True)
        
        # Setup text features
        self.text_list = ["atelectasis", "cardiomegaly", "pleural effusion", "infiltration", 
                         "lung mass", "lung nodule", "pneumonia", "pneumothorax", "consolidation", 
                         "edema", "emphysema", "fibrosis", "pleural thicken", "hernia"]
        
        # Precompute text features
        self.text_features = self._get_text_features()
        
        # Removing transform since we'll receive preprocessed images
        self.transform = None

    def _get_text_features(self):
        text_token = self.tokenizer(
            list(self.text_list),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.text_encoder.encode_text(text_token)
        return text_features

    def execute(self, requests):
        responses = []
        
        for request in requests:
            try:
                # Get input tensor (now expecting preprocessed image tensor)
                input_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                image = torch.from_numpy(input_tensor.as_numpy()).to(self.device)
                
                # Perform inference
                with torch.no_grad():
                    image_features, image_features_pool = self.image_encoder(image)
                    pred_class = self.model(image_features, self.text_features)
                    predictions = torch.sigmoid(pred_class)[:,:,0]
                
                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "predictions",
                    predictions.cpu().numpy().astype(np.float32)
                )
                
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            
            except Exception as e:
                error = pb_utils.TritonError(f"Error processing image: {str(e)}")
                inference_response = pb_utils.InferenceResponse(error=error)
            
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        pass
