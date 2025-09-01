import cv2
import torch
import numpy as np
import argparse
import os
from models.efficientnet_finetune import ConstructionMaterialClassifier
from data.transforms import get_transforms

class InferenceRunner:
    def __init__(self, model_path, class_names, device="cpu"):
        self.device = torch.device(device)
        self.class_names = class_names
        self.transform = get_transforms("test")
        
        # Load model (assuming FP32 for simple inference script)
        self.model = ConstructionMaterialClassifier(num_classes=len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval().to(self.device)
        
    def predict(self, image_path):
        # Load image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to RGB and PIL style for transforms
        from PIL import Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Preprocess
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get results
        prob, idx = torch.max(probabilities, 0)
        return self.class_names[idx], prob.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    classes = ["concrete", "brick", "metal", "wood", "stone"]
    runner = InferenceRunner(args.model, classes)
    
    label, confidence = runner.predict(args.image)
    print(f"Prediction: {label} ({confidence*100:.2f}%)")
