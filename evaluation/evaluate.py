import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import yaml

from models.efficientnet_finetune import ConstructionMaterialClassifier
from models.quantization import convert_to_int8
from data.transforms import get_transforms
from evaluation.metrics import compute_accuracy, compute_per_class_accuracy, get_full_report

def evaluate_model(model_path, data_dir, config_path, is_quantized=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu") # Benchmarking on CPU
    class_names = config['model']['classes']
    
    # Dataset
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=get_transforms("test"))
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model Loading
    model = ConstructionMaterialClassifier(num_classes=5)
    if is_quantized:
        # Prepare for quantization format
        import torch.quantization
        model.backbone.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model = torch.quantization.prepare_qat(model)
        model = torch.quantization.convert(model.eval())
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"\n--- Evaluating Model: {os.path.basename(model_path)} ---")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            all_preds.append(outputs.numpy())
            all_labels.append(labels.numpy())
            
    preds = np.vstack(all_preds)
    labels = np.concatenate(all_labels)
    
    acc = compute_accuracy(preds, labels)
    per_class = compute_per_class_accuracy(preds, labels, class_names)
    report = get_full_report(preds, labels, class_names)
    
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("Per-Class Accuracy:")
    for cls, val in per_class.items():
        print(f"  {cls}: {val}")
    print("\nDetailed Report:")
    print(report)
    
    return acc, per_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, default="data/synthetic/", help="Data directory")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Config path")
    parser.add_argument("--quantized", action="store_true", help="Set if model is INT8 quantized")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data, args.config, args.quantized)
