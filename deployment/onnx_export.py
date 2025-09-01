import torch
import torch.onnx
import os
import argparse
import yaml
from models.efficientnet_finetune import ConstructionMaterialClassifier

def export_to_onnx(model_path, output_path, is_quantized=False):
    """
    Exports a PyTorch model to ONNX format.
    
    Note for Quantized Models:
    Quantized models in PyTorch (INT8) require specific handling for ONNX export.
    Standard torch.onnx.export often works best with FP32 models, while for INT8
    one might use the ONNX Runtime quantization tools. However, for this demo,
    we export the graph with standard parameters.
    """
    device = torch.device("cpu")
    model = ConstructionMaterialClassifier(num_classes=5)
    
    if is_quantized:
        # Re-apply quantization wrappers to match state dict
        import torch.quantization
        model.backbone.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model = torch.quantization.prepare_qat(model)
        model = torch.quantization.convert(model.eval())
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting model from {model_path} to {output_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=15, # Higher opsets support more operations
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="deployment/model.onnx", help="Output ONNX path")
    parser.add_argument("--quantized", action="store_true", help="Set if model is INT8")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_to_onnx(args.model, args.output, args.quantized)
