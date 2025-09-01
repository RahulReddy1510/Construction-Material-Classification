import torch
import torch.nn as nn
import torch.quantization
import copy
import io

def prepare_for_qat(model, backend="fbgemm"):
    """
    Prepares a model for Quantization-Aware Training (QAT).
    
    EfficientNet-B0 requires QAT (not just PTQ) because its Swish/SiLU
    activations have an unbounded positive range and longer tails than ReLU.
    PTQ observers often underestimate these ranges, causing accuracy drops.
    QAT fine-tunes with fake quantization to learn appropriate scale/zero-points.
    
    Backend:
        'fbgemm': x86 CPUs (default)
        'qnnpack': ARM/Mobile
    """
    model_qat = copy.deepcopy(model)
    model_qat.train() # Must be in training mode for QAT preparation
    
    # Set quantization configuration
    model_qat.backbone.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    # Fuse common patterns (Conv + BN + Activation) for efficiency
    # Note: For EfficientNet, we fuse the common MBConv components.
    # This is a simplified fusion for the demo; full fusion depends on exact module names.
    # In real implementation, one would iterate over features and fuse modules that match.
    # For now, we rely on prepare_qat's automatic handling for standard patterns.
    
    # Prepares the model for QAT by inserting FakeQuantize modules
    model_qat = torch.quantization.prepare_qat(model_qat, inplace=True)
    
    return model_qat

def convert_to_int8(model_qat):
    """
    Converts a QAT-fine-tuned model into a real INT8 quantized model.
    CRITICAL: model.eval() must be called before conversion to freeze calibration stats.
    """
    model_int8 = copy.deepcopy(model_qat)
    model_int8.eval() # CRITICAL: Switch to eval mode
    
    # Converts FakeQuantize modules into actual quantized operations
    model_int8 = torch.quantization.convert(model_int8, inplace=True)
    
    return model_int8

def run_ptq(model, data_loader, backend="fbgemm", n_batches=32):
    """
    Post-Training Static Quantization (included for baseline comparison).
    Expected to drop accuracy on EfficientNet-B0.
    """
    model_ptq = copy.deepcopy(model)
    model_ptq.eval()
    
    model_ptq.backbone.qconfig = torch.quantization.get_default_qconfig(backend)
    model_ptq = torch.quantization.prepare(model_ptq, inplace=True)
    
    # Calibration step: pass data through the model to observe activation ranges
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            model_ptq(images)
            if i >= n_batches:
                break
                
    model_ptq = torch.quantization.convert(model_ptq, inplace=True)
    return model_ptq

def compare_sizes(fp32_model, int8_model):
    """Compares sizes of models in MB."""
    fp32_size = fp32_model.get_model_size_mb() if hasattr(fp32_model, 'get_model_size_mb') else 0
    
    # For quantized models, we measure the state dict size as well
    buffer = io.BytesIO()
    torch.save(int8_model.state_dict(), buffer)
    int8_size = buffer.tell() / (1024 * 1024)
    
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB")
    print(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
    return fp32_size, int8_size

if __name__ == "__main__":
    # Demo verification with a dummy model
    from .efficientnet_finetune import ConstructionMaterialClassifier
    model = ConstructionMaterialClassifier(num_classes=5)
    
    print("--- Testing QAT Preparation ---")
    model_prepared = prepare_for_qat(model)
    print("Model prepared for QAT (FakeQuant modules added).")
    
    print("\n--- Testing Conversion (Simulated) ---")
    # In a real run, you'd fine-tune here
    model_int8 = convert_to_int8(model_prepared)
    print("Model converted to INT8.")
    
    compare_sizes(model, model_int8)
