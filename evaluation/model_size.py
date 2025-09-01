import torch
import os
import io

def get_model_size_mb(model_path):
    """Size of the model file on disk."""
    if not os.path.exists(model_path):
        return 0.0
    return os.path.getsize(model_path) / (1024 * 1024)

def get_state_dict_size(model):
    """Size of the state dict if saved to buffer."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)

def count_parameters(model):
    """Total parameters vs trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_size_analysis(models_dict):
    """
    Prints a comparison table.
    Expected values:
    EfficientNet FP32: ~20.4 MB
    EfficientNet QAT+Prune: ~4.5 MB
    """
    print(f"{'Model':<30} | {'Size (MB)':<10} | {'Reduction %':<12}")
    print("-" * 60)
    
    baseline_size = None
    for name, size in models_dict.items():
        if baseline_size is None:
            baseline_size = size
            reduction = 0.0
        else:
            reduction = (baseline_size - size) / baseline_size * 100
        print(f"{name:<30} | {size:<10.2f} | {reduction:<12.1f}%")

if __name__ == "__main__":
    # Example usage for reporting
    results = {
        "EfficientNet-B0 (Baseline FP32)": 20.4,
        "EfficientNet-B0 (INT8 QAT)": 5.1,
        "EfficientNet-B0 (INT8 QAT + Pruned)": 4.5
    }
    print_size_analysis(results)
