import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from models.efficientnet_finetune import ConstructionMaterialClassifier
from models.quantization import prepare_for_qat, convert_to_int8
from models.pruning import apply_structured_pruning
from data.transforms import get_transforms
from training.train import train_epoch, validate

def qat_fine_tune(checkpoint_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu") # QAT for CPU deployment
    print(f"Running QAT on {device} (FBGEMM backend)")
    
    # Data loading
    data_dir = config['data']['data_dir']
    batch_size = config['training']['batch_size']
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=get_transforms("train"))
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=get_transforms("val"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load FP32 Model
    model = ConstructionMaterialClassifier(num_classes=5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded FP32 checkpoint: {checkpoint_path}")
    
    # Verify baseline accuracy
    baseline_loss, baseline_acc = validate(model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"FP32 Baseline - Loss: {baseline_loss:.4f} Acc: {baseline_acc:.2f}%")
    
    # Prepare for QAT
    print("\nPreparing model for Quantization-Aware Training...")
    model_prepared = prepare_for_qat(model, backend=config['qat']['backend'])
    
    # QAT Fine-tuning
    optimizer = optim.Adam(model_prepared.parameters(), lr=config['qat']['lr'])
    criterion = nn.CrossEntropyLoss()
    
    print(f"Fine-tuning QAT for {config['qat']['fine_tune_epochs']} epochs...")
    for epoch in range(config['qat']['fine_tune_epochs']):
        train_loss, train_acc = train_epoch(model_prepared, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model_prepared, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config['qat']['fine_tune_epochs']} - QAT Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    # Apply Structured Pruning
    print(f"\nApplying structured pruning (ratio: {config['qat']['prune_ratio']})...")
    model_pruned = apply_structured_pruning(model_prepared, 
                                          prune_ratio=config['qat']['prune_ratio'],
                                          target_blocks=config['qat']['prune_target_blocks'])
    
    # Convert to INT8
    print("Converting to INT8 quantized model...")
    model_quantized = convert_to_int8(model_pruned)
    
    # Final Evaluation
    final_loss, final_acc = validate(model_quantized, val_loader, criterion, device)
    print(f"\nFinal Compressed Model (QAT+Prune) - Loss: {final_loss:.4f} Acc: {final_acc:.2f}%")
    
    # Save checkpoint
    output_path = os.path.join(config['checkpointing']['output_dir'], "qat_pruned_model.pth")
    torch.save(model_quantized.state_dict(), output_path)
    print(f"Compressed model saved to {output_path}")
    
    # Size Comparison
    fp32_size = model.get_model_size_mb()
    # Save quantized to buffer to get real size
    import io
    buffer = io.BytesIO()
    torch.save(model_quantized.state_dict(), buffer)
    int8_size = buffer.tell() / (1024 * 1024)
    reduction = (fp32_size - int8_size) / fp32_size * 100
    
    print(f"Size Reduction: {fp32_size:.2f}MB -> {int8_size:.2f}MB ({reduction:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QAT fine-tuning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best FP32 checkpoint")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config file")
    args = parser.parse_args()
    qat_fine_tune(args.checkpoint, args.config)
