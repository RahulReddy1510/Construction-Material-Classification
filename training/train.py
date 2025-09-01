import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from models.efficientnet_finetune import ConstructionMaterialClassifier
from data.transforms import get_transforms
from data.dataset_prep import compute_class_weights

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*correct/total:.2f}%"})
        
    return running_loss / total, 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, 100. * correct / total

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    data_dir = config['data']['data_dir']
    batch_size = config['training']['batch_size']
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=get_transforms("train"))
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=get_transforms("val"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config['data']['num_workers'])
    
    # Model
    model = ConstructionMaterialClassifier(num_classes=5).to(device)
    
    # Loss with class weights if enabled
    if config['training']['weighted_loss']:
        weights = compute_class_weights(data_dir).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Phase 1: Freeze backbone, train head only
    print("\n--- Phase 1: Training Head Only (Backbone Frozen) ---")
    model.freeze_backbone()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['phase1_lr'])
    
    for epoch in range(config['training']['phase1_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config['training']['phase1_epochs']} - Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
    # Phase 2: Unfreeze last 2 blocks
    print("\n--- Phase 2: Fine-tuning Last 2 MBConv Blocks ---")
    model.freeze_backbone(unfreeze_last_n_blocks=2)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['phase2_lr'])
    
    best_acc = 0.0
    for epoch in range(config['training']['phase2_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config['training']['phase2_epochs']} - Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(config['checkpointing']['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config['checkpointing']['output_dir'], "efficientnet_best.pth"))
            
    # Phase 3: Full fine-tuning
    print("\n--- Phase 3: Full Fine-tuning ---")
    model.unfreeze_all()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['phase3_lr'])
    
    for epoch in range(config['training']['phase3_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config['training']['phase3_epochs']} - Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['checkpointing']['output_dir'], "efficientnet_best.pth"))

    print(f"\nTraining Complete. Best Val Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train construction material classifier")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
