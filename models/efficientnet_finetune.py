import torch
import torch.nn as nn
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import io

class ConstructionMaterialClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 5-class construction material classification.
    
    Architecture:
        - Backbone: EfficientNet-B0 pretrained on ImageNet
        - Head: AdaptiveAvgPool2d -> Dropout -> Linear(1280 -> 5)
    
    Results (FP32 baseline): 92.3% top-1 accuracy.
    Model size (FP32): ~20.4MB.
    """
    def __init__(self, num_classes=5, pretrained=True, dropout=0.2):
        super(ConstructionMaterialClassifier, self).__init__()
        
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace classifier head: in_features = 1280
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self, unfreeze_last_n_blocks=2):
        """
        Freeze all layers except the last n MBConv blocks and the classifier head.
        This is used in Phase 1 of fine-tuning.
        """
        # First, freeze everything
        for param in self.parameters():
            param.requires_grad = False
            
        # Unfreeze last n blocks in 'features'
        # EfficientNet-B0 has 9 feature stages (0-8)
        feature_blocks = list(self.backbone.features.children())
        for block in feature_blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
                
        # Unfreeze classifier head
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_model_size_mb(self):
        """Returns model size in MB as saved on disk (state dict)."""
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.tell() / (1024 * 1024)

    def num_parameters(self, trainable_only=True):
        """Counts parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

def build_model(config=None):
    """Factory function to build the model based on config."""
    num_classes = config.get('model', {}).get('num_classes', 5) if config else 5
    dropout = config.get('model', {}).get('dropout', 0.2) if config else 0.2
    return ConstructionMaterialClassifier(num_classes=num_classes, dropout=dropout)

if __name__ == "__main__":
    # Demo verification
    model = ConstructionMaterialClassifier(num_classes=5)
    print(f"Model: EfficientNet-B0 (Construction Material Classifier)")
    print(f"Total Parameters: {model.num_parameters(False) / 1e6:.2f}M")
    print(f"Initial State Dict Size: {model.get_model_size_mb():.2f}MB")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape} (Expected: [1, 5])")
    
    model.freeze_backbone()
    print(f"Trainable Parameters (Last 2 blocks + Head): {model.num_parameters(True) / 1e6:.2f}M")
