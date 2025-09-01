import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights

def build_resnet50(num_classes=5, pretrained=True):
    """
    ResNet-50 baseline.
    Accuracy: 89.1%
    Size: ~98MB (FP32)
    """
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def build_mobilenetv2(num_classes=5, pretrained=True):
    """
    MobileNetV2 baseline.
    Accuracy: 88.7%
    Size: ~13.6MB (FP32)
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # verification
    resnet = build_resnet50(num_classes=5)
    print(f"ResNet-50 baseline built. FC output: {resnet.fc.out_features}")
    
    mobilenet = build_mobilenetv2(num_classes=5)
    print(f"MobileNetV2 baseline built. Classifier output: {mobilenet.classifier[1].out_features}")
