import timm
import torch.nn as nn

def load_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """Load backbone ViT using timm."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return model
