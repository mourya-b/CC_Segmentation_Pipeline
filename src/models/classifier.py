import torch
import torch.nn as nn
import timm


class CCClassifier(nn.Module):
    """EfficientNet-based binary classifier for CC detection."""

    def __init__(self, backbone="efficientnet_b0", num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)