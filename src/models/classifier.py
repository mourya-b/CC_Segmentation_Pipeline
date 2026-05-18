import torch
import torch.nn as nn
import timm


class CCClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=2, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    def get_head_param_names(self):
        """Return parameter name substrings that identify the classifier head."""
        if "resnet" in self.backbone_name:
            return ["fc"]
        elif "efficientnet" in self.backbone_name:
            return ["classifier"]
        else:
            return ["head", "classifier", "fc"]

    def get_backbone_layers(self):
        """Return list of top-level backbone layer modules for staged unfreezing."""
        if "resnet" in self.backbone_name:
            # ResNet50 layers: layer1, layer2, layer3, layer4
            return [
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
            ]
        elif "efficientnet" in self.backbone_name:
            return list(self.model.blocks.children())
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

    def forward(self, x):
        return self.model(x)