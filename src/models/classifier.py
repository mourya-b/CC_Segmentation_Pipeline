import torch
import torch.nn as nn
import timm


class CCClassifier(nn.Module):
    """
    Classifier with optional auxiliary segmentation head.
    - cls_head: frame-level classification (used at inference)
    - seg_head: per-patch segmentation supervision (training only)
    """
    def __init__(self, backbone="efficientnet_b0", pretrained=True, use_aux_seg=True):
        super().__init__()
        self.backbone_name = backbone
        self.use_aux_seg = use_aux_seg

        # Encoder without head or global pooling — exposes the raw feature map
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        feat_dim = self.encoder.num_features  # 1280 for B0, 2048 for ResNet50

        # Classification head — single logit, use BCEWithLogitsLoss
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 1),
        )

        # Auxiliary segmentation head — tiny, only used during training
        if use_aux_seg:
            self.seg_head = nn.Sequential(
                nn.Conv2d(feat_dim, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )
        else:
            self.seg_head = None

    def forward(self, x):
        feats = self.encoder(x)                       # (B, C, H/32, W/32)
        cls_logit = self.cls_head(feats).squeeze(1)   # (B,)
        if self.use_aux_seg:
            seg_logits = self.seg_head(feats)         # (B, 1, H/32, W/32)
            return cls_logit, seg_logits
        return cls_logit, None

    def get_head_param_names(self):
        return ["cls_head", "seg_head"]

    def get_backbone_layers(self):
        if "resnet" in self.backbone_name:
            return [self.encoder.layer1, self.encoder.layer2,
                    self.encoder.layer3, self.encoder.layer4]
        elif "efficientnet" in self.backbone_name:
            return list(self.encoder.blocks.children())
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")