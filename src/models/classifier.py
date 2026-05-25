import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """Generalized Mean Pooling — focuses on localized high-confidence regions."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


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

        # Classification head — GeM pooling + single logit, BCEWithLogitsLoss
        self.cls_head = nn.Sequential(
            GeM(p=3.0),
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