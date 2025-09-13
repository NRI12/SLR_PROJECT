import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

class R2Plus1DModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone = r2plus1d_18(pretrained=pretrained)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)