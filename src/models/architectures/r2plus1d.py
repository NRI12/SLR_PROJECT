import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from typing import Optional

class R2Plus1DModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, dropout=0.5, in_channels: int = 3):
        super().__init__()
        self.backbone = r2plus1d_18(pretrained=pretrained)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        # Change the input channels and load pretrain for optical flow and rgb
        if in_channels != 3:
            old_conv = self.backbone.stem[0]
            new_conv = nn.Conv3d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            if pretrained and old_conv.weight.shape[1] == 3:
                with torch.no_grad():
                    if in_channels == 2:
                        # Average the RGB weights across channels and tile to 2 channels
                        w = old_conv.weight
                        mean_w = w.mean(dim=1, keepdim=True)  # [out,1,kT,kH,kW]
                        new_w = mean_w.repeat(1, 2, 1, 1, 1)
                        new_conv.weight.copy_(new_w)
                    else:
                        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                    if new_conv.bias is not None:
                        new_conv.bias.zero_()
            self.backbone.stem[0] = new_conv
    def forward(self, x):
        return self.backbone(x)