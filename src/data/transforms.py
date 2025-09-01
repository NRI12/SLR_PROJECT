import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T
import random

class VideoTransforms(nn.Module):
    def __init__(self, 
                 resize: Optional[Tuple[int, int]] = None,
                 normalize: bool = True,
                 horizontal_flip_prob: float = 0.5,
                 color_jitter: bool = True,
                 temporal_crop: bool = False,
                 max_frames: int = 64,
                 output_format: str = 'CTHW'):
        super().__init__()
        self.resize = resize
        self.normalize = normalize
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter
        self.temporal_crop = temporal_crop
        self.max_frames = max_frames
        self.output_format = output_format
        
        if normalize:
            self.norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if color_jitter:
            self.color_transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        T, H, W, C = video.shape
        video = video.permute(0, 3, 1, 2)  
        
        if self.resize:
            video = torch.nn.functional.interpolate(video, size=self.resize, mode='bilinear', align_corners=False)
        
        video = video.float() / 255.0
        
        if self.training and self.color_jitter:
            video = torch.stack([self.color_transform(frame) for frame in video])
        
        if self.training and random.random() < self.horizontal_flip_prob:
            video = torch.flip(video, dims=[3])
        
        if self.normalize:
            video = torch.stack([self.norm_transform(frame) for frame in video])
        
        if self.temporal_crop and self.training and T > self.max_frames:
            start_idx = random.randint(0, T - self.max_frames)
            video = video[start_idx:start_idx + self.max_frames]
        
        if self.output_format == 'CTHW':
            video = video.permute(1, 0, 2, 3)
        elif self.output_format == 'TCHW':
            pass
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        
        return video