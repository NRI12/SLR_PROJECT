import numpy as np
import torch
from pathlib import Path
from typing import Dict

def load_multimodal_data(base_path: str, filename: str) -> Dict[str, np.ndarray]:
    base_path = Path(base_path)
    data = {}
    
    modalities = ['rgb', 'keypoint', 'heatmap', 'optical_flow']
    for modality in modalities:
        file_path = base_path / modality / filename
        if file_path.exists():
            data[modality] = np.load(file_path, mmap_mode='r')
    return data

def random_temporal_crop(x, target_frames):
    B, C, T, H, W = x.shape
    if T < target_frames:
        repeat_factor = (target_frames + T - 1) // T
        x = x.repeat(1, 1, repeat_factor, 1, 1)
        T = x.shape[2]
    
    start = torch.randint(0, T - target_frames + 1, (1,)).item()
    return x[:, :, start:start+target_frames, :, :]

def multi_clip_inference(self, x, target_frames,num_test_clips):
    B, C, T, H, W = x.shape
    
    if T < target_frames:
        repeat_factor = (target_frames + T - 1) // T
        x = x.repeat(1, 1, repeat_factor, 1, 1)
        T = x.shape[2]
    
    if T <= target_frames:
        return self(x[:, :, :target_frames, :, :])
    
    clip_logits = []
    if T <= target_frames * num_test_clips:
        step = max(1, (T - target_frames) // (num_test_clips - 1))
        for i in range(num_test_clips):
            start = min(i * step, T - target_frames)
            clip = x[:, :, start:start+target_frames, :, :]
            logits = self(clip)
            clip_logits.append(logits)
    else:
        step = T // num_test_clips
        for i in range(num_test_clips):
            start = i * step
            end = min(start + target_frames, T)
            if end - start < target_frames:
                start = max(0, end - target_frames)
            clip = x[:, :, start:start+target_frames, :, :]
            logits = self(clip)
            clip_logits.append(logits)
    
    return torch.stack(clip_logits).mean(dim=0)