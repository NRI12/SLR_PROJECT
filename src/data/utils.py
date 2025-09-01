import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2

def load_multimodal_data(base_path: str, filename: str) -> Dict[str, np.ndarray]:
    base_path = Path(base_path)
    data = {}
    
    modalities = ['rgb', 'keypoint', 'heatmap', 'optical_flow']
    for modality in modalities:
        file_path = base_path / modality / filename
        if file_path.exists():
            data[modality] = np.load(file_path)
    
    return data

# def pad_sequence(data: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
#     current_length = data.shape[0]
#     if current_length >= target_length:
#         return data[:target_length]
    
#     pad_length = target_length - current_length
#     if data.ndim == 4:
#         pad_shape = (pad_length, data.shape[1], data.shape[2], data.shape[3])
#     elif data.ndim == 3:
#         pad_shape = (pad_length, data.shape[1], data.shape[2])
#     elif data.ndim == 2:
#         pad_shape = (pad_length, data.shape[1])
#     else:
#         raise ValueError(f"Unsupported data dimension: {data.ndim}")
    
#     padding = np.full(pad_shape, pad_value, dtype=data.dtype)
#     return np.concatenate([data, padding], axis=0)

def temporal_subsample(data: np.ndarray, target_frames: int) -> np.ndarray:
    current_frames = data.shape[0]
    if current_frames <= target_frames:
        return data
    
    indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
    return data[indices]
