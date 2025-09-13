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
def temporal_subsample(data: np.ndarray, target_frames: int) -> np.ndarray:
    current_frames = data.shape[0]
    if current_frames <= target_frames:
        return data
    
    indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
    return data[indices]
