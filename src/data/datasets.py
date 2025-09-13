
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from .utils import load_multimodal_data
from .transforms import VideoTransforms

class MultimodalSLRDataset(Dataset):
    def __init__(self,
                 annotation_file: str,
                 data_root: str,
                 split: str = 'train',
                 modalities: List[str] = ['rgb', 'keypoint', 'heatmap', 'optical_flow'],
                 target_frames: int = 16,
                 num_clips: int = 1,
                 video_transforms: Optional[VideoTransforms] = None,
                 target_fps: Optional[int] = None,
                 output_format: str = 'CTHW'):
        
        self.data_root = Path(data_root)
        self.split = split
        self.modalities = modalities
        self.target_frames = target_frames
        self.target_fps = target_fps
        self.num_clips = num_clips if split != 'train' else 1
        self.output_format = output_format

        self.df = pd.read_csv(annotation_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.num_classes = self.df['sign_id'].nunique()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.df['sign_id'].unique()))}
        
        self.video_transforms = video_transforms

        if self.video_transforms:
            self.video_transforms.training = (split == 'train')
    
    def __len__(self) -> int:
        if self.split == 'train':
            return len(self.df)
        else:
            return len(self.df) * self.num_clips
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split == 'train':
            video_idx = idx
            clip_idx = 0  
        else:
            video_idx = idx // self.num_clips
            clip_idx = idx % self.num_clips
        
        row = self.df.iloc[video_idx]
        video_path = row['video_path']
        sign_id = row['sign_id']
        person_id = row['person_id']
        
        filename = Path(video_path).stem + '.npy'
        folder_name = self._extract_folder_name(video_path)
        base_path = self.data_root / folder_name

        data = load_multimodal_data(str(base_path), filename)
        print(f"Loading idx={idx}, video_idx={video_idx}, clip_idx={clip_idx}")

        sample = {
            'sign_id': torch.tensor(self.class_to_idx[sign_id], dtype=torch.long),
            'person_id': torch.tensor(person_id, dtype=torch.long),
        }
        for modality in self.modalities:
            if modality not in data:
                raise ValueError(f"Modality '{modality}' not found in data.")
            
            if modality in ['rgb', 'optical_flow'] :
                tensor_data = self._process_modality(data[modality], modality, clip_idx)
                sample[modality] = tensor_data
        return sample
    
    def _process_modality(self, data: np.ndarray, modality: str, clip_idx: int) -> torch.Tensor:

        tensor_data = torch.from_numpy(data.astype(np.float32))
        tensor_data = self._sample_clips(tensor_data, clip_idx)
        
  
        if modality == 'rgb' and self.video_transforms:
            tensor_data = self.video_transforms(tensor_data)
            
        return tensor_data    
    def _sample_clips(self, data: torch.Tensor, clip_idx: int) -> torch.Tensor:
        T = data.shape[0]
        
        # Nếu video ngắn hơn target_frames, repeat
        if T < self.target_frames:
            repeat_factor = (self.target_frames + T - 1) // T
            data = data.repeat(repeat_factor, *([1] * (data.dim() - 1)))
            T = data.shape[0]
        
        if self.split == 'train':
            # Training: random crop
            if T == self.target_frames:
                return data
            start = torch.randint(0, T - self.target_frames + 1, (1,)).item()
            return data[start:start + self.target_frames]
        
        else:
            # Val/Test: uniform sampling
            segment_length = T // self.num_clips
            start = clip_idx * segment_length
            
            if start + self.target_frames > T:
                start = T - self.target_frames
                
            return data[start:start + self.target_frames]    
    def _extract_folder_name(self, video_path: str) -> str:
        return Path(video_path).parts[0]
