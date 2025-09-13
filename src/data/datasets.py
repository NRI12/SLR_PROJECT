
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
                 max_frames: int = None,
                 video_transforms: Optional[VideoTransforms] = None,
                 target_fps: Optional[int] = None,
                 output_format: str = 'CTHW'):
        
        self.data_root = Path(data_root)
        self.split = split
        self.modalities = modalities
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.output_format = output_format

        self.df = pd.read_csv(annotation_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.num_classes = self.df['sign_id'].nunique()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.df['sign_id'].unique()))}
        
        self.video_transforms = video_transforms

        if self.video_transforms:
            self.video_transforms.training = (split == 'train')
    
    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        video_path = row['video_path']
        sign_id = row['sign_id']
        person_id = row['person_id']
        
        filename = Path(video_path).stem + '.npy'
        folder_name = self._extract_folder_name(video_path)
        base_path = self.data_root / folder_name

        data = load_multimodal_data(str(base_path), filename)

        sample = {
            'sign_id': torch.tensor(self.class_to_idx[sign_id], dtype=torch.long),
            'person_id': torch.tensor(person_id, dtype=torch.long),
        }
        if 'rgb' in self.modalities and 'rgb' in data:
            rgb = torch.from_numpy(data['rgb'])
         
            if self.target_fps:
                rgb = self._subsample_by_fps(rgb, self.target_fps)
                
            if self.video_transforms:
                rgb = self.video_transforms(rgb)
            sample['rgb'] = rgb

        if 'keypoint' in self.modalities and 'keypoint' in data:
            keypoints = torch.from_numpy(data['keypoint'].astype(np.float32))
            if self.target_fps:
                keypoints = self._subsample_by_fps(keypoints, self.target_fps)
            sample['keypoint'] = keypoints

        if 'heatmap' in self.modalities and 'heatmap' in data:
            heatmaps = torch.from_numpy(data['heatmap'].astype(np.float32))
            if self.target_fps:
                heatmaps = self._subsample_by_fps(heatmaps, self.target_fps)
            sample["heatmap"] = heatmaps
            
        if 'optical_flow' in self.modalities and 'optical_flow' in data:
            flow = torch.from_numpy(data['optical_flow'].astype(np.float32))
            if self.target_fps:
                flow = self._subsample_by_fps(flow, self.target_fps)
            sample["optical_flow"] = flow
        return sample
    
    def _extract_folder_name(self, video_path: str) -> str:
        return Path(video_path).parts[0]

    def _subsample_by_fps(self, data: torch.Tensor, target_fps: int) -> torch.Tensor:
        current_frames = data.shape[0]
        if current_frames <= target_fps:
            return data
        
        indices = torch.linspace(0, current_frames - 1, target_fps).long()
        return data[indices]    

