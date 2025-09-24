
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from .utils import load_multimodal_data
from .transforms import VideoTransforms
from .transforms import KeypointTransforms

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

        sample = {
            'sign_id': torch.tensor(self.class_to_idx[sign_id], dtype=torch.long),
            'person_id': torch.tensor(person_id, dtype=torch.long),
            'video_idx': torch.tensor(video_idx, dtype=torch.long),
            'clip_idx': torch.tensor(clip_idx, dtype=torch.long),
        }
        
        for modality in self.modalities:
            if modality not in data:
                raise ValueError(f"Modality '{modality}' not found in data.")
            
            if modality in ['rgb', 'optical_flow']:
                tensor_data = self._process_modality(data[modality], modality, clip_idx)
                sample[modality] = tensor_data
        return sample
    
    def _process_modality(self, data: np.ndarray, modality: str, clip_idx: int) -> torch.Tensor:
        tensor_data = torch.from_numpy(data.astype(np.float32))
        tensor_data = self._sample_clips(tensor_data, clip_idx)
        
        if modality == 'rgb':
            if self.video_transforms:
                tensor_data = self.video_transforms(tensor_data)
        elif modality == 'optical_flow':
            if tensor_data.dim() == 4:
                if self.output_format == 'CTHW':
                    tensor_data = tensor_data.permute(3, 0, 1, 2)
                elif self.output_format == 'TCHW':
                    tensor_data = tensor_data.permute(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unsupported output format: {self.output_format}")
        return tensor_data
    
    def _sample_clips(self, data: torch.Tensor, clip_idx: int) -> torch.Tensor:
        T = data.shape[0]
        if T < self.target_frames:
            repeat_factor = (self.target_frames + T - 1) // T
            data = data.repeat(repeat_factor, *([1] * (data.dim() - 1)))
            T = data.shape[0]
        
        if self.split == 'train':
            if T == self.target_frames:
                return data
            start = torch.randint(0, T - self.target_frames + 1, (1,)).item()
            return data[start:start + self.target_frames]
        else:
            if T <= self.target_frames:
                return data[:self.target_frames]
            if self.num_clips == 1:
                start = (T - self.target_frames) // 2
            else:
                available_starts = T - self.target_frames
                if available_starts <= 0:
                    start = 0
                else:
                    starts = np.linspace(0, available_starts, self.num_clips, dtype=int)
                    start = starts[clip_idx]
            return data[start:start + self.target_frames]
    
    def _extract_folder_name(self, video_path: str) -> str:
        return Path(video_path).parts[0]



class SkeletonNPYDataset(Dataset):
    def __init__(self,
                 annotation_file: str,
                 data_root: str,
                 split: str = 'train',
                 keypoint_type: str = '27',
                 target_frames: int = 64,
                 use_confidence: bool = False,
                 apply_augment: bool = True):
        self.data_root = Path(data_root)
        self.split = split
        self.keypoint_type = keypoint_type
        self.target_frames = target_frames
        self.use_confidence = use_confidence
        self.apply_augment = apply_augment and (split == 'train')

        self.df = pd.read_csv(annotation_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.df['sign_id'].unique()))}
        self.num_classes = len(self.class_to_idx)

        selected = np.concatenate((
            [0, 5, 6, 7, 8, 9, 10],
            [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
            [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]
        ))
        self.selected = selected

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        filename = Path(row['video_path']).stem + '.npy'
        base_path = self.data_root / Path(row['video_path']).parts[0]
        keypoint_path = base_path / 'keypoint' / filename
        kpts = np.load(str(keypoint_path))  # [T, 133, 3]

        # select 27 and use x,y only
        kpts = kpts[:, self.selected, :2]  # [T, 27, 2]

        # temporal sample
        kpts = self._temporal_sample(kpts, self.target_frames, train=(self.split == 'train'))

        # to [C, T, V, M], C=2, M=1
        data_numpy = kpts.transpose(2, 0, 1)[..., None]  # [2, T, 27, 1]

        kp_tf = KeypointTransforms(target_frames=self.target_frames, apply_augment=self.apply_augment)
        data_numpy = kp_tf(data_numpy)

        sample = {
            'skeleton': torch.from_numpy(data_numpy),
            'sign_id': torch.tensor(self.class_to_idx[row['sign_id']], dtype=torch.long),
            'person_id': torch.tensor(row['person_id'], dtype=torch.long),
        }
        return sample

    def _temporal_sample(self, arr: np.ndarray, target_frames: int, train: bool) -> np.ndarray:
        T = arr.shape[0]
        if T == target_frames:
            return arr
        if T < target_frames:
            repeat = (target_frames + T - 1) // T
            arr = np.repeat(arr, repeat, axis=0)
            T = arr.shape[0]
        if train:
            start = np.random.randint(0, T - target_frames + 1)
        else:
            start = (T - target_frames) // 2
        return arr[start:start+target_frames]
