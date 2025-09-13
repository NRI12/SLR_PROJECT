import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from .datasets import MultimodalSLRDataset
from .transforms import VideoTransforms

class SLRDataModule(pl.LightningDataModule):
    def __init__(self,
        annotation_file: str,
        data_root: str,
        modalities: List[str] = ['rgb'],
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_fps: Optional[int] = None,
        target_frames: int = 16,
        num_test_clips: int = 5,
        video_size: Tuple[int, int] = (224, 224),
        output_format: str = 'CTHW',
        augmentation_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.save_hyperparameters()
        
        self.annotation_file = annotation_file
        self.data_root = data_root
        self.modalities = modalities
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_fps = target_fps
        self.target_frames = target_frames
        self.num_test_clips = num_test_clips
        self.video_size = video_size
        self.output_format = output_format
        
        self.augmentation_config = augmentation_config or {}
        
        self.train_dataset: Optional[MultimodalSLRDataset] = None
        self.val_dataset: Optional[MultimodalSLRDataset] = None
        self.test_dataset: Optional[MultimodalSLRDataset] = None
        
        self.num_classes: Optional[int] = None

    def prepare_data(self) -> None:
        if not Path(self.annotation_file).exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        if not Path(self.data_root).exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}") 
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MultimodalSLRDataset(
                annotation_file=self.annotation_file,
                data_root=self.data_root,
                split='train',
                modalities=self.modalities,
                target_frames=self.target_frames,
                num_clips=1,
                video_transforms=self._get_video_transforms(train=True),
                target_fps=self.target_fps,
                output_format=self.output_format
            )
            
            self.val_dataset = MultimodalSLRDataset(
                annotation_file=self.annotation_file,
                data_root=self.data_root,
                split='val',
                modalities=self.modalities,
                target_frames=self.target_frames,
                num_clips=self.num_test_clips,
                video_transforms=self._get_video_transforms(train=False),
                target_fps=self.target_fps,
                output_format=self.output_format
            )
            
            self.num_classes = self.train_dataset.num_classes
        
        if stage == "test" or stage is None:
            self.test_dataset = MultimodalSLRDataset(
                annotation_file=self.annotation_file,
                data_root=self.data_root,
                split='test',
                modalities=self.modalities,
                target_frames=self.target_frames,
                num_clips=self.num_test_clips,
                video_transforms=self._get_video_transforms(train=False),
                target_fps=self.target_fps,
                output_format=self.output_format
            )
            
            if self.num_classes is None:
                self.num_classes = self.test_dataset.num_classes
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def _get_video_transforms(self, train: bool = True) -> Optional[VideoTransforms]:            
        if train:
            return VideoTransforms(
                resize=self.video_size,
                normalize=True,
                horizontal_flip_prob=self.augmentation_config.get('horizontal_flip_prob', 0.5),
                color_jitter=self.augmentation_config.get('color_jitter', True),
                output_format=self.output_format
            )
        else:
            return VideoTransforms(
                resize=self.video_size,
                normalize=True,
                horizontal_flip_prob=0.0,
                color_jitter=False,
                output_format=self.output_format
            )