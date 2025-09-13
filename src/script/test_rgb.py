import os
import wandb
import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data.datamodules import SLRDataModule
from src.models.modules.rgb_lightning import RGBLightningModule

@hydra.main(version_base=None, config_path="../../configs", config_name="model/rgb")
def train(cfg: DictConfig):
    print(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.model.gpu_id)

    pl.seed_everything(42)
    wandb.login()
    
    datamodule = SLRDataModule(
        annotation_file=cfg.model.datamodule.annotation_file,
        data_root=cfg.model.datamodule.data_root,
        modalities=['rgb'],
        batch_size=cfg.model.training.batch_size,
        video_size=(cfg.model.training.video_size, cfg.model.training.video_size),
        num_workers=cfg.model.training.num_workers
    )
    
    datamodule.setup()
    
    model = RGBLightningModule(
        num_classes=datamodule.num_classes,
        learning_rate=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
        num_test_clips=cfg.model.training.num_test_clips,
        target_frames=cfg.model.training.target_frames
    )
    
    print(f"Number of samples in train set: {len(datamodule.train_dataloader().dataset)}")
    sample = datamodule.train_dataloader().dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample info: {{k: type(v) for k, v in sample.items()}}")
    print(f"Sample shape: {sample['rgb'].shape}")
    print(f"Sample label: {sample['sign_id']}")
    
if __name__ == "__main__":
    train()