import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data.datamodules import SLRDataModule
from src.models.modules.rgb_lightning import RGBLightningModule

@hydra.main(version_base=None, config_path="../../configs", config_name="model/rgb")
def train(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    
    
    pl.seed_everything(42)
    wandb.login()
    
    datamodule = SLRDataModule(
        annotation_file=cfg.datamodule.annotation_file,
        data_root=cfg.datamodule.data_root,
        modalities=['rgb'],
        batch_size=cfg.training.batch_size,
        video_size=(cfg.training.video_size, cfg.training.video_size),
        num_workers=cfg.training.num_workers
    )
    
    datamodule.setup()
    
    model = RGBLightningModule(
        num_classes=datamodule.num_classes,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_test_clips=cfg.training.num_test_clips,
        target_frames=cfg.training.target_frames
    )
    
    wandb_logger = WandbLogger(
        project=cfg.logging.wandb_project,
        name=f"r2plus1d-{cfg.training.experiment_name}",
        log_model=True,
        tags=["rgb", "r2plus1d"]
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename='r2plus1d-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=cfg.training.patience,
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        precision=16,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')
    
    wandb.finish()

if __name__ == "__main__":
    train()