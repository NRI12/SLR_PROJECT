import os
import wandb
import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data.datamodules import SLRDataModule
from src.models.modules.flow_lightning import FlowLightningModule

@hydra.main(version_base=None, config_path="../../configs", config_name="model/flow")
def train(cfg: DictConfig):
    print(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.model.gpu_id)

    pl.seed_everything(42)
    # wandb.login()
    
    datamodule = SLRDataModule(
        annotation_file=cfg.model.datamodule.annotation_file,
        data_root=cfg.model.datamodule.data_root,
        modalities=['optical_flow'],
        batch_size=cfg.model.training.batch_size,
        video_size=(cfg.model.training.video_size, cfg.model.training.video_size),
        num_workers=cfg.model.training.num_workers
    )
    
    datamodule.setup()
    
    model = FlowLightningModule(
        num_classes=datamodule.num_classes,
        learning_rate=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
        num_test_clips=cfg.model.training.num_test_clips,
        target_frames=cfg.model.training.target_frames
    )
    
    wandb_logger = WandbLogger(
        project=cfg.model.logging.wandb_project,
        name=f"r2plus1d-flow-{cfg.model.training.experiment_name}",
        log_model=True,
        tags=["optical_flow", "r2plus1d"]
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model.training.checkpoint_dir,
        filename='r2plus1d-flow-{epoch:02d}-{val_top1:.4f}',
        monitor='val_top1',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_top1',
        mode='max',
        patience=cfg.model.training.patience,
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.model.training.max_epochs,
        accelerator='gpu',
        devices=1,
        num_sanity_val_steps=10,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=cfg.model.logging.log_every_n_steps,
        val_check_interval=cfg.model.training.val_check_interval,
        precision=16,
        gradient_clip_val=1.0,
        enable_progress_bar=False
    )
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')
    
    # wandb.finish()

if __name__ == "__main__":
    train()

