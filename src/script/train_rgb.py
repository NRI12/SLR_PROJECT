import os
import wandb
import hydra
import torch
from omegaconf import DictConfig, ListConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data.datamodules import SLRDataModule
from src.models.modules.rgb_lightning import RGBLightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.profilers import SimpleProfiler

profiler = SimpleProfiler()
@hydra.main(version_base=None, config_path="../../configs", config_name="model/rgb")
def train(cfg: DictConfig):
    print_fn = rank_zero_only(print)
    print_fn(cfg)
    gpu_ids = getattr(cfg.model, 'gpu_ids', None)
    gpu_id_value = getattr(cfg.model, 'gpu_id', None)
    if gpu_ids is None and isinstance(gpu_id_value, (list, tuple, ListConfig)):
        gpu_ids = list(gpu_id_value)

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
        devices = len(gpu_ids)
        strategy = 'ddp'
    else:
        if isinstance(gpu_id_value, ListConfig):
            gpu_id_value = list(gpu_id_value)[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id_value)
        devices = 1
        strategy = 'auto'

    pl.seed_everything(42)
    # wandb.login()
    
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
    
    wandb_logger = WandbLogger(
        project=cfg.model.logging.wandb_project,
        name=f"r2plus1d-{cfg.model.training.experiment_name}",
        log_model=True,
        tags=["rgb", "r2plus1d"]
    )
    
    # Add TensorBoard logger for local logging
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="rgb_training",
        version=None
    )
    
    # Use both loggers
    # loggers = [wandb_logger, tb_logger]
    loggers = [tb_logger]

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model.training.checkpoint_dir,
        filename='r2plus1d-{epoch:02d}-{val_top1:.4f}',
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
        devices=devices,
        strategy=strategy,
        num_sanity_val_steps=0,
        logger=loggers,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=cfg.model.logging.log_every_n_steps,
        val_check_interval=cfg.model.training.val_check_interval,
        precision="16-mixed",
        enable_progress_bar=True,
        profiler=profiler,
        # gradient_clip_val=1.0
    )
    print(profiler.summary())

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')
    
    # wandb.finish()

if __name__ == "__main__":
    train()