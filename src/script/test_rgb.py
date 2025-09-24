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
    
    print(f"Number of samples in train set: {len(datamodule.train_dataloader().dataset)}")
    sample = datamodule.train_dataloader().dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample info: {{k: type(v) for k, v in sample.items()}}")
    print(f"Sample shape: {sample['rgb'].shape}")
    print(f"Sample label: {sample['sign_id']}")

    # Batches per epoch information
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print("\n" + "=" * 50)
    print("BATCHES PER EPOCH")
    print("=" * 50)
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Train batches/epoch: {len(train_loader)} (drop_last={train_loader.drop_last})")
    print(f"Val batches/epoch: {len(val_loader)} (drop_last={val_loader.drop_last})")
    print(f"Test batches/epoch: {len(test_loader)} (drop_last={test_loader.drop_last})")
    
    import itertools

    def print_batch_shapes(loader, name):
        print(f"\n{name.upper()} BATCH SHAPES:")
        for batch in itertools.islice(loader, 1):
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
            break

    print_batch_shapes(train_loader, "train")
    print_batch_shapes(val_loader, "val")
    print_batch_shapes(test_loader, "test")

    
if __name__ == "__main__":
    train()