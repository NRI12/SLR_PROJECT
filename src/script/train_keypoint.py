import sys
sys.path.append('./src')
import pytorch_lightning as pl
from src.data.datamodules import SLRDataModule
from src.models.modules.slgcn_lightning import SLGCNLightningModule
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../../configs/model', config_name='sl_gcn')
def main(cfg: DictConfig):
    dm = SLRDataModule(
        annotation_file=cfg.data.annotation_file,
        data_root=cfg.data.data_root,
        modalities=cfg.data.modalities,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        target_frames=cfg.data.target_frames,
    )
    dm.setup()

    model = SLGCNLightningModule(
        num_classes=dm.num_classes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        keep_prob=cfg.model.keep_prob,
        graph=cfg.model.graph,
        num_point=cfg.model.num_point,
        in_channels=cfg.model.in_channels,
        num_person=cfg.model.num_person,
        groups=cfg.model.groups,
        block_size=cfg.model.block_size,
        graph_args=cfg.model.graph_args,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        default_root_dir=cfg.trainer.default_root_dir,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()


