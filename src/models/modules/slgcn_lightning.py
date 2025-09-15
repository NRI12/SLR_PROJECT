import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from src.models.architectures.sl_gcn import Model as SLGCN


class SLGCNLightningModule(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 keep_prob: float = 0.9, graph: str = 'src.models.architectures.graph_27.Graph',
                 num_point: int = 27, in_channels: int = 2, num_person: int = 1,
                 groups: int = 8, block_size: int = 41, graph_args: dict = {}):
        super().__init__()
        self.save_hyperparameters()

        self.model = SLGCN(num_class=num_classes,
                           num_point=num_point,
                           num_person=num_person,
                           groups=groups,
                           block_size=block_size,
                           graph=graph,
                           graph_args=graph_args,
                           in_channels=in_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.keep_prob = keep_prob

        self.top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

    def forward(self, x):
        return self.model(x, keep_prob=self.keep_prob)

    def training_step(self, batch, batch_idx):
        x, y = batch['skeleton'], batch['sign_id']  # x: [B,2,T,V,1]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_top1', self.top1(logits, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_top5', self.top5(logits, y), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['skeleton'], batch['sign_id']
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_top1', self.top1(logits, y), on_epoch=True, prog_bar=True)
        self.log('val_top5', self.top5(logits, y), on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['skeleton'], batch['sign_id']
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_top1', self.top1(logits, y), on_epoch=True)
        self.log('test_top5', self.top5(logits, y), on_epoch=True)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


