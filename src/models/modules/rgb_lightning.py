import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from src.models.architectures.r2plus1d import R2Plus1DModel

class RGBLightningModule(pl.LightningModule):
    def __init__(self, num_classes=100, learning_rate=1e-3, weight_decay=1e-4, 
                 num_test_clips=5, target_frames=16):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = R2Plus1DModel(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.num_test_clips = num_test_clips
        self.target_frames = target_frames
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_predictions.append(logits)
        self.val_targets.append(y)    
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_predictions.append(logits)
        self.test_targets.append(y)
        self.log('test_loss', loss, on_epoch=True)
        return logits
    
    def on_validation_epoch_end(self):
        if not self.val_predictions:
            return
        all_logits = torch.cat(self.val_predictions, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)
        import pdb; pdb.set_trace()

        num_videos = len(all_logits) // self.num_test_clips
        grouped_logits = all_logits.view(num_videos, self.num_test_clips, -1)  # [N, clips, classes]
        grouped_targets = all_targets.view(num_videos, self.num_test_clips)    # [N, clips]
        
        # Average predictions across clips
        avg_logits = grouped_logits.mean(dim=1)  # [N, classes]
        video_targets = grouped_targets[:, 0]    # [N] - same labels
        
        self.val_acc(avg_logits, video_targets)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        
        self.val_predictions.clear()
        self.val_targets.clear()

    def on_test_epoch_end(self):
        if not self.test_predictions:
            return
            
        all_logits = torch.cat(self.test_predictions, dim=0)
        all_targets = torch.cat(self.test_targets, dim=0)
        
        num_videos = len(all_logits) // self.num_test_clips
        grouped_logits = all_logits.view(num_videos, self.num_test_clips, -1)
        grouped_targets = all_targets.view(num_videos, self.num_test_clips)
        
        avg_logits = grouped_logits.mean(dim=1)
        video_targets = grouped_targets[:, 0]
        
        self.test_acc(avg_logits, video_targets)
        self.log('test_acc', self.test_acc, on_epoch=True)
        
        self.test_predictions.clear()
        self.test_targets.clear()    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }