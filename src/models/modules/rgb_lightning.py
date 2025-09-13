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
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        x = self._random_temporal_crop(x, self.target_frames)
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        logits = self._multi_clip_inference(x, self.target_frames)
        loss = self.criterion(logits, y)
        
        self.val_acc(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        logits = self._multi_clip_inference(x, self.target_frames)
        loss = self.criterion(logits, y)
        
        self.test_acc(logits, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        
        return logits
    
    def _random_temporal_crop(self, x, target_frames):
        B, C, T, H, W = x.shape
        if T < target_frames:
            repeat_factor = (target_frames + T - 1) // T
            x = x.repeat(1, 1, repeat_factor, 1, 1)
            T = x.shape[2]
        
        start = torch.randint(0, T - target_frames + 1, (1,)).item()
        return x[:, :, start:start+target_frames, :, :]
    
    def _multi_clip_inference(self, x, target_frames):
        B, C, T, H, W = x.shape
        
        if T < target_frames:
            repeat_factor = (target_frames + T - 1) // T
            x = x.repeat(1, 1, repeat_factor, 1, 1)
            T = x.shape[2]
        
        if T <= target_frames:
            return self(x[:, :, :target_frames, :, :])
        
        clip_logits = []
        if T <= target_frames * self.num_test_clips:
            step = max(1, (T - target_frames) // (self.num_test_clips - 1))
            for i in range(self.num_test_clips):
                start = min(i * step, T - target_frames)
                clip = x[:, :, start:start+target_frames, :, :]
                logits = self(clip)
                clip_logits.append(logits)
        else:
            step = T // self.num_test_clips
            for i in range(self.num_test_clips):
                start = i * step
                end = min(start + target_frames, T)
                if end - start < target_frames:
                    start = max(0, end - target_frames)
                clip = x[:, :, start:start+target_frames, :, :]
                logits = self(clip)
                clip_logits.append(logits)
        
        return torch.stack(clip_logits).mean(dim=0)
    
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