import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from src.models.architectures.r2plus1d import R2Plus1DModel
from collections import defaultdict

class RGBLightningModule(pl.LightningModule):
    def __init__(self, num_classes=100, learning_rate=1e-3, weight_decay=1e-4, 
                 num_test_clips=5, target_frames=16):
        super().__init__()
        self.save_hyperparameters()
        self.model = R2Plus1DModel(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.num_test_clips = num_test_clips
        self.target_frames = target_frames

        self.train_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.train_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_clip_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_clip_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_clip_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.test_clip_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

        self.val_predictions = []
        self.test_predictions = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_top1', self.train_top1(logits, y), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_top5', self.train_top5(logits, y), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        video_idx = batch['video_idx']
        logits = self(x)
        loss = self.criterion(logits, y)
        for i in range(len(logits)):
            self.val_predictions.append({'video_idx': video_idx[i].item(),'logits': logits[i],'target': y[i].item()})
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_clip_top1', self.val_clip_top1(logits, y), on_epoch=True, sync_dist=True)
        self.log('val_clip_top5', self.val_clip_top5(logits, y), on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['rgb'], batch['sign_id']
        video_idx = batch['video_idx']
        logits = self(x)
        loss = self.criterion(logits, y)
        for i in range(len(logits)):
            self.test_predictions.append({'video_idx': video_idx[i].item(),'logits': logits[i],'target': y[i].item()})
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        self.log('test_clip_top1', self.test_clip_top1(logits, y), on_epoch=True, sync_dist=True)
        self.log('test_clip_top5', self.test_clip_top5(logits, y), on_epoch=True, sync_dist=True)
        return logits

    def on_validation_epoch_end(self):
        if not self.val_predictions:
            return
        video_predictions = defaultdict(list)
        video_targets = {}
        for pred in self.val_predictions:
            video_idx = pred['video_idx']
            video_predictions[video_idx].append(pred['logits'])
            video_targets[video_idx] = pred['target']
        video_logits = []
        video_labels = []
        for video_idx, clip_logits in video_predictions.items():
            avg_logits = torch.stack(clip_logits).mean(dim=0)
            video_logits.append(avg_logits)
            video_labels.append(video_targets[video_idx])
        if video_logits:
            video_logits = torch.stack(video_logits)
            video_labels = torch.tensor(video_labels, device=self.device)
            val_top1 = MulticlassAccuracy(num_classes=self.hparams.num_classes, top_k=1).to(self.device)
            val_top5 = MulticlassAccuracy(num_classes=self.hparams.num_classes, top_k=5).to(self.device)
            video_top1 = val_top1(video_logits, video_labels)
            video_top5 = val_top5(video_logits, video_labels)
            self.log('val_top1', video_top1, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_top5', video_top5, on_epoch=True, sync_dist=True)
            self.log('val_num_videos', len(video_predictions), on_epoch=True)
            self.log('val_clips_per_video', len(self.val_predictions) / len(video_predictions), on_epoch=True)
        self.val_predictions.clear()

    def on_test_epoch_end(self):
        if not self.test_predictions:
            return
        video_predictions = defaultdict(list)
        video_targets = {}
        for pred in self.test_predictions:
            video_idx = pred['video_idx']
            video_predictions[video_idx].append(pred['logits'])
            video_targets[video_idx] = pred['target']
        video_logits = []
        video_labels = []
        for video_idx, clip_logits in video_predictions.items():
            avg_logits = torch.stack(clip_logits).mean(dim=0)
            video_logits.append(avg_logits)
            video_labels.append(video_targets[video_idx])
        if video_logits:
            video_logits = torch.stack(video_logits)
            video_labels = torch.tensor(video_labels, device=self.device)
            test_top1 = MulticlassAccuracy(num_classes=self.hparams.num_classes, top_k=1).to(self.device)
            test_top5 = MulticlassAccuracy(num_classes=self.hparams.num_classes, top_k=5).to(self.device)
            video_top1 = test_top1(video_logits, video_labels)
            video_top5 = test_top5(video_logits, video_labels)
            self.log('test_top1', video_top1, on_epoch=True, sync_dist=True)
            self.log('test_top5', video_top5, on_epoch=True, sync_dist=True)
            self.log('test_num_videos', len(video_predictions), on_epoch=True)
            self.log('test_clips_per_video', len(self.test_predictions) / len(video_predictions), on_epoch=True)
        self.test_predictions.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
