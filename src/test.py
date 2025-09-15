import sys
sys.path.append('./src')
import pytorch_lightning as pl
from data.datamodules import SLRDataModule
from models.modules.slgcn_lightning import SLGCNLightningModule

dm = SLRDataModule(
    annotation_file="/home/huong.nguyenthi2/SLR_PROJECT/data/annotation/dataset.csv",
    data_root="/home/huong.nguyenthi2/SLR_PROJECT/data/processed",
    modalities=['keypoint'],
    batch_size=8,
    num_workers=4,
    target_frames=64,
)
dm.setup()

model = SLGCNLightningModule(num_classes=dm.num_classes)

trainer = pl.Trainer(
    max_epochs=1,
    accelerator='auto',
    devices=1,
    log_every_n_steps=10,
)

trainer.fit(model, dm)
# print(f"Batch keys: {list(batch.keys())}")
# print(f"RGB: {batch['keypoint'].shape}")
# print(f"Labels: {batch['sign_id'].shape}")
# batch_multi = SLRDataModule(
#     annotation_file="/home/huong.nguyenthi2/SLR_PROJECT/data/annotation/dataset.csv",
#     data_root="/home/huong.nguyenthi2/SLR_PROJECT/data/processed",
#     modalities=['rgb', 'keypoint', 'heatmap', 'optical_flow'],
#     batch_size=2,
#     max_frames=16
# )
# batch_multi.setup()
# sample = next(iter(batch_multi.train_dataloader()))

# for k, v in sample.items():
#     if isinstance(v, torch.Tensor):
#         print(f"{k}: {v.shape}")

# print("All tests passed!")