import sys
sys.path.append('./src')

import torch
from data.datamodules import SLRDataModule

dm = SLRDataModule(
    annotation_file="/home/huong.nguyenthi2/SLR_PROJECT/data/annotation/dataset.csv",
    data_root="./data/processed",
    modalities=['rgb'],
    batch_size=4,
    video_size=(224, 224),
    output_format='CTHW',
    max_frames=16,
    num_workers=2
)

dm.setup()

print(f"Classes: {dm.num_classes}")
print(f"Train/Val/Test: {len(dm.train_dataloader())}/{len(dm.val_dataloader())}/{len(dm.test_dataloader())}")

batch = next(iter(dm.train_dataloader()))
print(f"Batch keys: {list(batch.keys())}")
print(f"RGB: {batch['rgb'].shape}")
print(f"Labels: {batch['sign_id'].shape}")

batch_multi = SLRDataModule(
    annotation_file="./data/annotation/dataset.csv", 
    data_root="./data/processed",
    modalities=['rgb', 'keypoint', 'heatmap', 'optical_flow'],
    batch_size=2,
    max_frames=32
)
batch_multi.setup()
sample = next(iter(batch_multi.train_dataloader()))

for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")

print("✅ All tests passed!")