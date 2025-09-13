import sys
sys.path.append('./src')
from data.datamodules import SLRDataModule

dm = SLRDataModule(
    annotation_file="/home/huong.nguyenthi2/SLR_PROJECT/data/annotation/dataset.csv",
    data_root="/home/huong.nguyenthi2/SLR_PROJECT/data/processed",
    modalities=['rgb'],
    batch_size=4,
    video_size=(224, 224),
    max_frames=16,
    num_workers=4,
    output_format="CTHW"  # (C, T, H, W)
)

dm.setup()

print(f"Classes: {dm.num_classes}")
print(f"Train/Val/Test: {len(dm.train_dataloader())}/{len(dm.val_dataloader())}/{len(dm.test_dataloader())}")

batch = next(iter(dm.train_dataloader()))
print(f"Batch keys: {list(batch.keys())}")
print(f"RGB: {batch['rgb'].shape}")
print(f"Labels: {batch['sign_id']}")


from models.architectures.r2plus1d import R2Plus1DModel
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
model = R2Plus1DModel(num_classes=dm.num_classes)
for batch in dm.val_dataloader():
    videos = batch["rgb"]              # (B, 3, T, H, W) 
    labels = batch["sign_id"]          # (B,)
    person_ids = batch["person_id"]    # (B,) -> optional
    
    logits = model(videos)
    loss = criterion(logits, labels)
    print(f"Videos: {videos.shape}, Labels: {labels.shape}, Person IDs: {person_ids.shape}")
    print(videos, labels, person_ids)
    print(f"Logits: {logits.shape}, Loss: {loss.item()}")
    break
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