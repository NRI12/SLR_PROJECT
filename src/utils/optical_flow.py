import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision.models.optical_flow import raft_large
import urllib.request
import os
from tqdm import tqdm

torch.cuda.set_device(7)

checkpoint_dir = "/home/huong.nguyenthi2/SLR_PROJECT/src/checkpoint"
checkpoint_path = os.path.join(checkpoint_dir, "raft_large_C_T_SKHT_V2-ff5fadd5.pth")

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_dir, exist_ok=True)
    urllib.request.urlretrieve("https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth", checkpoint_path)

raft = raft_large(pretrained=False)
raft.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
raft = raft.eval().cuda()
file_path = "/home/huong.nguyenthi2/SLR_PROJECT/data/processed/A2P20/rgb/596_A2P20_.npy"
frames = np.load(file_path)
print(f"Total frames: {len(frames)}")

def preprocess(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0) / 255.0
    return img.cuda()

def flow_to_color(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

output_path = "/home/huong.nguyenthi2/SLR_PROJECT/src/results/optical_flow_58_A1P3.avi"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

flow_frames = []

for i in tqdm(range(len(frames)-1)):
    frame1, frame2 = frames[i], frames[i+1]
    image1, image2 = preprocess(frame1), preprocess(frame2)
    
    with torch.no_grad():
        flow_predictions = raft(image1, image2)
        flow_up = flow_predictions[-1]
    
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    flow_frames.append(flow)
    
    flow_color = flow_to_color(flow)
    flow_bgr = cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR)
    out.write(flow_bgr)

out.release()
np.save("/home/huong.nguyenthi2/SLR_PROJECT/src/results/final.npy", np.array(flow_frames))

print(f"Video saved: {output_path}")
