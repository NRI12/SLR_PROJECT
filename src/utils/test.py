import numpy as np
import cv2
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from models.hrnet import get_pose_net
import urllib.request
import os
import glob

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k[9:]] = v
        elif k.startswith("keypoint_head."):
            new_state_dict[k[14:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_model(device):
    weights_path = "checkpoint/hrnet_w48_wholebody.pth"
    os.makedirs("checkpoint", exist_ok=True)
    
    if not os.path.exists(weights_path):
        urllib.request.urlretrieve("https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth", weights_path)
    
    cfg = OmegaConf.load("../configs/model/hrnet/hrnet_w48.yaml")
    model = get_pose_net(cfg, is_train=False)
    
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = clean_state_dict(checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)
normalize = transforms.Normalize(mean=[0.185, 0.156, 0.106], std=[0.229, 0.224, 0.225])

# Create output folder
os.makedirs("output_kvk", exist_ok=True)

image_files = glob.glob("input_kvk/*.jpg") + glob.glob("input_kvk/*.png") + glob.glob("input_kvk/*.jpeg")

for img_path in image_files:
    print(f"Processing: {img_path}")
    
    # Process image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (384, 288)).astype(np.float32) / 255.0
    input_tensor = normalize(torch.from_numpy(img_resized).permute(2, 0, 1)).unsqueeze(0).to(device)
    
    # Extract keypoints
    with torch.no_grad():
        heatmaps = model(input_tensor)
        result = torch.argmax(heatmaps.reshape(133, -1), dim=1).cpu().numpy()
        h, w = heatmaps.shape[2:]
        y, x = result // w, result % w
        
        kpts = np.zeros((133, 3), dtype=np.float32)
        kpts[:, 0] = x * (384 / w)
        kpts[:, 1] = y * (288 / h)
        for j in range(133):
            kpts[j, 2] = heatmaps[0, j, y[j], x[j]].cpu().item()
    
    # Scale and draw
    orig_h, orig_w = img.shape[:2]
    scale_x = orig_w / 384
    scale_y = orig_h / 288
    
    # Define pairs for connections
    pairs = [(5,6),(5,7),(6,8),(8,10),(7,9),(9,91),(10,112),
            (91,92),(91,95),(91,96),(91,99),(91,100),(91,103),(91,104),(91,107),(91,108),(91,111),
            (112,113),(112,116),(112,117),(112,120),(112,121),(112,124),(112,125),(112,128),(112,129),(112,132)]


    for idx in range(len(kpts)):
        x, y, conf = kpts[idx]
        if conf > 0.1:
            x_orig, y_orig = int(x*scale_x), int(y*scale_y)
            cv2.circle(img, (x_orig, y_orig), 2, (0, 255, 0), -1)
        cv2.putText(img, str(idx+1), (x_orig+3, y_orig-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    # Draw connections
    for p1, p2 in pairs:
        x1, y1, c1 = kpts[p1]
        x2, y2, c2 = kpts[p2]
        if c1 > 0.1 and c2 > 0.1:
            x1_orig, y1_orig = int(x1*scale_x), int(y1*scale_y)
            x2_orig, y2_orig = int(x2*scale_x), int(y2*scale_y)
            cv2.line(img, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 0, 255), 2)    
    # Save output
    filename = os.path.basename(img_path)
    output_path = f"output_kvk/{filename}"
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

print("All images processed!")