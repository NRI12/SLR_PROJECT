import torch
import numpy as np
from omegaconf import OmegaConf
from src.models.architectures.hrnet import get_pose_net
import cv2
from torchvision import transforms
import os
import urllib.request
import ssl
from tqdm import tqdm
import glob
import hydra
from omegaconf import DictConfig
from datetime import datetime

ssl._create_default_https_context = ssl._create_unverified_context
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def process_npy(npy_path, device):
    img_data = np.load(npy_path)
    frames, h, w, c = img_data.shape
    processed = []
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for frame in img_data:
        img = cv2.resize(frame, (384, 288)).astype(np.float32) / 255.0
        img = normalize(torch.from_numpy(img).permute(2, 0, 1))
        processed.append(img)
    
    return torch.stack(processed).to(device)

def extract_keypoints(model, input_tensor):
    frames = input_tensor.shape[0]
    all_keypoints = []
    all_heatmaps = []
    
    with torch.no_grad():
        for i in range(frames):
            heatmaps = model(input_tensor[i:i+1])
            result = torch.argmax(heatmaps.reshape(133, -1), dim=1).cpu().numpy()
            
            h, w = heatmaps.shape[2:]
            y, x = result // w, result % w
            
            pred = np.zeros((133, 3), dtype=np.float32)
            pred[:, 0] = x * (384 / w)
            pred[:, 1] = y * (288 / h)
            
            for j in range(133):
                pred[j, 2] = heatmaps[0, j, y[j], x[j]].cpu().item()
            
            all_keypoints.append(pred)
            all_heatmaps.append(heatmaps[0].cpu().numpy())  # (133, H, W)
    
    return np.array(all_keypoints), np.array(all_heatmaps)

def process_file(args):
    npy_path, keypoint_output, heatmap_output, device = args
    log_file = f"script/logs/keypoint_extract_{datetime.now().strftime('%Y%m%d')}.log"
    
    try:
        model = load_model(device)
        input_tensor = process_npy(npy_path, device)
        keypoints, heatmaps = extract_keypoints(model, input_tensor)
        
        os.makedirs(os.path.dirname(keypoint_output), exist_ok=True)
        os.makedirs(os.path.dirname(heatmap_output), exist_ok=True)
        
        np.save(keypoint_output, keypoints)
        np.save(heatmap_output, heatmaps)
        
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - OK: {npy_path}\n")
        
        return True
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - ERR: {npy_path} - {e}\n")
        return False

def get_files(processed_output):
    files = []
    for class_name in os.listdir(processed_output):
        rgb_path = os.path.join(processed_output, class_name, "rgb")
        if os.path.exists(rgb_path):
            for npy in glob.glob(f"{rgb_path}/*.npy"):
                filename = os.path.basename(npy)
                keypoint_output = os.path.join(processed_output, class_name, "keypoint", filename)
                heatmap_output = os.path.join(processed_output, class_name, "heatmap", filename)
                files.append((npy, keypoint_output, heatmap_output))
    return files

@hydra.main(version_base=None, config_path="../../configs/data", config_name="dataset")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs("script/logs", exist_ok=True)
    
    files = get_files(cfg.processed_output)
    print(f"Found {len(files)} files")
    
    model = load_model(device)
    results = []
    
    for npy_path, keypoint_output, heatmap_output in tqdm(files, desc="Processing"):
        try:
            input_tensor = process_npy(npy_path, device)
            keypoints, heatmaps = extract_keypoints(model, input_tensor)
            
            os.makedirs(os.path.dirname(keypoint_output), exist_ok=True)
            os.makedirs(os.path.dirname(heatmap_output), exist_ok=True)
            
            np.save(keypoint_output, keypoints)
            np.save(heatmap_output, heatmaps)
            results.append(True)
            
            log_file = f"script/logs/keypoint_extract_{datetime.now().strftime('%Y%m%d')}.log"
            with open(log_file, "a") as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - OK: {npy_path}\n")
        except Exception as e:
            results.append(False)
            with open(log_file, "a") as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - ERR: {npy_path} - {e}\n")
    
    success = sum(results)
    print(f"Done: {success}/{len(results)} successful")

if __name__ == "__main__":
    main()