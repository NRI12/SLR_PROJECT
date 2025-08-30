import numpy as np
import cv2
import os
import glob
from utils.constraint import selected_joints,hand_pairs,hands
import random

def visualize_keypoints(data_path,keypoint_type=None,filename=None):
    rgb_path = f"{data_path}/rgb"
    keypoint_path = f"{data_path}/keypoint"
    
    if filename is None:
        filename = os.path.basename(glob.glob(f"{rgb_path}/*.npy")[0])
    
    rgb = np.load(f"{rgb_path}/{filename}")
    keypoints = np.load(f"{keypoint_path}/{filename}")
    
    mid_frame = len(rgb) // 2
    img = rgb[mid_frame].copy()
    kpts = keypoints[mid_frame]
    
    # Scale size origional
    orig_h, orig_w = img.shape[:2]
    scale_x = orig_w / 384
    scale_y = orig_h / 288
    if keypoint_type == "hands":
        select_indices = hands
    elif keypoint_type:
        select_indices = selected_joints[keypoint_type]
    else:
        select_indices = range(len(kpts))
    for idx in select_indices:
        if idx < len(kpts):
            x,y,conf = kpts[idx]
            if conf > 0.3:
                x_orig = int(x*scale_x)
                y_orig = int(y*scale_y)
                cv2.circle(img, (x_orig, y_orig), 3, (0, 255, 0), -1)
                cv2.putText(img, str(idx), (x_orig, y_orig), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    if keypoint_type == "hands":
        for p1, p2 in hand_pairs:
            if p1 < len(kpts) and p2 < len(kpts):
                x1, y1, c1 = kpts[p1]
                x2, y2, c2 = kpts[p2]
                if c1 > 0.8 and c2 > 0.8:
                    x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
                    x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
                    cv2.line(img, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 255, 0), 2)

    output = f"visual_{filename.replace('.npy', '.png')}"
    cv2.imwrite(output, img)
    print(f"Saved: {output}")

folders = glob.glob("/home/huong.nguyenthi2/SLR_PROJECT/data/processed/*/")
data_path = random.choice(folders)
visualize_keypoints(data_path,"hands")