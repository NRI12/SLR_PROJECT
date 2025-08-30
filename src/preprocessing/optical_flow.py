import hydra
from omegaconf import DictConfig
import torch
from torchvision.models.optical_flow import raft_large
import urllib.request
import os
from tqdm import tqdm
import glob
import numpy as np
from hydra.core.hydra_config import HydraConfig
from datetime import datetime
torch.cuda.set_device(7)

def process_files_batch(file_list, raft, optical_flow_dirs, batch_size=8, log_file=None):
    def preprocess(img):
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img.cuda()
    
    all_pairs = []
    file_frame_mapping = []
    
    for idx, (npy_file, flow_dir) in enumerate(zip(file_list, optical_flow_dirs)):
        frames = np.load(npy_file)
        if len(frames) < 2:
            continue
            
        for i in range(len(frames)-1):
            all_pairs.append((preprocess(frames[i]), preprocess(frames[i+1])))
            file_frame_mapping.append((idx, i))
    
    # Process in batches
    results = {}
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Batch processing", leave=False):
        batch_pairs = all_pairs[i:i+batch_size]
        
        if not batch_pairs:
            continue
            
        # Create batched tensors
        images1 = torch.stack([pair[0] for pair in batch_pairs])
        images2 = torch.stack([pair[1] for pair in batch_pairs])
        
        with torch.no_grad():
            flows = raft(images1, images2)[-1]
        
        # Store results
        for j, (file_idx, frame_idx) in enumerate(file_frame_mapping[i:i+len(batch_pairs)]):
            if file_idx not in results:
                results[file_idx] = {}
            results[file_idx][frame_idx] = flows[j].permute(1, 2, 0).cpu().numpy()
    
    # Save results
    with open(log_file, 'a') as f:
        for file_idx, frame_flows in results.items():
            if not frame_flows:
                continue
                
            # Sort by frame index and convert to array
            sorted_flows = [frame_flows[i] for i in sorted(frame_flows.keys())]
            
            npy_file = file_list[file_idx]
            flow_dir = optical_flow_dirs[file_idx]
            output_path = os.path.join(flow_dir, os.path.basename(npy_file))
            np.save(output_path, np.array(sorted_flows))
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {output_path} -> oke\n")
            f.flush()

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    
    log_file = os.path.join(output_dir, "optical_flow.log")
    
    torch.cuda.set_device(cfg.gpu_id)
    
    checkpoint_path = os.path.join(cfg.checkpoint.dir, cfg.checkpoint.filename)
    if not os.path.exists(checkpoint_path):
        os.makedirs(cfg.checkpoint.dir, exist_ok=True)
        urllib.request.urlretrieve(cfg.checkpoint.url, checkpoint_path)
    
    raft = raft_large(pretrained=False)
    raft.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    raft = raft.eval().cuda()
    
    base_dir = cfg.data.processed_output
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for folder in tqdm(folders, desc="Processing folders"):
        rgb_dir = os.path.join(base_dir, folder, "rgb")
        optical_flow_dir = os.path.join(base_dir, folder, cfg.optical_flow.output_folder)
        
        if not os.path.exists(rgb_dir):
            continue
        os.makedirs(optical_flow_dir, exist_ok=True)
        
        npy_files = glob.glob(os.path.join(rgb_dir, "*.npy"))
        if not npy_files:
            continue
            
        tqdm.write(f"Processing {folder}: {len(npy_files)} files")
        
        optical_flow_dirs = [optical_flow_dir] * len(npy_files)
        process_files_batch(npy_files, raft, optical_flow_dirs, batch_size=16, log_file=log_file)

if __name__ == "__main__":
    main()