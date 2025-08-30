import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class VideoProcessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.source_path = Path(cfg.source_path)
        self.output_path = Path(cfg.processed_output)
        
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        return np.array(frames)
    
    def process_single_video(self, row):
        video_path = self.source_path / row['video_path']
        output_file = self.output_path / row['video_path'].replace('.avi', '.npy')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.exists():
            return
            
        try:
            frames = self.load_video_frames(video_path)
            np.save(output_file, frames)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    def process_dataset(self):
        annotations_file = Path(self.cfg.output_path) / "dataset.csv"
        df = pd.read_csv(annotations_file)
        
        print(f"Processing {len(df)} videos...")
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(tqdm(executor.map(self.process_single_video, [row for _, row in df.iterrows()]), total=len(df)))
        
        print(f"Processed videos saved to: {self.output_path}")


@hydra.main(version_base=None, config_path="../../configs/data", config_name="dataset")
def main(cfg: DictConfig):
    processor = VideoProcessor(cfg)
    processor.process_dataset()


if __name__ == "__main__":
    main()