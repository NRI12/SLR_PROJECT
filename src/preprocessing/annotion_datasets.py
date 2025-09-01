import os
import re
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

@hydra.main(version_base=None, config_path="../../configs/data", config_name="dataset")
def scan_dataset(cfg: DictConfig):
    source_path = Path(cfg.source_path)
    output_path = Path(cfg.output_path)
    
    os.makedirs(output_path, exist_ok=True)


    samples = []
    person_folders = sorted([d for d in source_path.iterdir() 
                           if d.is_dir() and re.match(r'A\d+P\d+', d.name)])
    for person_folder in person_folders:
        person_id = person_folder.name
        rgb_folder = person_folder / "rgb"

        for video_file in rgb_folder.glob("*.avi"):
            match = re.match(r'\d+_A(\d+)P(\d+)_', video_file.name)
            # import pdb; pdb.set_trace()
            if match:
                sign_id = int(match.group(1))
                person_num = int(match.group(2))
                samples.append({
                    'video_path': str(video_file.relative_to(source_path)),
                    'sign_id': sign_id,
                    'person_id': person_num
                })
    df = pd.DataFrame(samples)
    unique_persons = sorted(df['person_id'].unique())

    train_persons ,temp = train_test_split(unique_persons, 
        train_size=cfg.train_ratio, 
        random_state=42, 
        shuffle=True
    )
    val_persons, test_persons = train_test_split(temp,
        train_size=cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio), 
        random_state=42, 
        shuffle=True
    )
    for split_name, persons in [('Train', train_persons), ('Val', val_persons), ('Test', test_persons)]:
        print(f"{split_name}: {sorted(persons)}")
    df['split'] = df['person_id'].apply(
        lambda x: 'train' if x in train_persons else
                  ('val' if x in val_persons else 'test')
    )
    df[["video_path", "sign_id", "person_id", "split"]].to_csv(
        output_path / "dataset.csv", index=False
    )
    print(f"Total: {len(df)} samples")
    print(f"Signs: {df['sign_id'].nunique()}")
    print(f"Persons: {df['person_id'].nunique()}")
    print("Split:", df['split'].value_counts().to_dict())
    print(f"Saved: {output_path / 'dataset.csv'}")
if __name__ == "__main__":
    scan_dataset()