import os
import re
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig

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

    person_counts = {
        35: 10, 34: 10, 31: 10, 29: 10, 30: 10, 42: 10, 223: 10,
        28: 20, 26: 40, 23: 90, 24: 90, 25: 120, 22: 157,
        21: 1194, 15: 1303, 11: 1335, 19: 1356, 12: 1367, 14: 1391,
        20: 1412, 9: 1416, 13: 1429, 4: 1441, 8: 1463, 5: 1464,
        18: 1483, 7: 1573, 10: 1577, 6: 1587, 17: 1671, 16: 1722,
        1: 1830, 3: 1902, 2: 2003
    }
    
    total_samples = len(df)
    target_train = int(0.8 * total_samples)
    target_val = int(0.1 * total_samples)
    target_test = int(0.1 * total_samples)
    
    persons_sorted = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)
    
    train_persons = []
    val_persons = []
    test_persons = []
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for person_id, count in persons_sorted:
        if train_count < target_train:
            train_persons.append(person_id)
            train_count += count
        elif val_count < target_val:
            val_persons.append(person_id)
            val_count += count
        else:
            test_persons.append(person_id)
            test_count += count
    
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
    
    # Print final stats
    actual_counts = df['split'].value_counts()
    print(f"\nFinal Split Distribution:")
    print(f"Train: {actual_counts['train']} samples ({actual_counts['train']/len(df):.1%})")
    print(f"Val: {actual_counts['val']} samples ({actual_counts['val']/len(df):.1%})")
    print(f"Test: {actual_counts['test']} samples ({actual_counts['test']/len(df):.1%})")

if __name__ == "__main__":
    scan_dataset()