import argparse
from pathlib import Path
import random
import shutil

def split_data(items, train_ratio, val_ratio):
    random.shuffle(items)
    total_items = len(items)
    
    train_end = int(total_items * train_ratio)
    val_end = train_end + int(total_items * val_ratio)
    
    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]
    
    return train_items, val_items, test_items

def organize_files(image_dir: Path, label_dir: Path, dest_dir: Path, train_ratio: float, val_ratio: float):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_stems = {f.stem for f in image_dir.glob('*.*')}
    label_stems = {f.stem for f in label_dir.glob('*.txt')}
    
    common_stems = list(image_stems.intersection(label_stems))

    if not common_stems:
        raise FileNotFoundError(f"No matching image/label pairs found between {image_dir} and {label_dir}")

    train_stems, val_stems, test_stems = split_data(common_stems, train_ratio, val_ratio)
    
    split_map = {
        "train": train_stems,
        "val": val_stems,
        "test": test_stems
    }

    for split_name, stems in split_map.items():
        img_path_out = dest_dir / "images" / split_name
        lbl_path_out = dest_dir / "labels" / split_name
        img_path_out.mkdir(parents=True, exist_ok=True)
        lbl_path_out.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            try:
                img_file = next(image_dir.glob(f'{stem}.*'))
                lbl_file = label_dir / f'{stem}.txt'
                
                shutil.copy(str(img_file), str(img_path_out / img_file.name))
                shutil.copy(str(lbl_file), str(lbl_path_out / lbl_file.name))
            except StopIteration:
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, default="data/raw_frames")
    parser.add_argument("--label_dir", type=Path, default="data/labels")
    parser.add_argument("--output_dir", type=Path, default="data/food_dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    if not (0 <= (args.train_ratio + args.val_ratio) <= 1):
        raise ValueError("Sum of train and val ratios must be between 0 and 1.")

    organize_files(args.image_dir, args.label_dir, args.output_dir, args.train_ratio, args.val_ratio)
    
    print("Script finished.")

if __name__ == "__main__":
    main()