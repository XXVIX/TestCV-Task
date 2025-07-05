import cv2
import albumentations as A
import argparse
from pathlib import Path

def get_annotations(label_path, img_height, img_width):
    if not label_path.exists():
        return [], []
    
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_c, y_c, w, h = [float(p) for p in parts[1:5]]
            
            box_w = w * img_width
            box_h = h * img_height
            x_min = x_c * img_width - box_w / 2
            y_min = y_c * img_height - box_h / 2
            x_max = x_min + box_w
            y_max = y_min + box_h
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            
            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(class_id)
            
    return bboxes, class_labels

def save_annotations(label_path, augmented_data, img_height, img_width):
    with open(label_path, 'w') as f:
        if not augmented_data['bboxes']:
            return
            
        for i in range(len(augmented_data['bboxes'])):
            class_id = augmented_data['class_labels'][i]
            x_min, y_min, x_max, y_max = augmented_data['bboxes'][i]
            
            w = (x_max - x_min)
            h = (y_max - y_min)
            x_c = (x_min + w / 2)
            y_c = (y_min + h / 2)

            f.write(f"{class_id} {x_c/img_width:.6f} {y_c/img_height:.6f} {w/img_width:.6f} {h/img_height:.6f}\n")


def augment_dataset(dataset_dir: Path, num_reps: int):
    train_image_dir = dataset_dir / "images" / "train"
    train_label_dir = dataset_dir / "labels" / "train"

    if not train_image_dir.is_dir() or not train_label_dir.is_dir():
        raise FileNotFoundError(f"Training directories not found in {dataset_dir}")

    image_files = list(train_image_dir.glob('*.png')) + list(train_image_dir.glob('*.jpg')) + list(train_image_dir.glob('*.jpeg'))

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4, brightness_limit=0.2, contrast_limit=0.2),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))

    for image_path in image_files:
        if "_aug_" in image_path.stem:
            continue

        label_path = train_label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        bboxes, class_labels = get_annotations(label_path, h, w)
        
        if not bboxes:
            continue
            
        for i in range(num_reps):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                if not augmented['bboxes']:
                    continue

                aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                
                new_img_path = train_image_dir / f"{image_path.stem}_aug_{i}.png"
                new_label_path = train_label_dir / f"{image_path.stem}_aug_{i}.txt"
                
                cv2.imwrite(str(new_img_path), aug_img)
                save_annotations(new_label_path, augmented, h, w)
            except Exception:
                # Silently ignore errors for single image augmentation
                continue

def main():
    parser = argparse.ArgumentParser(description="Apply offline Albumentations to the training split of a YOLO dataset.")
    parser.add_argument("--dataset_dir", type=Path, default="data/food_dataset")
    parser.add_argument("--reps", type=int, default=4)
    args = parser.parse_args()
    
    augment_dataset(args.dataset_dir, args.reps)
    
    print("Script finished.")

if __name__ == "__main__":
    main()