import argparse
import requests
from pathlib import Path
from ultralytics import YOLO

def download_model_weights(url: str, output_path: Path):
    if output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    except requests.exceptions.RequestException as e:
        if output_path.exists():
            output_path.unlink()
        raise e

def train_model(model_path: Path, config_path: str, epochs: int, batch_size: int, img_size: int, exp_name: str):
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = YOLO(model_path)
    
    model.train(
        data=config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=25,
        project="runs/detect",
        name=exp_name,
        verbose=False
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data/dataset_configs/food_dataset.yaml")
    parser.add_argument("--model_url", type=str, default="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt")
    parser.add_argument("--model_filename", type=str, default="yolo11m.pt")
    parser.add_argument("--weights_dir", type=Path, default="weights")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--exp_name", type=str, default="experiment_yolo11m")
    args = parser.parse_args()

    local_model_path = args.weights_dir / args.model_filename
    
    download_model_weights(args.model_url, local_model_path)
    
    train_model(local_model_path, args.config, args.epochs, args.batch_size, args.img_size, args.exp_name)

    print("Script finished.")

if __name__ == "__main__":
    main()