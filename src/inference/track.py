import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

def process_video_with_tracking(model_path: Path, video_path: Path, output_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found at {video_path}")

    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(str(video_path))
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results and results[0].boxes.id is not None:
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        else:
            out.write(frame)

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="reports/tracked_video.mp4")
    args = parser.parse_args()

    process_video_with_tracking(args.model, args.video, args.output)
    
    print("Script finished.")

if __name__ == "__main__":
    main()