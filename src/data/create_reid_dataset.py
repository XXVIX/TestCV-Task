import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil

def process_single_video(model, video_path: Path, output_dir: Path, tracker_config: Path, sample_rate_sec: float):
    video_stem = video_path.stem
    class_names = model.names
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_rate_frames = int(fps * sample_rate_sec)

    results_generator = model.track(
        source=str(video_path),
        persist=True,
        tracker=str(tracker_config),
        stream=True,
        verbose=False
    )
    
    last_saved_frame = {}

    for frame_idx, results in enumerate(results_generator):
        if results.boxes.id is None:
            continue
        
        frame = results.orig_img
        track_ids = results.boxes.id.int().cpu().tolist()
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results.boxes.cls.int().cpu().tolist()

        for i in range(len(track_ids)):
            if class_names[cls_ids[i]] != 'cutlery':
                continue

            track_id = track_ids[i]
            box = boxes[i]
            global_track_id_str = f"{video_stem}_{track_id}"

            if frame_idx >= last_saved_frame.get(global_track_id_str, 0) + sample_rate_frames:
                last_saved_frame[global_track_id_str] = frame_idx
                
                track_dir = output_dir / global_track_id_str
                track_dir.mkdir(exist_ok=True)
                
                x1, y1, x2, y2 = box
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                cropped_obj = frame[y1:y2, x1:x2]
                output_filename = track_dir / f"frame_{frame_idx}.png"
                cv2.imwrite(str(output_filename), cropped_obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default="data/reid_dataset_raw")
    parser.add_argument("--tracker_config", type=Path, default="config/trackers/botsort.yaml")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    video_files = sorted(list(args.video_dir.glob('*.mov')) + list(args.video_dir.glob('*.mp4')))
    
    if not video_files:
        raise FileNotFoundError(f"No .mov or .mp4 files found in {args.video_dir}")

    model = YOLO(args.model)
    
    for video_path in video_files:
        process_single_video(model, video_path, args.output_dir, args.tracker_config, args.interval)
            
    print("Script finished.")

if __name__ == "__main__":
    main()