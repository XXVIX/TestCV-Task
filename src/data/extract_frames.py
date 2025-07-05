import cv2
import argparse
from pathlib import Path
import numpy as np

def calculate_frame_diff(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresholded)
    return non_zero_count / (frame1.shape[0] * frame1.shape[1])

def process_video(video_path: Path, output_dir: Path, threshold: float, min_interval_sec: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        return

    min_frame_interval = int(fps * min_interval_sec)
    
    last_saved_frame = None
    last_saved_frame_pos = -min_frame_interval
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if last_saved_frame is None or (frame_count - last_saved_frame_pos) >= min_frame_interval:
            should_save = True
            if last_saved_frame is not None:
                diff_ratio = calculate_frame_diff(frame, last_saved_frame)
                if diff_ratio < threshold:
                    should_save = False
            
            if should_save:
                frame_filename = output_dir / f"{video_path.stem}_frame_{frame_count:05d}.png"
                cv2.imwrite(str(frame_filename), frame)
                last_saved_frame = frame.copy()
                last_saved_frame_pos = frame_count
                
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default="data/raw_frames")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--interval", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(args.input_dir.glob('*.mov')) + list(args.input_dir.glob('*.mp4'))
    if not video_files:
        raise FileNotFoundError(f"No .mov or .mp4 files found in {args.input_dir}")

    for video_path in video_files:
        process_video(video_path, args.output_dir, args.threshold, args.interval)

    print("Script finished.")

if __name__ == "__main__":
    main()