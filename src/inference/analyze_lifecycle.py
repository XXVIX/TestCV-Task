import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import torch
import torchreid
from scipy.spatial.distance import cdist

def create_id_merge_map(boxes, class_names, id_map, distance_thresh):
    if boxes is None or boxes.id is None:
        return id_map
        
    ids = boxes.id.int().cpu().tolist()
    cls_ids = boxes.cls.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().numpy()
    
    cutlery_indices = np.array([i for i, cid in enumerate(cls_ids) if class_names[cid] == 'cutlery'])
    
    if len(cutlery_indices) < 2:
        return id_map

    cutlery_boxes = xyxy[cutlery_indices]
    cutlery_ids = np.array(ids)[cutlery_indices]
    
    centers = np.vstack([(cutlery_boxes[:, 0] + cutlery_boxes[:, 2]) / 2, (cutlery_boxes[:, 1] + cutlery_boxes[:, 3]) / 2]).T
    dist_matrix = cdist(centers, centers)
    
    x_min1, x_max1 = cutlery_boxes[:, 0], cutlery_boxes[:, 2]
    x_min2, x_max2 = x_min1[:, np.newaxis], x_max1[:, np.newaxis]
    overlap_matrix = (x_min2 < x_max1) & (x_max2 > x_min1)
    
    potential_merges = (dist_matrix < distance_thresh) & overlap_matrix
    np.fill_diagonal(potential_merges, False)
    
    merge_indices = np.where(potential_merges)
    
    for i, j in zip(*merge_indices):
        id1, id2 = cutlery_ids[i], cutlery_ids[j]
        master_id, slave_id = min(id1, id2), max(id1, id2)
        canonical_master_id = id_map.get(master_id, master_id)
        id_map[slave_id] = canonical_master_id
        
    return id_map

def draw_custom_annotations(frame, boxes, id_merge_map, class_names):
    if boxes.id is None:
        return frame
    
    unique_tracks = {}
    for i in range(len(boxes.id)):
        track_id = boxes.id.int().cpu().tolist()[i]
        conf = boxes.conf.cpu().tolist()[i]
        
        if track_id not in unique_tracks or conf > unique_tracks[track_id]['conf']:
            unique_tracks[track_id] = {
                'box': boxes.xyxy[i].cpu().numpy().astype(int),
                'cls_id': boxes.cls.int().cpu().tolist()[i],
                'conf': conf
            }

    for track_id, data in unique_tracks.items():
        canonical_id = id_merge_map.get(track_id, track_id)
        box, cls_id, conf = data['box'], data['cls_id'], data['conf']
        label = f"id:{canonical_id} {class_names[cls_id]} {conf:.2f}"
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (box[0], box[1] - h - 10), (box[0] + w, box[1]), (82, 139, 255), -1)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (82, 139, 255), 2)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return frame

def load_custom_reid_model(weights_path: Path, num_classes: int, device):
    if not weights_path.exists():
        raise FileNotFoundError(f"Custom Re-ID weights not found at {weights_path}")
    
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def process_video_for_lifecycle(args):
    model = YOLO(args.model)
    class_names = model.names
    
    custom_reid_model = load_custom_reid_model(args.reid_model, num_classes=3, device=model.device)

    cap = cv2.VideoCapture(str(args.video))
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(args.output_video), fourcc, fps, (w, h))

    id_merge_map, tracked_objects, event_log = {}, {}, []
    frame_number = 0
    tracker_initialized = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        results = model.track(frame, persist=True, tracker=str(args.tracker_config), verbose=False, iou=0.5, conf=0.5, half=True)
        
        if not tracker_initialized and hasattr(model.predictor, 'trackers') and model.predictor.trackers:
            tracker = model.predictor.trackers[0]
            tracker.encoder.model = custom_reid_model
            tracker_initialized = True

        annotated_frame = frame.copy()
        if results and results[0].boxes.id is not None:
            id_merge_map = create_id_merge_map(results[0].boxes, class_names, id_merge_map, args.merge_dist)
            
            for i in range(len(results[0].boxes.id)):
                track_id = results[0].boxes.id.int().cpu().tolist()[i]
                class_id = results[0].boxes.cls.int().cpu().tolist()[i]
                canonical_id = id_merge_map.get(track_id, track_id)
                current_class_name = class_names[class_id]
                timestamp = frame_number / fps
                
                if canonical_id not in tracked_objects:
                    tracked_objects[canonical_id] = {'class': current_class_name, 'first_seen': timestamp}
                    event = f"[{timestamp:.2f}s] NEW: Object {canonical_id} appeared as '{current_class_name}'"
                    event_log.append(event)
                elif tracked_objects[canonical_id]['class'] != current_class_name:
                    previous_class_name = tracked_objects[canonical_id]['class']
                    event = f"[{timestamp:.2f}s] STATE_CHANGE: Object {canonical_id} changed from '{previous_class_name}' to '{current_class_name}'"
                    event_log.append(event)
                    tracked_objects[canonical_id]['class'] = current_class_name

            annotated_frame = draw_custom_annotations(frame, results[0].boxes, id_merge_map, class_names)
        
        out_video.write(annotated_frame)

    with open(args.output_log, 'w') as f:
        for event in event_log:
            f.write(f"{event}\n")

    cap.release()
    out_video.release()

def main():
    parser = argparse.ArgumentParser(description="Analyze object lifecycle with a fine-tuned Re-ID model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reid_model", type=Path, default="weights/reid_finetuned/osnet_x1_0_finetuned.pth")
    parser.add_argument("--output_video", type=Path, default="reports/lifecycle_video_final.mp4")
    parser.add_argument("--output_log", type=Path, default="reports/events_final.log")
    parser.add_argument("--tracker_config", type=Path, default="config/trackers/botsort.yaml")
    parser.add_argument("--merge_dist", type=int, default=300)
    args = parser.parse_args()

    process_video_for_lifecycle(args)
    
    print("Script finished.")

if __name__ == "__main__":
    main()