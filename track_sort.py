import os
import cv2
import sys
import time
import numpy as np
from sort import Sort  # SORT tracker implementation


# CONFIG
VIDEO_PATH = 'data/video_1.mp4'
DETECTIONS_FILE = 'data/detections.txt'
OUTPUT_FILE = 'results_1/results_sort.txt'
MAX_AGE = 30  # frames a track persists

# Class mapping (YOLO person → MOT pedestrian)
CLASS_ID_MAP = {
    0: 1,
}

def load_detections(detection_file):
    """Load detections from file and organize by frame."""
    detections = {}
    try:
        with open(detection_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    frame, x1, y1, x2, y2, conf, cls = line.strip().split(',')
                    frame = int(frame)
                    det = [float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)]
                    detections.setdefault(frame, []).append(det)
    except FileNotFoundError:
        print(f"Error: Detections file not found at {detection_file}. Please run detection first.")
        sys.exit(1)
    return detections

def run_sort():
    """Run SORT tracker and return total frames processed."""
    tracker = Sort(max_age=MAX_AGE)
    detections = load_detections(DETECTIONS_FILE)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    total_frames = 0
    print(f"Starting SORT tracking. Results will be written to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as out:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            frame_dets = detections.get(frame_id, [])
            dets_for_sort = []

            for det in frame_dets:
                x1, y1, x2, y2, conf, cls = det
                mapped_cls = CLASS_ID_MAP.get(cls, cls)
                # SORT expects bbox in [x1, y1, x2, y2, score]
                dets_for_sort.append([x1, y1, x2, y2, conf])

            if len(dets_for_sort) > 0:
                dets_for_sort_np = np.array(dets_for_sort, dtype=np.float32)
            else:
                dets_for_sort_np = np.empty((0, 5), dtype=np.float32)

            # Update tracker
            tracks = tracker.update(dets_for_sort_np)

            for trk in tracks:
                x1, y1, x2, y2, track_id = trk
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                w, h = x2 - x1, y2 - y1
                conf = 1.0 
                output_cls = 1  # assume pedestrians

                # MOT format
                out.write(f"{frame_id},{int(track_id)},{x1},{y1},{w},{h},{conf:.2f},{output_cls},1\n")

            frame_id += 1

    cap.release()
    print(f"SORT tracking complete. Results saved to {OUTPUT_FILE}")
    return total_frames

if __name__ == '__main__':
    start_time = time.time()
    total_frames = run_sort()
    end_time = time.time()

    elapsed = end_time - start_time
    fps = total_frames / elapsed if total_frames > 0 else 0.0

    output_dir = os.path.dirname(OUTPUT_FILE)
    fps_file = os.path.join(output_dir, "fps_log.txt")

    with open(fps_file, "a") as f:
        f.write(f"SORT | time: {elapsed:.2f}s | FPS: {fps:.2f} | Frames: {total_frames}\n")

    print(f"SORT tracking completed in {elapsed:.2f}s, FPS: {fps:.2f}")
