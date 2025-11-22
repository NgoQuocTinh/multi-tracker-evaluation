import os
import cv2
import sys
import time
from deep_sort_realtime.deepsort_tracker import DeepSort


# CONFIG
VIDEO_PATH = 'data/video_1.mp4'
DETECTIONS_FILE = 'data/detections.txt'
OUTPUT_FILE = 'results_1/results_deepsort.txt'
MAX_AGE = 30  # frames a track persists without detections

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

def run_deepsort():
    """Run DeepSORT tracker and return number of frames processed."""
    tracker = DeepSort(max_age=MAX_AGE)
    detections = load_detections(DETECTIONS_FILE)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_frames = 0
    print(f"Starting DeepSORT tracking. Results will be written to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as out:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame_dets = detections.get(frame_id, [])
            input_dets = []

            for det in frame_dets:
                x1, y1, x2, y2, conf, cls = det
                w, h = x2 - x1, y2 - y1
                mapped_cls = CLASS_ID_MAP.get(cls, cls)
                input_dets.append(([x1, y1, w, h], conf, frame, mapped_cls))

            tracks = tracker.update_tracks(input_dets, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom
                x1, y1, x2, y2 = map(int, ltrb)
                w, h = x2 - x1, y2 - y1
                conf = 1.0  # DeepSORT track confidence
                output_cls = 1  # Assume all tracks are pedestrians

                # MOT format
                out.write(f"{frame_id},{track_id},{x1},{y1},{w},{h},{conf:.2f},{output_cls},1\n")

            frame_id += 1

    cap.release()
    print(f"DeepSORT tracking complete. Results saved to {OUTPUT_FILE}")
    return total_frames

if __name__ == '__main__':
    start_time = time.time()
    total_frames = run_deepsort()
    end_time = time.time()

    elapsed = end_time - start_time
    fps = total_frames / elapsed if total_frames > 0 else 0.0

    # Save FPS in same folder as results
    output_dir = os.path.dirname(OUTPUT_FILE)
    fps_file = os.path.join(output_dir, "fps_log.txt")

    with open(fps_file, "a") as f:
        f.write(f"DeepSORT | time: {elapsed:.2f}s | FPS: {fps:.2f} | Frames: {total_frames}\n")

    print(f"DeepSORT tracking completed in {elapsed:.2f}s, FPS: {fps:.2f}")
