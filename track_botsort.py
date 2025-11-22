import os
import sys
import time
from ultralytics import YOLO



# CONFIG
VIDEO_PATH = 'data/video_1.mp4'
OUTPUT_FILE = 'results_1/results_botsort.txt'
YOLO_MODEL = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.3
TRACKER_CONFIG = 'botsort.yaml'

# Class mapping (YOLO person → MOT pedestrian)
CLASS_ID_MAP = {
    0: 1, # Map YOLO's 'person' (class 0) to MOT17's 'pedestrian' (class 1)
}

def run_ultralytics_botsort():
    """Run BOTSort using Ultralytics and return number of frames processed."""
    print("Running BOTSort via Ultralytics...")

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file {VIDEO_PATH} not found.")
        sys.exit(1)

    # Load YOLO model
    model = YOLO(YOLO_MODEL)

    # Run tracking with BOTSort
    results = model.track(
        source=VIDEO_PATH,
        tracker=TRACKER_CONFIG,
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        stream=True,
        verbose=False
    )

    all_tracks = []
    total_frames = 0

    for frame_id, result in enumerate(results):
        total_frames += 1

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x_center, y_center, w, h = boxes[i]
                x1 = x_center - w / 2
                y1 = y_center - h / 2

                original_cls = classes[i]
                mapped_cls = CLASS_ID_MAP.get(original_cls, original_cls)

                all_tracks.append({
                    'frame_id': frame_id,
                    'track_id': track_ids[i],
                    'bbox': [x1, y1, w, h],
                    'confidence': confidences[i],
                    'class_id': mapped_cls
                })

    # Save tracking results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for track in all_tracks:
            frame = track['frame_id']
            tid = track['track_id']
            x1, y1, w, h = track['bbox']
            conf = track['confidence']
            cls = track['class_id']

            # MOT format
            f.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.3f},{cls},1\n")

    print(f"BOTSort complete. Results saved to {OUTPUT_FILE}")

    return total_frames


if __name__ == "__main__":
    start_time = time.time()
    total_frames = run_ultralytics_botsort()
    end_time = time.time()

    elapsed = end_time - start_time
    fps = total_frames / elapsed if total_frames > 0 else 0.0

    output_dir = os.path.dirname(OUTPUT_FILE)
    fps_file = os.path.join(output_dir, "fps_log.txt")

    with open(fps_file, "a") as f:
        f.write(f"BOTSORT | time: {elapsed:.2f}s | FPS: {fps:.2f} | Frames: {total_frames}\n")

    print(f"BOTSORT tracking completed in {elapsed:.2f}s, FPS: {fps:.2f}")
