import cv2
import os
import sys
from ultralytics import YOLO

# Parameters (Adjust as needed)
VIDEO_PATH = 'data/video_1.mp4'
OUTPUT_PATH = 'data/detections.txt'
MODEL_PATH = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
CONF_THRESHOLD = 0.3  # Confidence threshold for detections

if not os.path.exists(VIDEO_PATH):
    print(f"Error: Video file {VIDEO_PATH} not found.")
    sys.exit(1)


def main():
    """Detect objects in a video using YOLO and write bounding boxes to a text file in MOT format."""
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as out:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # Write: frame, x1, y1, x2, y2, conf, class_id
                    out.write(f"{frame_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{conf:.4f},{cls}\n")

            frame_id += 1

    cap.release()
    print(f"Detections written to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
