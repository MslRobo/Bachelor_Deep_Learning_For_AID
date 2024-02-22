import cv2
import torch
import numpy as np
# from tools.deep_sort import BBox
from tools.deep_sort import nn_matching, linear_assignment, iou_matching
from tools.deep_sort.tracker import Tracker
from yolov5 import detect  # Assuming a custom `detect.py` script for object detection using YOLOv5

# Load video capture
cap = cv2.VideoCapture("data/rawData/timelapse.mp4")  # Replace with your video path

# Load object detection model (assuming YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Adjust model as needed

# Create DeepSORT tracker with relevant parameters
max_cosine_distance = 0.4  # Adjustable threshold for appearance similarity
iou_threshold = 0.5        # Adjustable threshold for bounding box overlap
tracker = Tracker(max_cosine_distance, iou_threshold)

# Create colors for track IDs
color_list = [(int(255 * a), int(255 * b), int(255 * c)) for a, b, c in np.random.rand(100, 3)]

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects and get detections in format: [(class_id, confidence, xmin, ymin, xmax, ymax)]
    detections = detect(frame, model)

    # Convert detections to DeepSORT-compatible format (adjust if using a different detection format)
    bbox_xywh = []
    confidences = []
    class_ids = []
    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls_id = det
        w = x_max - x_min
        h = y_max - y_min
        bbox_xywh.append([x_min, y_min, w, h])
        confidences.append(conf)
        class_ids.append(cls_id)

    # Update tracker
    tracks = tracker.update(bbox_xywh, confidences, class_ids)

    # Draw tracked objects and labels
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        id = track.track_id
        x, y, w, h = track.get_state()
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color_list[id % len(color_list)], 2)
        cv2.putText(frame, f"ID: {id}", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[id % len(color_list)], 2)

    # Display the result
    cv2.imshow("Object Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
