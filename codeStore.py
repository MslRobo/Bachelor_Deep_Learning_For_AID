import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image
from tools.deep_sort.tracker import Tracker
from tools.deep_sort import preprocessing, nn_matching
from tools.deep_sort.detection import Detection
from tools.helpers.generate_detections import create_box_encoder

# Initialize the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize the DeepSORT tracker with your model's feature extractor
max_cosine_distance = 0.4 # Hard-coded for non hard-coded approach look at tracking_model.py from Vedvik. TODO: Remove
nn_budget = None # Hard-coded for non hard-coded approach look at tracking_model.py from Vedvik. TODO: Remove
nms_max_overlap = 1.0 # Hard-coded for non hard-coded approach look at tracking_model.py from Vedvik. TODO: Remove
encoder = create_box_encoder('tools/model_data/mars-small128.pb', batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Video file to process
video_path = 'data/rawData/timelapse.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file could not be opened.")
    sys.exit(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Processing finished.")
        break

    # # Image preprocessing for YOLOv5
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = torch.from_numpy(img).to(torch.float32) / 255.0
    # img = img.permute(2, 0, 1).unsqueeze(0)
    # Image preprocessing for YOLOv5
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarrat(frame)

    frame_size = frame.shape[:2]
    img_data = frame
    img_data = img_data / 255.
    img_data = img_data[np.newaxis, ...].astype(np.float32)

    # YOLOv5 inference
    pred = model(img, augment=False)

    # Assuming 'pred' is a list of tensors, each tensor for one image in the batch
    # If you're processing one image at a time, you can directly work with the first tensor
    detections_tensor = pred[0]
    print("predictions: ", pred)
    print("Detections: ", pred[0])

    # Process detections
    # Note: Ensure that 'detections_tensor' contains the expected detection format
    bboxes = detections_tensor[:, :4]  # Bounding box coordinates
    scores = detections_tensor[:, 4]   # Confidence scores
    object_classes = detections_tensor[:, 5]  # Class IDs
    print("bboxes: ", bboxes)

    # # Convert xyxy to xywh format for Deep SORT compatibility
    # bboxes_xywh = bboxes.clone()
    # bboxes_xywh[:, 2:] = bboxes_xywh[:, 2:] - bboxes_xywh[:, :2]  # Convert bottom right to width and height
    # bboxes_xywh = bboxes_xywh.cpu().numpy()
    # scores = scores.cpu().numpy()
    # object_classes = object_classes.cpu().numpy()

    bboxes_xywh = []
    for bbox in bboxes:
        print("bbox: ", bbox)
        print("bbox type: ", type(bbox))
        bbox = bbox.cpu().numpy()
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width < 0 or height < 0:
            print(f"Invalid bbox dimensions: {bbox}")
            continue  # Skip or handle invalid bounding boxes
        x, y = x1, y1  # Assuming (x1, y1) is the top-left corner
        bboxes_xywh.append([x, y, abs(width), abs(height)])
    bboxes_xywh = np.array(bboxes_xywh)
    print("bboxes_xywh: ", bboxes_xywh)

    features = encoder(frame, bboxes_xywh)
    detections = [Detection(bbox, score, 'object', feature) for bbox, score, feature in zip(bboxes_xywh, scores, features)]

    # Apply NMS
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])  # Placeholder: Adjust as needed
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Tracker update
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.is_deleted():
            continue  # Skip unconfirmed or deleted tracks

        bbox = track.to_tlwh()
        x, y, w, h = map(int, bbox)
        color = (255, 255, 255) if track.is_tentative() else (0, 255, 0)  # white for tentative, green for confirmed

        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Draw the track ID
        cv2.putText(frame, f"ID: {track.track_id}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        # Draw the class name if available
        if track.get_class():
            cv2.putText(frame, f"{track.get_class()}", (x, y - 25), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()