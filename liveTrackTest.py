import torch
import cv2
import numpy as np
from PIL import Image
from tools.deep_sort import nn_matching, linear_assignment, preprocessing
from tools.deep_sort import detection as dt
from tools.deep_sort.tracker import Tracker
from tools import visualize_objects as vo
from tools.helpers import generate_detections as gdet

# Initialize device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True).to(device)
model.eval()

encoder = gdet.create_box_encoder("tools/model_data/mars-small128.pb", batch_size=1)

# DeepSORT parameters
nn_budget = 100
max_cosine_distance = 0.4
track_init_iou = 0.4
track_iou_min = 0.1
max_age = 30
nms_max_overlap = 0.7

# Optional display of detected objects
show_det = True

# Function to draw detections and trajectories
def draw_trajectories(frame, detections, tracks):
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > max_age:
            continue

        bbox = track.get_state()[0:4].astype(int)
        track_id = track.track_id
        color = (255, 255, 255)  # Adjust color as needed

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, str(track_id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        if show_det:
            # Display detected object information
            x_center, y_center, w, h, object_class, conf = detections[track.track_id]
            x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)
            cv2.putText(frame, f"{object_class} - {conf:.2f}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Main processing loop
cap = cv2.VideoCapture("data/rawData/Tunnel5.mp4")  # Replace with your video path

# Initialize the DeepSORT tracker with your model's feature extractor
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

tracker = Tracker(metric)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # # Preprocess frame for YOLOv5
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = torch.from_numpy(img).unsqueeze(0)
    # img = img.float() / 255.0  # Normalize
    # img = img.to(device)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    frame_size = frame.shape[:2]
    img_data = frame
    img_data = img_data / 255.
    img_data = img_data[np.newaxis, ...].astype(np.float32)

    # Run YOLOv5 inference
    with torch.no_grad():
        results = model(img)

    # Parse detections
    detections = []
    bbox_xywh = []
    confidences = []
    class_ids = []
    class_names = []
    for index, det in results.pandas().xyxy[0].iterrows():
        # print(det)
        # print(results.pandas().xyxy)
        # print(type(results.pandas().xyxy))
        # print(type(results.pandas().xyxy[0]))
        # print(type(results.pandas().xyxy[0].confidence[0]))
        # print(type(det))
        # print(det.values)
        if det["confidence"] > 0.5:
            xmin, ymin, xmax, ymax, conf, object_class, object_name = det.values

            x_center = xmin
            y_center = ymin
            w = xmax - xmin
            h = ymax - ymin



            # x_center, y_center, w, h, conf, object_class = det.values
            # detections.append([x_center, y_center, w, h, object_class, conf])
            bbox_xywh.append([xmin, ymin, w, h])
            confidences.append(conf)
            class_ids.append(object_class)
            class_names.append(object_name)
            # indices = preprocessing.non_max_suppression(bbox_xywh, object_class, nms_max_overlap, conf)
            # detections.append(dt.Detection([xmin, ymin, w, h], conf, nms_max_overlap, object_class))
            
    features = encoder(frame, bbox_xywh)
    detections = [dt.Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bbox_xywh, confidences, class_ids, features)]

    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxes, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]


    # DeepSORT tracking
    print(detections)
    print(boxes)
    print(scores)
    if detections:
        # print(bbox_xywh, " ", confidences, " ", object_class)
        # detection = dt.Detection(bbox_xywh, confidences, class_names, class_ids)
        tracker.predict()
        tracker.update(detections)
        tracks = tracker.tracks

    # Visualize and display frame
    print(tracks)
    if tracks:
        for track in tracks:
            vo.draw_rectangle(frame, track, 25)
        # draw_trajectories(frame, detections, tracks)
    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()