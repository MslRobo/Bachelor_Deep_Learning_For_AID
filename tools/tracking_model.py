import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from ultralytics import YOLO

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from helpers import generate_detections as gdet


class Tracking_Model:
    def __init__(self, model_filename, tracker_type="DeepSort", max_cosine_distance=0.4, nn_budget=None, nms_max_overlap=1.0):
        self.model_filename = model_filename
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.nms_max_overlap = nms_max_overlap
        self.encoder = None
        self.tracker = None
        self.tracker_type = tracker_type
        self.init_tracker()
    
    def init_tracker(self):
        if self.tracker_type == "DeepSort":
            """
            ************************************************************************************************
            The code below until the END statement is taken from the source below:

            Title: yolov4-deepsort
            File: preprocessing.py
            Author: The AI Guy
            Date: 21.08.2021
            Code version: 1.0
            Availability: https://github.com/theAIGuysCode/yolov4-deepsort

            ************************************************************************************************
            """
            encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
            self.encoder = encoder
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
            tracker = Tracker(metric)
            """
            END
            """
        elif self.tracker_type == "yolov8":
            tracker = YOLO('yolov8x.pt')
        else:
            tracker = Simple_Tracker()
        self.tracker = tracker
    
    def track(self, model_detections):
        if self.tracker_type == "DeepSort":
            """
            ************************************************************************************************
            The code below until the END statement is taken from the source below:

            Title: yolov4-deepsort
            File: preprocessing.py
            Author: The AI Guy
            Date: 21.08.2021
            Code version: 1.0
            Availability: https://github.com/theAIGuysCode/yolov4-deepsort

            ************************************************************************************************
            """
            frame, boxes, scores, object_classes = model_detections["frame"], model_detections["boxes"], model_detections["scores"], model_detections["object_classes"]
            bboxes = np.array(boxes)
            scores = np.array(scores)
            object_classes = np.array(object_classes)

            features = self.encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, object_classes, features)]

            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxes, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()
            self.tracker.update(detections)
            """
            END
            """
        else:
            self.tracker.update(model_detections)
    
    def get_tracks(self):
        return self.tracker.tracks


class Simple_Tracker:
    def __init__(self):
        self.vehicle_count = 0
        self.all_tracks = []
        self.tracks = []
        self.age = 0

    def calculate_IoU(self, new_detection, old_detection):
        if new_detection[0] > old_detection[0]: 
            x_min = new_detection[0]
        else:                      
            x_min = old_detection[0]
        if new_detection[1] > old_detection[1]: 
            y_min = new_detection[1]
        else:                      
            y_min = old_detection[1]
        if new_detection[2] < old_detection[2]: 
            x_max = new_detection[2]
        else:                      
            x_max = old_detection[2]
        if new_detection[3] < old_detection[3]: 
            y_max = new_detection[3]
        else:                      
            y_max = old_detection[3]
        
        intersection_area = (x_max - x_min) * (y_max - y_min)
        if intersection_area < 0 or (x_max - x_min) < 0 or (y_max - y_min) < 0:
            return 0
        
        union_area = ((old_detection[2] - old_detection[0]) * (old_detection[3] - old_detection[1])) + ((new_detection[2] - new_detection[0]) * (new_detection[3] - new_detection[1])) - intersection_area

        IoU = intersection_area / union_area
        return IoU
    
    def update(self, model_detections):
        detected_objects, object_classes = model_detections["boxes"], model_detections["object_classes"]

        self.tracks = []
        for detected_object, object_class in zip(detected_objects, object_classes):
            detected_object = [detected_object[0], detected_object[1], detected_object[0]+detected_object[2], detected_object[1]+detected_object[3]]
            for track in self.all_tracks:
                bboxes = track.to_tlbr()
                IoU = self.calculate_IoU(detected_object, bboxes)
                if IoU > 0.5:
                    track.boxes = detected_object
                    track.class_name = object_class
                    track.age = self.age
                    break
            else:
                self.vehicle_count += 1
                track = Simple_Track(detected_object, self.vehicle_count, object_class, self.age)
                self.all_tracks.append(track)
            self.tracks.append(track)
        
        for track in self.all_tracks:
            if self.age - track.age > 10:
                del track
        self.age += 1


class Simple_Track:
    def __init__(self, boxes, track_id, class_name, age):
        self.time_since_update = 0
        self.boxes = boxes
        self.track_id = track_id
        self.class_name = class_name
        self.age = age
    
    def get_class(self):
        return self.class_name

    def is_confirmed(self):
        return True

    def to_tlbr(self):
        return self.boxes