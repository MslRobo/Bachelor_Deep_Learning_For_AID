import os
import tensorflow as tf
import torch

import numpy as np
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


"""
************************************************************************************************
The function "detect_fn" below is taken from the source below:

Title: TFODCourse
File: 2. Training and Detection.ipynb
Author: Nicholas Renotte
Date: 03.04.2021
Code version: 1.0
Availability: https://github.com/nicknochnack/TFODCourse

************************************************************************************************
"""
@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


class Detection_Model:
    def __init__(self, model_type, classes, paths={}, ckpt_number=3):
        self.model_type = model_type
        self.classes = classes
        self.class_ids = classes
        self.paths = paths
        self.ckpt_no = 'ckpt-' + str(ckpt_number)
        self.configs = None
        self.ckpt = None
        self.category_index = None
        self.CONFIDENCE_LEVEL = 0.6  # Confidence level for the TensorFlowAPI models

        self.model = None
        self.init_model()
    
    @property
    def class_ids(self):
        return self._class_ids
    
    @class_ids.setter
    def class_ids(self, classes):
        class_ids = {}
        for class_name in classes:
            id = classes[class_name]
            class_ids[id] = class_name 
        self._class_ids = class_ids

    def init_model(self):
        if self.model_type == "yolov5":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
            # YOLOv5 stores and use cache by default. Use the line below instead if there are any problems with cache.
            # model = torch.hub.load('ultralytics/yolov5', 'yolov5x', force_reload=True)
            self.model = model
        elif self.model_type == "yolov5_trained":
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='training/yolov5/yolov5/runs/train/yolov5x_trained/weights/best.pt')
            self.model = model
        else:
            """
            ************************************************************************************************
            The code below until the END statement is taken from the source below:

            Title: TFODCourse
            File: 2. Training and Detection.ipynb
            Author: Nicholas Renotte
            Date: 03.04.2021
            Code version: 1.0
            Availability: https://github.com/nicknochnack/TFODCourse

            ************************************************************************************************
            """
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            self.configs = config_util.get_configs_from_pipeline_file(self.paths['PIPELINE_CONFIG'])
            self.model = model_builder.build(model_config=self.configs['model'], is_training=False)

            self.ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
            self.ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], self.ckpt_no)).expect_partial()
            self.category_index = label_map_util.create_category_index_from_labelmap(self.paths['LABELMAP'])
            """
            END
            """

    def detect(self, frame, w=0, h=0):
        return_object = {"frame": frame, "boxes": [], "scores": [], "object_classes": []}
        
        if self.model_type == "yolov5" or self.model_type == "yolov5_trained":
            results = self.model(frame)
            df = results.pandas().xyxy[0]
            for row in df.itertuples():
                obj_class = str(row[7]).lower()
                if obj_class not in self.classes:
                    continue
                return_object["boxes"].append([float(row[1]), float(row[2]), (float(row[3])-float(row[1])), (float(row[4])-float(row[2]))])
                return_object["scores"].append(float(row[5]))
                return_object["object_classes"].append(obj_class)
        else:
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor, self.model)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            for i, score in enumerate(detections["detection_scores"]):
                if float(score) >= self.CONFIDENCE_LEVEL:
                    x1 = float(detections['detection_boxes'][i][1]) * float(w)
                    x2 = float(detections['detection_boxes'][i][3]) * float(w)
                    y1 = float(detections['detection_boxes'][i][0]) * float(h)
                    y2 = float(detections['detection_boxes'][i][2]) * float(h)
                    class_id = str(int(detections['detection_classes'][i]) + 1)
                    obj_class = self.class_ids.get(class_id)

                    return_object["boxes"].append([x1, y1, (x2-x1), (y2-y1)])
                    return_object["scores"].append(float(score))
                    return_object["object_classes"].append(obj_class)

        return return_object