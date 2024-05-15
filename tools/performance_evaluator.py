import json
import cv2
import os
import time
import numpy as np
from helpers.retinex import SSR
from helpers.retinex import MSR
from noise_manager import Noise_Manager

class Evaluate_Performance:
    def __init__(self, type, dataset_path, classes, detection_model, tracking_model, mask="", noise_type=None):
        self.vid = None
        self.width = 0
        self.height = 0
        self.scale = 1
        self.entries = []
        self.next_entry_index = 0
        self.type = type
        self.detected_objects = []
        self.detected_objects_previous = {}
        self.dataset_paths = dataset_path
        self.datasets = {}
        self.classes = classes
        self.detection_model = detection_model
        self.tracking_model = tracking_model
        self.current_video = ""
        self.mask = mask
        self.queue = {}
        self.prepare()

        self.detection_time_current = 0
        self.tracking_time_current = 0
        self.total_time_current = 0
        self.fps_current = 0

        self.image_enhancement_current = 0
        self.mean_image_enhancement_time = 0

        self.mean_detection_time = 0
        self.min_detection_time = -1
        self.max_detection_time = 0
        
        self.mean_tracking_time = 0
        self.min_tracking_time = -1
        self.max_tracking_time = 0
        
        self.mean_total_time = 0
        self.min_total_time = -1
        self.max_total_time = 0

        self.missed_detections = 0
        self.total_number_of_real_detections = 0
        self.total_number_of_valid_detections = 0
        self.total_number_of_valid_detections_adjusted = 0
        self.false_positives_detections = 0 
        self.false_positives_detections_previous = 0 

        self.missed_tracks = 0

        self.detection_accuracy = 0
        self.detection_accuracy_adjusted = 0
        self.tracking_accuracy = 0
        self.tracking_id_switches = 0
        self.tracking_id_duplicates = 0
        self.incident_accuracy = 0
        self.missed_incidents = 0
        self.false_alarms = 0

        self.mean_fps = 0
        self.min_fps = -1
        self.max_fps = 0

        self.number_of_frames = 0

        self.cars_outside_mask = 0
        self.objects_inside_mask = 0

        self.queues_detected = 0
        self.queue_changes = 0
        self.mean_queue_changes = 0
        self.max_queue_length = 0
        self.min_queue_length = -1
        self.avg_queue_length = 0
        self.min_queue_detection_time = -1
        self.max_queue_detection_time = 0
        self.mean_queue_detection_time = 0
        self.queue_detection_time = []
        
        self.track_values = {}
        self.frame_queues = {}
        self.frameData = []

        self.noise_manager = Noise_Manager(noise_type)

    @property
    def dataset_paths(self):
        return self._dataset_paths

    @dataset_paths.setter
    def dataset_paths(self, datasets):
        dataset_paths = {}
        for dataset in datasets:
            images = dataset.get("images")
            annotations = dataset.get("annotations")
            dataset_name = dataset.get("dataset")

            video = dataset.get("video")
            if video:
                dataset_paths["video"] = video
            elif images is None or annotations is None or dataset_name is None:
                continue
            else:
                dataset_paths[dataset_name] = {"images": images, "annotations": annotations}
        
        self._dataset_paths = dataset_paths

    def prepare(self):
        if self.type == "Video":
            self.vid = cv2.VideoCapture(self.dataset_paths.get("video"))
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.type == "Images":
            print("Doing this dataset thing")
            for dataset_name in self.dataset_paths:
                try:
                    if "self_annotated" in dataset_name:
                        self.prepare_self_annotated(dataset_name)
                except Exception as e:
                    print(e)
                    self.datasets[dataset_name] = {"entries": []}
            self.prepare_all_entries()
        
        mask = cv2.imread(self.mask, cv2.IMREAD_GRAYSCALE)
        _, self.mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)

    def prepare_self_annotated(self, dataset_name="self_annotated"):
        dataset = self.dataset_paths.get(dataset_name)
        # print("Dataset value: ", dataset)
        if dataset is None:
            return
        anno_path = dataset.get("annotations")
        img_path = dataset.get("images")
        mask_path = anno_path.replace("annotations.json", "mask.png")

        if anno_path is None:
            return
        
        with open(anno_path, "r") as annotations:
            data = json.load(annotations)

        annotation_classes_path = anno_path.replace("annotations", "classes")
        with open(annotation_classes_path, "r") as annotation_classes:
            annotation_classes = json.load(annotation_classes)
            
        images_list = {"entries": []}
        for i, img in enumerate(data):
            if i <= 0:
                continue
            filename = img
            row = {"images_path": img_path, "filename": filename, "objects": [], "mask_path": mask_path }
        
            for object in data[img]['instances']:

                info = {}
                for class_ in annotation_classes:
                    if class_["id"] == object["classId"]:
                        class_name = class_["name"]

                        for object_attribute in object["attributes"]:
                            for attribute_group in class_["attribute_groups"]:
                                if object_attribute["groupId"] == attribute_group["id"]:
                                    for attribute_ in attribute_group["attributes"]:
                                        if attribute_["id"] == object_attribute["id"]:
                                            info[attribute_group["name"]] = attribute_["name"]
                
                if class_name == "people":
                    class_name = "person"

                if class_name not in self.classes:
                    continue
                    
                x1 = float(object["points"]["x1"])
                y1 = float(object["points"]["y1"])
                x2 = float(object["points"]["x2"])
                y2 = float(object["points"]["y2"])

                row["objects"].append({"class": class_name, "class_id": self.classes.get(class_name),  "x1": x1, "y1": y1, "x2": x2, "y2": y2, "info": info})
            
            images_list["entries"].append(row)

        images_list['entries'] = sorted(images_list['entries'], key = lambda i: i['filename'])
        self.datasets[dataset_name] = images_list

    def prepare_all_entries(self):
        if self.type != "Images":
            return

        print("\nEntries:")
        classes = {}
        number_of_objects = 0
        entries = []
        for dataset in self.datasets:
            for entry in self.datasets[dataset]["entries"]:
                entries.append(entry)
                number_of_objects += len(entry["objects"])
                for obj in entry["objects"]:
                    if obj["class"] in classes:
                        classes[obj["class"]] += 1
                    else:
                        classes[obj["class"]] = 1
        
        print(f"Number of files: {len(entries)}")
        print(f"Number of objects: {number_of_objects}")
        for obj_class in classes:
            print(f" - {obj_class}: {classes[obj_class]}")
        self.entries = entries

    def performance(self, track, text):
        bbox = track.to_tlbr()
        object_class = track.get_class()
        track_id = track.track_id
        
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        incident = False
        best_IoU = {"score": 0, "object": None, "real_object": None}
        for real_object in self.entries[self.next_entry_index-1]["objects"]:
            real_object["x1"] *= self.scale
            real_object["y1"] *= self.scale
            real_object["x2"] *= self.scale
            real_object["y2"] *= self.scale
            if x1 > real_object["x1"]: 
                x_min = x1
            else:                      
                x_min = real_object["x1"]
            if y1 > real_object["y1"]: 
                y_min = y1
            else:                      
                y_min = real_object["y1"]
            if x2 < real_object["x2"]: 
                x_max = x2
            else:                      
                x_max = real_object["x2"]
            if y2 < real_object["y2"]: 
                y_max = y2
            else:                      
                y_max = real_object["y2"]
            
            intersection_area = (x_max - x_min) * (y_max - y_min)
            if intersection_area < 0 or (x_max - x_min) < 0 or (y_max - y_min) < 0:
                continue
            
            union_area = ((real_object["x2"] - real_object["x1"]) * (real_object["y2"] - real_object["y1"])) + ((x2 - x1) * (y2 - y1)) - intersection_area
            if union_area < 0:
                print(f"IA: {intersection_area}")
                print(f"UA: {union_area}")
                print(f"RO: x1 = {real_object['x1']}, y1 = {real_object['y1']}, x2 = {real_object['x2']}, y2 = {real_object['y2']}")
                print(f"DO: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
                raise ValueError

            IoU = intersection_area / union_area

            if (IoU - 1)**2 < (best_IoU["score"] - 1)**2:
                best_IoU["object"] = {"bbox": bbox, "class": object_class, "ID": track_id}
                best_IoU["real_object"] = real_object
                best_IoU["score"] = IoU

        if best_IoU["object"] is not None and best_IoU["score"] > 0.4:
            if best_IoU["real_object"]['info']['status'] == "Incident":
                incident = True
            self.detected_objects.append(best_IoU)
        else:
            self.false_positives_detections += 1
        
        if incident and ("Stopped vehicle" in text or "Pedestrian" in text):
            self.incident_accuracy += 1
        elif incident:
            self.missed_incidents += 1
        elif "Stopped vehicle" in text or "Pedestrian" in text:
            self.false_alarms += 1
        
        if self.track_values == {}:
            self.track_values = {
                'centerpoint': {
                    'x': [(x_min + x_max) / 2],
                    'y': [(y_min + y_max) / 2]
                },
                'true_labels': [],
                'predicted_labels': []
            }
        else:
            self.track_values['centerpoint']['x'].append((x_min + x_max) / 2)
            self.track_values['centerpoint']['y'].append((y_min + y_max) / 2)
        

        

    def frame_analytics(self, frameData):
        print(":I")

    def image_enhancement(self, frame, image_enhancement="", mask=None, brightness=None):
        img_enh_start = time.time()
        # frame = frame[1]
        # print(frame)
        # print("Image Enhancement: ", image_enhancement)
        if image_enhancement == "gray_linear":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            # print("Gray Linear Enhancement")
            # print(frame)
        elif image_enhancement == "gray_nonlinear":
            # print("Gray Non Linear Enhancements")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gamma=2.0
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif image_enhancement == "he":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif image_enhancement == "retinex_ssr":
            variance=300
            img_ssr=SSR(frame, variance)
            frame = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2RGB)
        elif image_enhancement == "retinex_msr":
            variance_list=[200, 200, 200]
            img_msr=MSR(frame, variance_list)
            frame = cv2.cvtColor(img_msr, cv2.COLOR_BGR2RGB)
        elif image_enhancement == "mask":
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if brightness is not None:
            brightness_percent = 1 + (brightness / 100)
            frame = cv2.convertScaleAbs(frame, alpha=brightness_percent, beta=0)

        if self.noise_manager.noise_type is not None:
            frame = self.noise_manager.add_noise(frame)
        
        img_enh_end = time.time()
        self.image_enhancement_current = img_enh_end - img_enh_start
        self.mean_image_enhancement_time += self.image_enhancement_current

        return frame

    def detect(self, frame):
        detection_start = time.time()
        model_detections = self.detection_model.detect(frame, self.width, self.height)
        detection_end = time.time()
        self.detection_time_current = detection_end - detection_start

        if self.detection_time_current < 10:
            self.mean_detection_time += self.detection_time_current

        if self.min_detection_time == -1 or (self.detection_time_current < self.min_detection_time and self.detection_time_current > 0):
            self.min_detection_time = self.detection_time_current 
        if self.detection_time_current > self.max_detection_time and self.detection_time_current < 10:
            self.max_detection_time = self.detection_time_current

        return model_detections
    
    def track(self, model_detections):
        track_start = time.time()
        self.tracking_model.track(model_detections)
        track_end = time.time()
        self.tracking_time_current = track_end - track_start

        self.mean_tracking_time += self.tracking_time_current

        if (self.min_tracking_time == -1 or self.tracking_time_current < self.min_tracking_time) and self.tracking_time_current > 0:
            self.min_tracking_time = self.tracking_time_current 
        if self.tracking_time_current > self.max_tracking_time:
            self.max_tracking_time = self.tracking_time_current

    def detect_and_track(self, frame):
        self.number_of_frames += 1
        model_detections = self.detect(frame)
        self.track(model_detections)

        self.total_time_current = self.detection_time_current + self.tracking_time_current + self.image_enhancement_current
        self.mean_total_time += self.total_time_current

        if (self.min_total_time == -1 or self.total_time_current < self.min_total_time) and self.total_time_current > 0:
            self.min_total_time = self.total_time_current
        if self.total_time_current > self.max_total_time:
            self.max_total_time = self.total_time_current

        self.fps_current = 1.0 / (self.total_time_current)
        self.fps_current = round(self.fps_current, 3)

        self.mean_fps += self.fps_current
        if (self.min_fps == -1 or self.fps_current < self.min_fps) and self.fps_current > 0:
            self.min_fps = self.fps_current
        if self.fps_current > self.max_fps:
            self.max_fps = self.fps_current
        
    def read(self, resize=1, new_resolution=False):
        if self.type == "Video":
            return True, self.vid.read(), False, None
        else:
            frame = None
            ret = False
            new_video = False
            mask = None
            try:
                entry = self.entries[self.next_entry_index]
                path = entry["images_path"]
                mask_path = entry["mask_path"]
                image_path = os.path.join(path, '{}'.format(entry["filename"]))
                frame = cv2.imread(image_path)
                self.height, self.width, _ = frame.shape
                mask = cv2.imread(mask_path, 0)
                ret = True

                if resize <= 1:
                    self.scale = resize
                    self.width = int(self.width * self.scale)
                    self.height = int(self.height * self.scale)
                    frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_AREA)
                    mask = cv2.resize(mask, (self.width, self.height), interpolation = cv2.INTER_AREA)

                if new_resolution:
                    resolutions = {
                        '720': {'width': 1280, 'height': 720},
                        '648': {'width': 1152, 'height': 648},
                        '576': {'width': 1024, 'height': 576},
                        '360': {'width': 640, 'height': 360}
                    }
                    new_resolution = resolutions[new_resolution]
                    self.scale = resize
                    self.width = int(new_resolution['width'])
                    self.height = int(new_resolution['height'])
                    frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_AREA)
                    mask = cv2.resize(mask, (self.width, self.height), interpolation = cv2.INTER_AREA)

                if self.current_video != entry['images_path'] and self.current_video != "":
                    new_video = True
                self.current_video = entry['images_path']
            except IndexError as e:
                print(e)
            self.next_entry_index += 1

            return ret, frame, new_video, mask
    
    def get_tracks(self):
        return self.tracking_model.get_tracks()
    
    def queue_performance(self, queue_stats, queue_time):
        cars = []
        self.queues_detected = len(queue_stats)

        length = 0
        max_queue_length = 0
        min_queue_length = -1
        for queue in queue_stats:
            queue = queue_stats[queue]
            length += len(queue)
            
            if (self.min_queue_length == -1 or self.min_queue_length > len(queue)):
                self.min_queue_length = len(queue)
            
            if self.max_queue_length < len(queue):
                self.max_queue_length = len(queue)
            
            if (min_queue_length == -1 or min_queue_length > len(queue)):
                min_queue_length = len(queue)
            
            if max_queue_length < len(queue):
                max_queue_length = len(queue)

            for car in queue:
                if car not in cars:
                    cars.append(car)
            
        self.avg_queue_length = length / len(queue_stats)

        for car in cars:
            car_tracker = 0
            for queue in queue_stats:
                queue = queue_stats[queue]
                if car in queue:
                    car_tracker += 1
            
            if car_tracker > 1:
                self.queue_changes += car_tracker - 1
        self.mean_queue_changes = self.queue_changes / len(cars)
            
        self.min_queue_detection_time = queue_time if queue_time < self.min_queue_detection_time or self.min_queue_detection_time == -1 else self.min_queue_detection_time
        self.max_queue_detection_time = queue_time if queue_time > self.max_queue_detection_time else self.max_queue_detection_time
        self.queue_detection_time.append(queue_time)
        self.mean_queue_detection_time = sum(self.queue_detection_time) / len(self.queue_detection_time)

        self.frame_queues = {
            'avg_queue_length': length / len(queue_stats),
            'max_queue_length': max_queue_length,
            'min_queue_length': min_queue_length,
            'mean_queue_changes': self.queue_changes / len(cars)
        }
    
    def status(self, frameData):
        detection_time = int((self.detection_time_current) * 1000)
        track_time = int(self.tracking_time_current * 1000)
        print(f"\nFrame: {self.number_of_frames}")
        print(f"FPS: {self.fps_current}")
        print(f"IE time: {int(self.image_enhancement_current* 1000)} ms")
        print(f"Detection time: {detection_time} ms")
        print(f"Tracking time: {track_time} ms")
        print(f"Total time: {int(self.total_time_current* 1000)} ms")

        class_to_id = {'car': 1, 'person': 2, 'truck': 3, 'bus': 4, 'bike': 5, 'motorbike': 6, 'Road anomaly': 10}
        avg_score = 0
        avg_score_adjusted = 0
        number_of_detections_adjusted = 0
        number_of_correct_classes = 0
        number_of_wrong_classes = 0
        number_of_correct_ids = 0
        number_of_wrong_ids = 0
        number_of_duplicate_ids = 0
        object_ids = []
        print("Detected objects:")
        for detected_object in self.detected_objects:
            print("Detected object: ", detected_object)
            print("Class: ", detected_object['object']['class'])
            print(f"\t- {detected_object['object']['class']}, {round(detected_object['score']*100, 2)} %")
            avg_score += detected_object['score']
            if detected_object["real_object"]['info']['occluded'] == "False":
                avg_score_adjusted += detected_object['score']
                number_of_detections_adjusted += 1
            if detected_object["object"]["class"] == detected_object["real_object"]["class"]:
                print("\t\t- Correct Class")
                number_of_correct_classes += 1
            else:
                print("\t\t- Wrong Class")
                number_of_wrong_classes += 1
            
            x1, y1, x2, y2 = detected_object["object"]["bbox"]
            bx, by = (x1 + x2) // 2, (y1 + y2) // 2
            bx, by = int(bx), int(by)
            if self.mask[by, bx] == 255:
                self.cars_outside_mask += 1
            
            if 'ID' in detected_object['real_object']['info']:
                if detected_object['real_object']['info']['ID'] in self.detected_objects_previous:
                    if self.detected_objects_previous[detected_object['real_object']['info']['ID']] == detected_object['object']['ID']:
                        if detected_object['real_object']['info']['ID'] in object_ids:
                            print("\t\t- Duplicate ID")
                            number_of_duplicate_ids += 1
                        else:
                            print("\t\t- Correct ID")
                            number_of_correct_ids += 1
                    else:
                        print("\t\t- Wrong ID")
                        if detected_object["real_object"]["class"] != "person":
                            number_of_wrong_ids += 1
                        self.detected_objects_previous[detected_object['real_object']['info']['ID']] = detected_object['object']['ID']
                else:
                    self.detected_objects_previous[detected_object['real_object']['info']['ID']] = detected_object['object']['ID']
                object_ids.append(detected_object['real_object']['info']['ID'])
            
            self.track_values['true_labels'].append(detected_object['real_object']['class_id'])
            self.track_values['predicted_labels'].append(detected_object['score'])

        self.tracking_accuracy += number_of_correct_ids
        self.tracking_id_switches += number_of_wrong_ids
        self.tracking_id_duplicates += number_of_duplicate_ids

        self.detection_accuracy += avg_score
        self.detection_accuracy_adjusted += avg_score_adjusted
        self.total_number_of_valid_detections += len(self.detected_objects)
        self.total_number_of_valid_detections_adjusted += number_of_detections_adjusted
        if len(self.detected_objects): avg_score /= len(self.detected_objects)
        if number_of_detections_adjusted > 0: avg_score_adjusted /= number_of_detections_adjusted
        print(f"Average score: {round(avg_score*100, 2)} %")
        print(f"Average score adjusted: {round(avg_score_adjusted*100, 2)} %")

        tmp_missed = 0
        if object_ids != []:
            for real_object in self.entries[self.next_entry_index-1]["objects"]:
                if real_object['info']['ID'] not in object_ids:
                    self.missed_detections += 1
                    tmp_missed += 1
            self.total_number_of_real_detections += len(self.entries[self.next_entry_index-1]["objects"])

        print(f"Missed detections: {tmp_missed}")
        try:
            print(f"Missed detections: {round(100*tmp_missed/len(self.entries[self.next_entry_index-1]['objects']), 1)} %")
        except Exception as e:
            print(e)
        print(f"False positive detections: {self.false_positives_detections - self.false_positives_detections_previous}")

        systemAnalytics = frameData['computational_data']
        frame_data = {
            'frame_number': frameData['frame_number'],
            'current_time': frameData['current_time']
        }
        if self.incident_accuracy+self.missed_incidents > 0:
            incident_accuracy_accumulated = round(100*self.incident_accuracy/(self.incident_accuracy+self.missed_incidents), 1)
        else:
            incident_accuracy_accumulated = 0
        frameInfo = {
            'frame_data': frame_data,
            'computational_data': systemAnalytics,
            'mean_detection_time': int(1000 * self.mean_detection_time / self.number_of_frames),
            'min_detection_time': int(1000 * self.min_detection_time),
            'max_detection_time': int(1000 * self.max_detection_time),
            'mean_tracking_time': int(1000 * self.mean_tracking_time / self.number_of_frames),
            'min_tracking_time': int(1000 * self.min_tracking_time),
            'max_tracking_time': int(1000 * self.max_tracking_time),
            'avg_score': avg_score,
            'number_of_valid_detections': len(self.detected_objects),
            'number_of_detections_adjusted': number_of_detections_adjusted,
            'number_of_correct_classes': number_of_correct_classes,
            'number_of_wrong_classes': number_of_wrong_classes,
            'number_of_correct_ids': number_of_correct_ids,
            'number_of_wrong_ids': number_of_wrong_ids,
            'number_of_duplicate_ids': number_of_duplicate_ids,
            'false_positive_detections': self.false_positives_detections - self.false_positives_detections_previous,
            'missed_detections': round(100*tmp_missed/len(self.entries[self.next_entry_index-1]['objects']), 1),
            'incident_accuracy_accumulated': incident_accuracy_accumulated,
            'queue_info': self.frame_queues
        }

        self.frame_queues = {}

        self.frameData.append(frameInfo)

        self.detected_objects = []

        self.false_positives_detections_previous = self.false_positives_detections
        
    def summary(self):
        text = "\n"
        jsonFormat = {}
        try:
            total_detections = self.total_number_of_valid_detections + self.false_positives_detections
            text += f"Scale: {int(self.scale*100)} %\n"
            text += f"Resolution: {int(self.width*self.scale)}x{int(self.height*self.scale)} px\n"
            text += f"Mean image enhancement time: {int(1000 * self.mean_image_enhancement_time / self.number_of_frames)} ms\n"
            text += "\n"
            text += f"Mean detection time: \t{int(1000 * self.mean_detection_time / self.number_of_frames)} ms\n"
            text += f"Min detection time: \t{int(1000 * self.min_detection_time)} ms\n"
            text += f"Max detection time: \t{int(1000 * self.max_detection_time)} ms\n"
            text += "\n"
            text += f"Mean tracking time: \t{int(1000 * self.mean_tracking_time / self.number_of_frames)} ms\n"
            text += f"Min tracking time: \t\t{int(1000 * self.min_tracking_time)} ms\n"
            text += f"Max tracking time: \t\t{int(1000 * self.max_tracking_time)} ms\n"
            text += "\n"
            text += f"Mean total time: \t{int(1000 * self.mean_total_time / self.number_of_frames)} ms\n"
            text += f"Min total time: \t{int(1000 * self.min_total_time)} ms\n"
            text += f"Max total time: \t{int(1000 * self.max_total_time)} ms\n"
            text += "\n"
            text += f"Mean fps: \t{round(self.mean_fps / self.number_of_frames, 1)}\n"
            text += f"Min fps: \t{int(self.min_fps)}\n"
            text += f"Max fps: \t{int(self.max_fps)}\n"
            text += "\n"
            text += f"False positive detections: \t{round(100*self.false_positives_detections/total_detections, 1)} %\n"
            text += f"Missed detections: \t\t\t{round(100*self.missed_detections/self.total_number_of_real_detections, 1)} %\n"
            text += "\n"
            text += f"Detection accuracy: \t\t\t{round(100*self.detection_accuracy/self.total_number_of_valid_detections, 1)} %\n"
            text += f"Detection accuracy adjusted: \t{round(100*self.detection_accuracy_adjusted/self.total_number_of_valid_detections_adjusted, 1)} %\n"
            text += f"Tracking accuracy: \t\t\t\t{round(100*self.tracking_accuracy/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %\n"
            text += f"Tracking ID duplicates: \t\t{round(100*self.tracking_id_duplicates/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %\n"
            text += f"Tracking ID switches: \t\t\t{round(100*self.tracking_id_switches/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %\n"
            text += "\n"
            text += f"Incident accuracy: \t{round(100*self.incident_accuracy/(self.incident_accuracy+self.missed_incidents), 1)} %\n"
            text += f"Missed incidents: \t{round(100*self.missed_incidents/(self.incident_accuracy+self.missed_incidents), 1)} %\n"
            text += f"False alarms: \t\t{round(100*self.false_alarms/total_detections, 1)} %\n"
            text += "\n"
            text += f"Total number of valid detections: {self.total_number_of_valid_detections}\n"
            text += f"Total number of detections: {total_detections}\n"
            
            # Own statistics
            text += "\n"
            text += f"Cars detected outside of the mask: {self.cars_outside_mask}\n"
            # text += f"Foreign objects detected inside the mask: \n"

            # Queue statistics:
            text += "\n"
            text += f"Queues detected: {self.queues_detected}\n"
            text += f"Amount queue changes: {self.queue_changes}\n"
            text += f"Average queue changes per car: {self.mean_queue_changes}\n"
            text += f"Max length of a queue: {self.max_queue_length}\n"
            text += f"Min length of a queue: {self.min_queue_length}\n"
            text += f"Avg length of a queue: {self.avg_queue_length}\n"

            text += f"Min queue detection time: {self.min_queue_detection_time}\n"
            text += f"Max queue detection time: {self.max_queue_detection_time}\n"
            text += f"Mean queue detection time: {self.mean_queue_detection_time}\n"

            # Json format:
            resolution = f"{int(self.width*self.scale)}x{int(self.height*self.scale)}"
            jsonFormat = {
                'scale': int(self.scale*100),
                'resolution': resolution,
                'mean_image_enhancement_time': int(1000 * self.mean_image_enhancement_time / self.number_of_frames),
                'mean_detection_time': int(1000 * self.mean_detection_time / self.number_of_frames),
                'min_detection_time': int(1000 * self.min_detection_time),
                'max_detection_time': int(1000 * self.max_detection_time),
                'mean_tracking_time': int(1000 * self.mean_tracking_time / self.number_of_frames),
                'min_tracking_time': int(1000 * self.min_tracking_time),
                'max_tracking_time': int(1000 * self.max_tracking_time),
                'mean_total_time': int(1000 * self.mean_total_time / self.number_of_frames),
                'min_total_time': int(1000 * self.min_total_time),
                'max_total_time': int(1000 * self.max_total_time),
                'mean_fps': round(self.mean_fps / self.number_of_frames, 1),
                'min_fps': int(self.min_fps),
                'max_fps': int(self.max_fps),
                'false_positive_detections': round(100*self.false_positives_detections/total_detections, 1),
                'missed_detections': round(100*self.missed_detections/self.total_number_of_real_detections, 1),
                'detection_accuracy': round(100*self.detection_accuracy/self.total_number_of_valid_detections, 1),
                'detection_accuracy_adjusted': round(100*self.detection_accuracy_adjusted/self.total_number_of_valid_detections_adjusted, 1),
                'tracking_accuracy': round(100*self.tracking_accuracy/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1),
                'tracking_id_duplicates': round(100*self.tracking_id_duplicates/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1),
                'tracking_id_switches': round(100*self.tracking_id_switches/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1),
                'incident_accuracy': round(100*self.incident_accuracy/(self.incident_accuracy+self.missed_incidents), 1),
                'missed_incidents': round(100*self.missed_incidents/(self.incident_accuracy+self.missed_incidents), 1),
                'false_alarms': round(100*self.false_alarms/total_detections, 1),
                'total_number_of_valid_detections': self.total_number_of_valid_detections,
                'total_number_of_detections': total_detections,
                'cars_detected_outside_mask': self.cars_outside_mask,
                'queues_detected': self.queues_detected,
                'min_queue_length': self.min_queue_length,
                'max_queue_length': self.max_queue_length,
                'avg_queue_length': self.avg_queue_length,
                'queue_changes': self.queue_changes,
                'mean_queue_changes': self.mean_queue_changes,
                'min_queue_detection_time': self.min_queue_detection_time,
                'max_queue_detection_time': self.max_queue_detection_time,
                'mean_queue_detection_time': self.mean_queue_detection_time,
                'frame_data': self.frameData,
                'detection_data': self.track_values
            }


            
        except Exception as e:
            print("Exception happened in performance evaluator")
            print(e)

        return text, jsonFormat
