import math
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
import time
from scipy.spatial.distance import pdist, squareform

class Evaluate_Incidents:
    def __init__(self, classes, colors=None, driving_direction=None):
        self.classes = classes
        self.colors = colors
        self.objects = {}
        self.driving_direction = driving_direction
        self.TTL = 240  # Number of frames before a track is removed
        self.PF = 2#7  # PF = Previous Frame: Number of frames used to determine direction
        self.STOPPED_DISTANCE = 3  # Distance in number of pixels from current to previous frame to determine stopped vehicle
        self.DIRECTION_THRESHOLD = 10  # Amount the x and y vectors can deviate when determining if vehicle is wrong-way driving
        self.min_number_of_frames = 2 # 24  # How many frames must there be to evaluate stopped vehicle
        self.update_number_of_frames = 2#12  # How often stopped vehicle should be evaluated
        self.min_number_of_driving_directions = 5 # How many driving directions needed to create a general vector for driving directions

        #Queue detections:
        self.queue_detection_radius = 100 # Radius value used in simple queue detection
        self.dbscan_eps = 1 # Epsilon value used for the DBSCAN clustering algorithm
        self.min_queue_size = 3 # Minimum number of vehicles in proximity to a core point to be considered a queue
        self.common_driving_direction = (93, -130) #(133, -100) #(93, -130) # This is the mathematical vector definition for direction of traffic flow (x, y)
        self.secondary_driving_direction = (133, -100) # NOTE: Static value that have no actual function currently due to not being implemented
        self.driving_direction_margin = 10 # Tolerance zone for being considered driving in the same lane
        self.queues = {}
        # self.queue_map = {}
    
    @property
    def colors(self):
        return self._colors
    
    @colors.setter
    def colors(self, colors):
        colors_default = {"alarm": (255,128,128), "ok": (128,128,255), "queue": (15,255,80)}
        if colors and colors.get("alarm") and colors.get("ok"):
            colors_default = colors
        self._colors = colors_default
    
    @property
    def driving_direction(self):
        return self._driving_direction
    
    @driving_direction.setter
    def driving_direction(self, driving_direction):
        # Driving direction should be defined with an upstream and downstream direction
        # Each direction should be defined as a vector: [x, y]
        if driving_direction is None:
            driving_direction = {"Upstream": [], "Downstream": []}
        if driving_direction.get("Upstream") is None:
            driving_direction["Upstream"] = []
        if driving_direction.get("Downstream") is None:
            driving_direction["Downstream"] = []
        self._driving_direction = driving_direction

    def purge(self, frame_number):
        if frame_number % 24 != 0:
            return
        dict_of_objects = self.objects.copy()
        for object in dict_of_objects:
            if dict_of_objects[object]["last_frame"] < frame_number - self.TTL:
                del self.objects[object]

    # Can be used to calculate direction based on center points from several frames
    def simple_linear_regression(self, track_id, frame_number):
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= 5:
            return None, None

        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        if frame_number % 12 != 0:
            direction = self.objects[track_id].get("direction")
            if direction:
                next_point_x = current_point[0] + direction["distance"]
                next_point_y = direction["alpha"] + direction["beta"] * next_point_x
                next_point = (int(next_point_x), int(next_point_y))
                
                return current_point, next_point
        
        if n > 10:
            n = 10
        center_points = track["center_points"][-n:]
        
        x_sum = 0
        y_sum = 0
        for center_point in center_points:
            x_sum += center_point[0]
            y_sum += center_point[1]
            
        x_mean = x_sum / n
        y_mean = y_sum / n

        numerator = 0
        denominator = 0
        for center_point in center_points:
            x = center_point[0]
            y = center_point[1]
            numerator = (x - x_mean) * (y - y_mean)
            denominator = (x - x_mean) ** 2
        
        try:
            beta = numerator / denominator
        except Exception as e:
            print(e)
            beta = 0
        
        alpha = y_mean - beta * x_mean
        
        d = 1
        if (center_points[-1][0] - center_points[-2][0]) < 0:
            d = -1
        distance = d * math.sqrt((center_points[-1][0] - center_points[-2][0])**2 + (center_points[-1][1] - center_points[-2][1])**2)
        next_point_x = center_points[-1][0] + distance
        next_point_y = alpha + beta * next_point_x
        next_point = (int(next_point_x), int(next_point_y))
        
        self.objects[track_id]["direction"] = {"alpha": alpha, "beta": beta, "distance": distance}
        return current_point, next_point

    # Can be used to calculate direction based on center points from current and previous frame
    def simple_direction(self, track_id, frame_number):
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= 8:
            return None, None

        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        if frame_number % 12 != 0:
            direction = self.objects[track_id].get("direction")
            if direction:
                x_vector = direction["x_vector"]
                y_vector = direction["y_vector"]
                length = direction["length"]
                next_point = (int(current_point[0] + x_vector * length), int(current_point[1] + y_vector * length))
                
                return current_point, next_point
        
        previous_point = track["center_points"][-self.PF]

        x_vector = current_point[0] - previous_point[0]
        y_vector = current_point[1] - previous_point[1]
        length_vector = math.sqrt(x_vector**2 + y_vector**2)
        try:
            x_vector /= length_vector
            y_vector /= length_vector
        except Exception as e:
            print(e)
            return None, None

        length= 50

        next_point = (int(current_point[0] + x_vector * length), int(current_point[1] + y_vector * length))

        self.objects[track_id]["direction"] = {"length": length, "x_vector": x_vector, "y_vector": y_vector}
        return current_point, next_point
    
    # TODO: Implement angular speed (Current implementation only contains a simple solution for distance from last center point to current point)
    def simple_speed(self, track_id):
        if track_id not in self.objects:
            print("Track id not in self.objects")
            return -1
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= self.min_number_of_frames:
            print("min number of frames not met")
            return -1

        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        previous_point = track["center_points"][-self.PF]

        speed = math.sqrt((current_point[0] - previous_point[0])**2 + (current_point[1] - previous_point[1])**2)

        # self.objects[track_id]["speed"] = distance
        return speed

    def pedestrian(self, class_name):
        if class_name == "person":
            return True
        return False
    
    def stopped_vehicle(self, track_id, frame_number):
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= self.min_number_of_frames:
            return False

        if frame_number % self.update_number_of_frames != 0:
            stopped = self.objects[track_id].get("stopped")
            if stopped:
                return True
            return False
        
        current_point = (int(track["center_points"][-1][0]), int(track["center_points"][-1][1]))
        previous_point = track["center_points"][-self.PF]

        distance = math.sqrt((current_point[0] - previous_point[0])**2 + (current_point[1] - previous_point[1])**2)

        if distance <= self.STOPPED_DISTANCE:
            self.objects[track_id]["stopped"] = True
            return True
        self.objects[track_id]["stopped"] = False
        return False
    
    def wrong_way_driving(self, track_id, frame_number, current_point, next_point, lane="Upstream"):
        if len(self.driving_direction.get(lane)) <= 0:
            return False
            
        track = self.objects[track_id]
        n = len(track["center_points"])
        if n <= self.min_number_of_frames:
            return False

        if frame_number % self.update_number_of_frames != 0:
            wrong_way = self.objects[track_id].get("wrong_way")
            if wrong_way:
                return True
            return False

        if not current_point or not next_point:
            return False
        
        # print("Next point: ", next_point)
        # print("Current point: ", current_point)
        
        vehicle_direction = [next_point[0] - current_point[0], next_point[1] - current_point[1]]
        lane_direction = self.driving_direction.get(lane)

        if abs(lane_direction[0] - vehicle_direction[0]) < self.DIRECTION_THRESHOLD and abs(lane_direction[1] - vehicle_direction[1]) < self.DIRECTION_THRESHOLD:
            self.objects[track_id]["wrong_way"] = False
            return False
        self.objects[track_id]["wrong_way"] = True
        return True
    # TODO: Claim Credit
    def same_lane_driving(self, center_point1, center_point2):
        lane_vector = [center_point2[0] - center_point1[0], center_point2[1] - center_point1[1]]

        angle_radians = math.atan2(lane_vector[1], lane_vector[0])
        angle_degrees = math.degrees(angle_radians)

        # print(self.common_driving_direction)    
        driving_direction_angle = math.degrees(math.atan2(self.common_driving_direction[0], self.common_driving_direction[1]))
        angle_difference = abs(angle_degrees - driving_direction_angle)

        angle_difference = min(angle_difference, 360 - angle_difference)
        print("Angle difference: ", angle_difference)

        return angle_difference <= self.driving_direction_margin or 180 - angle_difference <= self.driving_direction_margin
    
    def cars_furthest_apart(coordinates, cluster_indices):
        pairwise_distance = squareform(pdist(coordinates[cluster_indices]))

        furthest_pair_indices = np.unravel_index(np.argmax(pairwise_distance, axis=None), pairwise_distance.shape)

        return cluster_indices[furthest_pair_indices[0]], cluster_indices[furthest_pair_indices[1]]
    
    def cars_furthest_apart_simple(self, cars):
        max_distance = 0
        car_pair = None

        for i in range(len(cars)):
            for j in range(len(cars)):
                if i == j: 
                    continue
                distance = np.sqrt((cars[i][0] - cars[j][0])**2 + (cars[i][1] - cars[j][1])**2)

                if distance > max_distance:
                    max_distance = distance
                    car_pair = [cars[i][2], cars[j][2]]

        return car_pair
    
    # TODO: Implement / Claim Credit
    def queue(self, frame_number):
        start_time = time.time()
        queue_map = {} # Map of all detected queues
        track_to_queue_map = {} # Simple map for tracking the lane a track belongs to {trackId: laneId}
        furthest_apart = {} # Map of the cars furthest apart in each lane {laneId: [CarId1, CarId2]}
        amount_of_queues = 0
        filtered_tracks = {key: val for key, val in self.objects.items() if frame_number - val.get('last_frame') <= 1}
        
        for track_id in self.objects:
            track = self.objects[track_id]
            class_to_id = {0: 'None',1: 'car', 2: 'person', 3: 'truck', 4: 'bus', 5: 'bike', 6: 'motorbike', 10: 'Road anomaly'}

            print(track["class"])
            if track["class"] in class_to_id:
                track["class"] = class_to_id[track["class"]]
            try:
                track["class"].upper()
            except AttributeError as e:
                track["class"] = "NONE"

        for track_id in filtered_tracks:
            track = self.objects[track_id]

            dif = frame_number - track['last_frame']
            if dif > 1:
                continue

            if track["speed"] == -1 or len(track["center_points"]) < 1:
                continue

            if track["speed"] > 10:
                continue

            vehicles = ["CAR", "BUS", "TRUCK", "STOPPED VEHICLE"]
            if track["class"].upper() not in vehicles:
                continue

            cx1, cy1 = track["center_points"][-1] # Center point
            if track_id in track_to_queue_map:
                lane = track_to_queue_map[track_id]
            else:
                if track_to_queue_map == {}:
                    lane = 1
                else:
                    lane = len(track_to_queue_map) + 1

            for track_id2 in filtered_tracks:
                track2 = self.objects[track_id2]
                if track_id == track_id2:
                    continue
                if track2["speed"] == -1 or len(track2["center_points"]) < 2:
                    continue

                if track2["class"].upper() not in vehicles:
                    continue

                dif2 = frame_number - track2['last_frame']
                if dif2 > 1:
                    continue

                x, y = track2["center_points"][-1]
                distance = np.sqrt((x - cx1)**2 + (y - cy1)**2)

                if distance > self.queue_detection_radius or distance < 10:
                    continue
                if not self.common_driving_direction:
                    continue

                if self.same_lane_driving((cx1, cy1), (x, y)):
                    trackInfo = {"center_point": (cx1, cy1), "speed": track["speed"], "track": track}
                    trackInfo2 = {"center_point": (y, x), "speed": track2["speed"], "track": track2}
                    print(f"These cars are same lane driving {track_id} and {track_id2}.")
                    if track_id not in track_to_queue_map:
                        if lane not in queue_map:
                            queue_map[lane] = {}
                        queue_map[lane][track_id] = trackInfo
                        queue_map[lane][track_id2] = trackInfo2
                        track_to_queue_map[track_id] = lane
                        track_to_queue_map[track_id2] = lane
                    else:
                        if lane not in queue_map:
                            queue_map[lane] = {}
                        queue_map[lane][track_id2] = trackInfo2
                        track_to_queue_map[track_id2] = lane
        

        for lane in queue_map:
            cars = []
            lane_id = lane
            lane = queue_map[lane]
            for car in lane:
                cars.append((lane[car]["center_point"][0], lane[car]["center_point"][1], car))
            
            found_in_lane = None
            for l in self.queues:
                counter = 0
                for car in lane:
                    if car in self.queues[l]:
                        counter += 1
                if counter >= self.min_queue_size:
                    found_in_lane = l
                    break

            if not found_in_lane:
                n = len(self.queues) + 1
                self.queues[n] = [c[2] for c in cars]
                    

            print(f"Cars lenght: {len(cars)}")
            if len(cars) < self.min_queue_size:
                # continue
                print(":)")
            furthest_apart[lane_id] = self.cars_furthest_apart_simple(cars)
        
        # lanes = []
        # print("Track to queue map: ", track_to_queue_map)
        # for lane in furthest_apart:
        #     lanes.append(furthest_apart[lane])
        laneDetails = {}
        # print("Track to queue map: ", track_to_queue_map)
        for lane in furthest_apart:
            queue_lane = queue_map[lane]
            # print(f"Queue_lane: {queue_lane}")
            car_ids = furthest_apart[lane]
            if not self.same_lane_driving(self.objects[car_ids[0]]["center_points"][-1], self.objects[car_ids[1]]["center_points"][-1]):
                continue
            tracks = {lane: [car, car_info["track"]] for car, car_info in queue_lane.items()}
            laneDetails[lane] = {"furthest_apart": furthest_apart[lane], "tracks": tracks}
        
        end_time = time.time()
        total_time = end_time - start_time

        return laneDetails, (self.queues, total_time)



    
    # TODO: Implement / Claim Credit
    def queue_dbscan(self):
        # features = np.array([])
        features = [] # features is a 2D array containing the centerpoint and speed of a vehicle
        track_map  = [] # Track_map maps the track with the key feature_id to the track and track_id since not all tracks make it to the feature array
        feature_counter = 0 # Serves as the current index of the feature array
        for track_id in self.objects:
            track = self.objects[track_id]
            # print("Track: ", track)
            if track["speed"] and len(track["center_points"]) > 2:
                x1, y1 = track["center_points"][-1]
                # x2, y2 = track["center_points"][-1]
                # features.append([x1, y1, x2, y2, track["speed"]])
                features.append([x1, y1, track["speed"]])
                track_map.append({"track_id": track_id, "track": track})
                feature_counter += 1

            # print("Features: \n", features)
        if len(features) > 4:
            features = np.array(features)
            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_queue_size)
            clusters = dbscan.fit_predict(features)

            unique_clusters = set(clusters) - {-1}

            refined_clusters = []

            for cluster_id in unique_clusters:
                indices = np.where(clusters == cluster_id)[0]
                cluster_features = features[indices]
                cluster_tracks = track_map[indices]

                G = nx.Graph()

                for index in indices:
                    G.add_node(index)
                
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if self.same_lane_driving(cluster_tracks[i]["track"]["center_points"][-1], cluster_tracks[j]["track"]["center_points"][-1]):
                            G.add_edge(indices[i], indices[j])
                
                for component in nx.connected_components(G):
                    refined_clusters_indices = list(component)
                    refined_clusters.append(refined_clusters_indices)
            
            # refined_clusters_track_ids is a list of all clustered vehicles with their original track id
            refined_clusters_track_ids = []
            for cluster in refined_clusters:
                track_ids = [track_map[i] for i in cluster]
                refined_clusters_track_ids.append(track_ids)
            
            coordinates = np.array([...])
            cluster_assignments = [...]
            for cluster_id in refined_clusters:
                cluster_inices = [i for i, x in enumerate(cluster)]
                
            # print("Refined clusters: \n", refined_clusters)


            # print("Clusters: \n", clusters)

    def evaluate(self, track, frame_number, eval_queue=False):
        class_name = track.get_class()
        text = f"{class_name} - {track.track_id}"
        color = self.colors["ok"]
        bbox = track.to_tlbr()
        center_point = ((int(bbox[0]) + (int(bbox[2]) - int(bbox[0])) / 2), int(bbox[1]) + (int(bbox[3]) - int(bbox[1])) / 2)
        speed = self.simple_speed(track.track_id)
        # print(f"Speed: {speed}")
        if track.track_id in self.objects:
            self.objects[track.track_id]["center_points"].append(center_point)
            self.objects[track.track_id]["last_frame"] = frame_number
            self.objects[track.track_id]["speed"] = speed
            self.objects[track.track_id]["class"] = class_name
        else:
            self.objects[track.track_id] = {"center_points": [center_point], "last_frame": frame_number, "speed": speed, "class": class_name}
        
        # Used to determine vehicle direction: 
        #   Current_point is the current center location of the vehicle
        #   Next point is calculated by creating a vector from the current center point and the previous center point, and then multiplying it with a length. (Used to draw an arrow in the vehicle direction)
        current_point, next_point = self.simple_direction(track.track_id, frame_number)

        # if len(self.driving_direction) >= self.min_number_of_driving_directions:
        # if current_point:
        #     self.common_driving_direction = (next_point[0] - current_point[0], next_point[1] - current_point[1])




        if self.pedestrian(class_name):
            color = self.colors["alarm"]
            text = "INCIDENT: Pedestrian"
            current_point, next_point = None, None
        elif self.stopped_vehicle(track.track_id, frame_number):
            color = self.colors["alarm"]
            text = "INCIDENT: Stopped vehicle"
            current_point, next_point = None, None
        elif self.wrong_way_driving(track.track_id, frame_number, current_point, next_point):
            print("WRONG WAY DRIVER!!!")
            color = self.colors["alarm"]
            text = "INCIDENT: Wrong-way driver"
        # TODO: Introduce a queue state to this if statement

        # print(self.driving_direction["Upstream"])
        # TODO: Finish implementation
        if eval_queue:
            # print("eval_queue frame_number:", frame_number)
            queue_details, queue_stats = self.queue(frame_number)
            # print("Queue details: ", queue_details)
            # return color, text, current_point, next_point, self.common_driving_direction, queue_details
            if self.driving_direction["Upstream"]:
                return color, text, current_point, next_point, (self.driving_direction["Upstream"][0], self.driving_direction["Upstream"][1]), [queue_details, queue_stats]
            else:
                return color, text, current_point, next_point, ([]), [queue_details, queue_stats]
        
        # return color, text, current_point, next_point, self.common_driving_direction
        if self.driving_direction["Upstream"]:
            return color, text, current_point, next_point, (self.driving_direction["Upstream"][0], self.driving_direction["Upstream"][1])
        else:
            return color, text, current_point, next_point, ([])
            
        
