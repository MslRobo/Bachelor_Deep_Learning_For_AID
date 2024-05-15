import os
import sys
import glob
import time
import GPUtil
import psutil

PATH_TO_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH_TO_THIS_FILE + '\\tools\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\tools\\deep_sort')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\tensorflowapi\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\tensorflowapi\\research\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\tensorflowapi\\research\\object_detection')

import cv2
import json
import numpy as np
from tools.detection_model import Detection_Model
from tools.tracking_model import Tracking_Model
from tools.incident_evaluator import Evaluate_Incidents
from tools.performance_evaluator import Evaluate_Performance
import argparse
from tools.visualize_objects import draw_rectangle, draw_text, draw_line, draw_parallelogram
from tools.tunnel_manager import Tunnel_Manager


"""
Parser setup has been influenced by the implementation used by Alexander Vedvik in his bachelor thesis that
layed the ground work for this thesis.
"""
parser = argparse.ArgumentParser(
    description="Realtime object detection and tracking"
)
parser.add_argument("-m",
                    "--model",
                    help="Choose perferred detection model to be utilized",
                    type=str)
parser.add_argument("-c",
                    "--checkpoint",
                    help="Choose checkpoint number to be used, 3 will be used as default",
                    type=str)
parser.add_argument("-p",
                    "--pretrained",
                    help="Choose whether or not to use a pre-trained model or not. 1 = Ture, 0 = False (0 is defautt)",
                    type=str)
parser.add_argument("--skip_frames",
                    help="Choose number of frames to skip",
                    type=int)
parser.add_argument("-r",
                    "--resize",
                    help="Define a scale factor to resize the input video",
                    type=float)
parser.add_argument("--resolution",
                    help="Set the resolution wanted for the input expecting height, width will be automatically calculated",
                    type=int)
parser.add_argument("-b",
                    "--brightness",
                    help="Set the brightness level for the input in percentage expected integer in range -100 to 100",
                    type=int)
parser.add_argument("-n",
                    "--noise",
                    help="Set the noise level to simulate a bad quality feed accepted noise types are: gauss, salt, speckle",
                    type=str)
parser.add_argument("-t",
                    "--tracking",
                    help="Choose tracking model to utilize, DeepSort will be used as default")
parser.add_argument("-q",
                    "--queue",
                    help="Choose queueing algorithm, DBScan or Simple")
parser.add_argument("-f",
                    "--file",
                    help="Define the name of the saved file",
                    type=str)
parser.add_argument("-i",
                    "--img_enh",
                    help="Specify how the image should be enhanced. By default no enhancements will be applied",
                    type=str)
parser.add_argument("-s",
                    "--source",
                    help="Specify the source of the video to be analyzed",
                    type=str)
parser.add_argument("--mode",
                    help="Specify the mode of the application, by default live tracking will be used, other modes include analysis and training",
                    type=str) 
parser.add_argument("--show",
                    help="Show a live feed of the tracking process, values are 1 = true, 0 = false, by default live feed will be disabled.",
                    type=int)
parser.add_argument("--datamode",
                    help="Specify how the source should be handled. json expects a json file with session configurations, and mp4 expects a single mp4 file. Default is mp4",
                    default="mp4",
                    type=str)
parser.add_argument("--filetype",
                    default="mp4",
                    help="Specify what file type the feed is should be jpg or mp4")
parser.add_argument("--iterations",
                    help="Specify how many iterations should be done, each iteration changes the value of brightness or noise depending on which is active",
                    type=int)
parser.add_argument("-a",
                    "--analysis",
                    help="Should statistics analysis be ran after completion (0 (default) or 1) (Should not be used in its current state)",
                    type=int,
                    default=0)
parser.add_argument("--downscale",
                    help="1 if downscaling should be performed, downscaling from max resolution down to 360p at the lowest iteration",
                    default=0,
                    type=int)

args = parser.parse_args()

"""
Commands used in testing:
python liveTrack.py -s StandardAnalysis.json --file StatisticsTest00 --datamode json --filetype jpg --show 1
python liveTrack.py -s StandardAnalysis.json --file StatisticsTest00 --datamode json --filetype jpg --iterations 21 --show 1
"""

def argReplacement(file):
    if not args.model:
        args.model = file['model']
    if not args.checkpoint:
        args.checkpoint = file['checkpoint']
    if not args.pretrained:
        args.pretrained = file['pretrained']
    if not args.tracking:
        args.tracking = file['tracking']
    args.skip_frames = file['skip_frames']
    args.resize = file['resize']
    args.noise = file['noise']
    args.file = file['file']
    args.img_enh = file['img_enh']
    args.mode = file['mode']
    args.show = file['show']
    args.queue = file['queue']
    args.filetype = file['filetype']
    args.datamode = file['datamode']

# Extracts the json data
def extractJSONFile(jsonFile):
    dataCollectionDir = r'.\\SessionConfigurations'
    dataCollectionFile = os.path.join(dataCollectionDir, jsonFile)
    
    with open(dataCollectionFile, 'r') as f:
        config = json.load(f)

    return config

def main(video, dirNames, iterationOptions=None, new_resolution=False):
    datasets = []
    percentDone = 0
    sourceDir = r'.\\data\\rawData'
    sourceFile = os.path.join(sourceDir, video)
    datasetDir = r'.\\data\\incidents'
    baseOutputDir = r'.\\data\\output'
    maskDir = r'.\\data\\tunnel_data\\masks'

    if args.datamode == "json":
        if dirNames['sessionDir']:
            baseOutputDir = os.path.join(baseOutputDir, dirNames['sessionDir'])
            baseOutputDir = os.path.join(baseOutputDir, dirNames['runDir'])
        else:
            baseOutputDir = os.path.join(baseOutputDir, dirNames['runDir'])

        if not os.path.exists(baseOutputDir):
            os.makedirs(baseOutputDir)

    if args.filetype == "mp4" or not args.filetype:
        datasets.append({"dataset": "selectedVideo", "video": sourceFile})
        maskPath = os.path.join(maskDir, video.split(".")[0] + ".png")
        mask = maskPath
    else:
        datasetTunnelDir = os.path.join(datasetDir, video)
        # print("Dataset: ", datasetTunnelDir, " Exists: ", os.path.exists(datasetTunnelDir))
        image_dir = os.path.join(datasetTunnelDir, "images")
        anno_dir = os.path.join(datasetTunnelDir, "annotations.json")
        datasets.append({"dataset": video + "self_annotated", "images": image_dir, "annotations": anno_dir})
        # print("Dataset: ", datasets)
        mask = os.path.join(maskDir, video + ".png")

    # print("Mask: ", mask)

    model_filename = os.path.join(PATH_TO_THIS_FILE, 'tools/model_data/mars-small128.pb')

    paths = {
        "CHECKPOINT_PATH": "./training/models/ssd_mobnet/",
        "PIPELINE_CONFIG": "./training/models/ssd_mobnet/pipeline.config", 
        "LABELMAP": "./training/annotations/label_map.pbtxt",
        "DEEPSORT_MODEL": model_filename
    }

    image_enhancement_methods = ["gray_linear", "gray_nonlinear", "he", "retinex_ssr", "retinex_msr", "mask"]
    models = ["yolov5", "yolov5_trained",  "yolov8", "yolov7"]
    tracking_models = ["DeepSort"]
    noise_types = ["gauss", "salt", "speckle"]
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}

    model_name = "yolov5"
    if args.model in models:
        paths["CHECKPOINT_PATH"] = "./training/models/" + args.model + "/"
        paths["PIPELINE_CONFIG"] = "./training/models/" + args.model + "/pipeline.config"
        model_name = args.model

    tracking_model_name = "DeepSort"
    if args.tracking and args.tracking in tracking_models:
        tracking_model_name = args.tracking

    ckpt_number = "3"
    if args.checkpoint is not None:
        ckpt_number = args.checkpoint

    filename = ""
    if args.file is not None:
        filename = args.file

    image_enhancement = "None"
    # print("Args Img_enh: ", args.img_enh)
    if args.img_enh is not None and args.img_enh in image_enhancement_methods:
        image_enhancement = args.img_enh

    brightness = None
    if args.brightness is not None:
        brightness = args.brightness
        brightness -= 10*iterationOptions['current_iteration_index']

    if args.queue == "":
        args.queue = "simple"

    resolution = None
    if args.resolution is not None:
        resolution = args.resolution

    noise_type = None
    if args.noise is not None and args.noise in noise_types:
        noise_type = args.noise

    if args.pretrained == "1":
        paths["CHECKPOINT_PATH"] = "./training/pre-trained-models/" + args.model + "/checkpoint/"
        paths["PIPELINE_CONFIG"] = "./training/pre-trained-models/" + args.model + "/pipeline.config"
        paths["LABELMAP"] = "./training/annotations/mscoco_label_map.pbtxt"
        model_name = "Pretrained"
        ckpt_number = "0"
        classes = {"car": "3", "truck": "8", "bus": "6", "bike": "2", "person": "1", "motorbike": "4"}

    skip_frames = 1
    if args.skip_frames:
        skip_frames = int(args.skip_frames)

    resize = 1
    if args.resize:
        resize = float(args.resize)

    tunnel_manager = Tunnel_Manager()
    tunnel_data = tunnel_manager.get_tunnel_data(video.split(".")[0])
    driving_direction = None
    if tunnel_data:
        driving_direction = tunnel_data["driving_direction"]

    model = Detection_Model(model_name, classes, paths, ckpt_number)
    tracker_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
    evaluater = Evaluate_Incidents(classes, driving_direction=driving_direction)
    if args.filetype == "mp4":
        pe = Evaluate_Performance("Video", datasets, classes, model, tracker_model, mask=mask, noise_type=noise_type)
    else:
        pe = Evaluate_Performance("Images", datasets, classes, model, tracker_model, mask=mask, noise_type=noise_type)

    # if args.file != None or args.mode == "analysis":
    if args.mode == "analysis":
        if args.filetype == "mp4":
            cap = cv2.VideoCapture(sourceFile)
        else:
            ret, frame, new_video, mask = pe.read(resize)
            cap = frame
        length = 0
        if cap and not cap.isOpened() and args.filetype == "mp4":
            print("Error: Could not open video")
        elif args.filetype == "mp4":
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print("Width: %d, Height: %d" % (width, height))
        elif cap is not None:
            height, width = cap.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # if args.mode == "analysis":
        output_video_path = os.path.join(baseOutputDir, f'{video}_{new_resolution}.mp4')
        # else:
        #     output_video_path = os.path.join(baseOutputDir, (args.file + ".mp4"))
        frame_rate = 20
        if width is not None and height is not None:
            frame_size = (int(width), int(height))
        # print(frame_size) 

        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)
    else:
        if os.path.exists(datasets[0]["images"]):
            # print("In the right place")
            path = datasets[0]["images"]
            png_files = glob.glob(f"{path}/*.png")
            jpg_files = glob.glob(f"{path}/*.jpg")
            length = len(png_files) + len(jpg_files)
        else:
            print("In the else")
            length = 0

    resolutions = {
        '720': {'width': 1280, 'height': 720},
        '648': {'width': 1152, 'height': 648},
        '576': {'width': 1024, 'height': 576},
        '360': {'width': 640, 'height': 360}
    }
    if not new_resolution:
        res = {'width': 1280, 'height': 720}
    else:
        res = resolutions[new_resolution]
    frame_size = (int(res['width']), int(res['height']))
    output_video_path = f'{baseOutputDir}/{video}_{res["height"]}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    frame_rate = 20
    print(output_video_path)
    # raise ValueError
    # out = cv2.VideoWriter("loloutput.mp4", fourcc, frame_rate, frame_size)
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)
        # print("Length: ", length)


    frame_number = 0
    # print(f"Running Video {video} in the {dirNames['runDir']} configuration")
    while True:
        ret, frame, new_video, mask = pe.read(resize, new_resolution)

        if frame is None:
            break

        # print("Frame: ", frame)
        # print("Frame type: ", type(frame))
        if isinstance(frame, tuple):
            # print("This is a tuple")
            frame = frame[1]
            # print("Frame[1] type: ", type(frame))
        if args.filetype != "mp4":
            height, width = frame.shape[:2]
        frame_number += 1
        if frame_number % skip_frames != 0:
            continue

        if ret:
            frame = pe.image_enhancement(frame, image_enhancement, mask=mask, brightness=brightness)
        else:
            print('Video has ended!')
            break

        if new_video:
            new_tracking_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
            pe.tracking_model = new_tracking_model
        
        pe.detect_and_track(frame)

        evaluater.purge(frame_number)

        counter = 0
        queue_details = None
        tracks = pe.get_tracks()
        # queue_map = {1: [car1, car2, car3], 2: [car1, car2, car3]} where 1, 2, etc is the queue index and car1, car2 is the id of the car track
        queue_map = {}
        queue_colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}
        for track in pe.get_tracks():
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                
            # print("Track: \n", track)
            # print("Tracks: \n", pe.get_tracks())
            
            if counter == 0:
                color, text, current_point, next_point, driving_dir, queue_details = evaluater.evaluate(track, frame_number, True)
                queue_stats = queue_details[1][0]
                queue_time = queue_details[1][1]
                queue_details = queue_details[0]
                if queue_details:
                    for lane in queue_details:
                        queue_map[lane] = queue_details[lane]["tracks"]
                
                if queue_stats != {}:
                    pe.queue_performance(queue_stats, queue_time)
            else:
                color, text, current_point, next_point, driving_dir = evaluater.evaluate(track, frame_number)

            if args.filetype == 'jpg':
                pe.performance(track, text)
            draw_rectangle(frame, track, color)
            
            if queue_details:
                found = False
                for lane, track_list in queue_map.items():
                    # print(f"Track_list: {track_list}")
                    for _, trackInfo in track_list.items():
                        trackId = trackInfo[0]
                        # print(f"track_info: {trackInfo}")
                        # print(f"track_id: {track.track_id}")
                        # print(f"trackId = {trackId}")
                        if track.track_id == trackId:
                            found = True
                            if lane in queue_colors:
                                color = queue_colors[lane]
                            else:
                                color = (255, 255, 255)
                            break
                print(f"found: {found}")
                if found:
                    print(":)")
                    # draw_rectangle(frame, track, color)
            
            
            # draw_text(frame, track, text)
            if current_point and next_point:
                draw_line(frame, current_point, next_point)
            if driving_dir:
                draw_line(frame, (int(width/2), int(height/2)), (int((width/2)+driving_dir[0]), int((height/2)+driving_dir[1])))

            if counter == 0:
                if queue_details:
                    # lanes = queue_details["furthest_apart"]
                    for lane in queue_details:
                        lane_details = queue_details[lane]["furthest_apart"]
                        for track in tracks:
                            if track.track_id == lane_details[0]:
                                car1 = track
                            if track.track_id == lane_details[1]:
                                car2 = track
                        try:
                            car1 = car1.to_tlwh()
                            car2 = car2.to_tlwh()
                        except AttributeError as e:
                            print("EXEPTION")
                            continue

                        if car1[1] > car2[1]:
                            draw_parallelogram(frame, (car2[0], car2[1]), (car2[0] + car2[2], car2[1]), (car1[0], car1[1] + car1[3]), (car1[0] + car1[2], car1[1] + car1[3]))
                        else:
                            draw_parallelogram(frame, (car1[0], car1[1]), (car1[0] + car1[2], car1[1]), (car2[0], car2[1] + car2[3]), (car2[0] + car2[2], car2[1] + car2[3]))

        
            counter += 1

        current_time = time.time() - timeStart
        gpu = GPUtil.getGPUs()
        gpu = gpu[0]
        cpu_usage = psutil.cpu_percent(interval=None)
        computational_data = {'time': current_time, 'gpu_load_percent': gpu.load*100, 'gpu_memory_used': gpu.memoryUsed, 'gpu_memory_usage': gpu.memoryUtil*100, 'cpu_usage': cpu_usage}
        frameData = {'frame': frame, 'frame_number': frame_number, 'current_time': current_time, 'computational_data': computational_data}

        if args.filetype != 'mp4':
            pe.status(frameData)
                
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(result)
        try:

            # if args.file != None or args.mode == "analysis":
            # if args.mode == "analysis":

            if args.show == 1:
                cv2.imshow("Output Video", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print("BREAKING OUT")
            break
        
        # result = np.asarray(frame)
        # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if args.file != None or args.mode == "analysis":
        #     out.write(result)

        # if args.show == 1:
        #     cv2.imshow("Output Video", result)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        ########################################################################
        percentStorage = percentDone
        if (frame_number / length) * 100 - percentStorage > 1:
            percentDone = np.floor((frame_number / length) * 100)
            print(f"Percent completed {percentDone}%")
    cv2.destroyAllWindows()

    summary, jsonFormat = pe.summary()
    print(summary)
    if filename != '':
        if iterationOptions != None:
            output_file = baseOutputDir + "/" + video.split(".")[0] + "_" + str(iterationOptions["current_iteration_index"]) + ".txt"
            output_json = baseOutputDir + "/" + video.split(".")[0] + "_" + str(iterationOptions["current_iteration_index"]) + ".json"
            print(":)")
        elif new_resolution:
            output_file = baseOutputDir + "/" + video.split(".")[0] + "_" + new_resolution + ".txt"
            output_json = baseOutputDir + "/" + video.split(".")[0] + "_" + new_resolution + ".json"
        else:
            output_file = baseOutputDir + "/" + video.split(".")[0] + ".txt"
            output_json = baseOutputDir + "/" + video.split(".")[0] + ".json"
        with open(output_file, "w") as file:
            output = f"Image enhancement: {image_enhancement}\n"
            output += f"Detection: {model_name}\n"
            output += f"Tracking: {tracking_model_name}\n"
            output += f"noise_type: {noise_type}\n"
            output += f"brightness: {brightness}\n"
            output += summary
            file.write(output)

        with open(output_json, "w") as file:
            versionInfo = {
                'image_enhancement': image_enhancement,
                'detection': model_name,
                'tracking': tracking_model_name,
                'noise_type': noise_type,
                'brightness_level': brightness
            }
            outputJson = {**versionInfo, **jsonFormat}
            json.dump(outputJson, file)

if __name__ == '__main__':

    if args.datamode == 'json':
        
        # videoList = extractVideoFile(args.source)
        config = extractJSONFile(args.source)

        # print(videoList)

        argsStorage = args
        runConfig = []
        if config['type'] == 'SessionConfig':
            dirName = config['dir']
            
            i = 1
            while os.path.exists(os.path.join(r'.\\data\\output', dirName)):
                dirName = config['dir'] + str(i)
                i += 1
            os.mkdir(os.path.join(r'.\\data\\output', dirName))

            for object in config['runConfig']:
                runConfig.append(extractJSONFile(object))
        else:
            # runConfig = [extractJSONFile(config)]
            runConfig = config
            dirName = runConfig['dir']

            i = 1
            while os.path.exists(os.path.join(r'.\\data\\output', dirName)):
                dirName = runConfig['dir'] + str(i)
                i += 1
            os.mkdir(os.path.join(r'.\\data\\output', dirName))
            runConfig = [config]
        
        print(runConfig)

        # Runs the main program for each video listed in the json file
        # for video in videoList:
        timeStart  = time.time()
        for file in runConfig:
            # print(file)
            for video in file['data']:

                if file['configurations']['argOverride']:
                    argReplacement(file['configurations']['args'])
                else:
                    args = argsStorage

                if video != None or '':
                    #try:
                    # if True: # Just a testing line
                        # print(video)
                    dirNames = {"sessionDir": None if config['type'] == 'RunConfig' else dirName, "runDir": file['dir'] if config['type'] == 'SessionConfig' else dirName}
                    if args.iterations and args.iterations > 0:
                        iterations = args.iterations
                        iterationOptions = {"max_iterations": int(args.iterations), "current_iteration_index": 0}

                    else:
                        iterations = 1
                        iterationOptions = None

                    downscale = False
                    if args.downscale != 0:
                        downscale = True

                    if iterationOptions:
                        for i in range(iterations):
                            iterationOptions['current_iteration_index'] = i
                            main(video, dirNames, iterationOptions, downscale)
                        continue

                    if downscale:
                        resolutions = ['720', '648', '576', '360']
                        for resolution in resolutions:
                            main(video, dirNames, iterationOptions, resolution)
                        continue

                    main(video, dirNames, iterationOptions, downscale)

                    #except Exception as e:
                        #print("This is the error: "+ str(e))
                        # print(e.message)
                        #continue
                # print(";)")
                else:
                    print("Video was None or ''")
        timeEnd = time.time()
        totalTime = timeEnd - timeStart
        print("Total time: "+ str(totalTime))
    
    else:
        main(args.source, args.source.split(".")[0] if args.mode == "analysis" else args.file)