import os
import sys

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
from tools.visualize_objects import draw_rectangle, draw_text, draw_line


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
parser.add_argument("-t",
                    "--tracking",
                    help="Choose tracking model to utilize, DeepSort will be used as default")
parser.add_argument("-f",
                    "--file",
                    help="Define the name of the saved file",
                    type=str)
parser.add_argument("-i",
                    "--img_enh",
                    help="Specify how the image should be enhanced. By default no enhancements will be applied",
                    type=str) # TODO: Currently this does nothing
parser.add_argument("-s",
                    "--source",
                    help="Specify the source of the video to be analyzed",
                    type=str)
parser.add_argument("--mode",
                    help="Specify the mode of the application, by default live tracking will be used, other modes include analysis and training",
                    type=str) # TODO: Currently only analysis is recognized (evaluate if this is what analysis should do, currently it takes in a video file and outputs an analyzed video file with the same name)
parser.add_argument("--show",
                    help="Show a live feed of the tracking process, values are 1 = true, 0 = false, by default live feed will be disabled.",
                    type=int)
parser.add_argument("--statistics",
                    help="Gather statistics, default is 0",
                    default=0,
                    type=int)
parser.add_argument("--datamode",
                    help="Specify how the source should be handled. json expects a json file with session configurations, and mp4 expects a single mp4 file. Default is mp4",
                    default="mp4",
                    type=str)

args = parser.parse_args()

"""
Commands used in testing:
python liveTrackTwo.py -s Tunnel8.mp4 --mode analysis
python liveTrackTwo.py -s testing.txt --mode analysis --datamode multiple --show 1

"""

# TODO: This function does not return an error, but has not been tested to work for a real test just that the code passes through
# This function replaces the arguments defined in the command line with the values defined in the json file
def argReplacement(file):
    args.model = file['model']
    args.checkpoint = file['checkpoint']
    args.pretrained = file['pretrained']
    args.skip_frames = file['skip_frames']
    args.resize = file['resize']
    args.tracking = file['tracking']
    args.img_enhance = file['img_enh']
    args.mode = file['mode']
    args.show = file['show']
    args.statistics = file['statistics']

# Extracts the json data
def extractJSONFile(jsonFile):
    dataCollectionDir = r'.\\SessionConfigurations'
    dataCollectionFile = os.path.join(dataCollectionDir, jsonFile)
    
    with open(dataCollectionFile, 'r') as f:
        config = json.load(f)

    return config

def main(video, dirNames):
    datasets = []
    sourceDir = r'.\\data\\rawData'
    sourceFile = os.path.join(sourceDir, video)
    baseOutputDir = r'.\\data\\output'

    if args.datamode == "json":
        if dirNames['sessionDir']:
            baseOutputDir = os.path.join(baseOutputDir, dirNames['sessionDir'])
            baseOutputDir = os.path.join(baseOutputDir, dirNames['runDir'])
        else:
            baseOutputDir = os.path.join(baseOutputDir, dirNames['runDir'])

        if not os.path.exists(baseOutputDir):
            os.makedirs(baseOutputDir)

    datasets.append({"dataset": "selectedVideo", "video": sourceFile})

    model_filename = os.path.join(PATH_TO_THIS_FILE, 'tools/model_data/mars-small128.pb')

    paths = {
        "CHECKPOINT_PATH": "./training/models/ssd_mobnet/",
        "PIPELINE_CONFIG": "./training/models/ssd_mobnet/pipeline.config", 
        "LABELMAP": "./training/annotations/label_map.pbtxt",
        "DEEPSORT_MODEL": model_filename
    }

    image_enhancement_methods = ["gray_linear", "gray_nonlinear", "he", "retinex_ssr", "retinex_msr", "mask"]
    models = ["ssd_mobnet", "faster_rcnn", "yolov5", "yolov5_trained", "efficientdet"]
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}

    model_name = "yolov5"
    if args.model in models:
        paths["CHECKPOINT_PATH"] = "./training/models/" + args.model + "/"
        paths["PIPELINE_CONFIG"] = "./training/models/" + args.model + "/pipeline.config"
        model_name = args.model

    tracking_model_name = "DeepSort"
    if args.tracking:
        tracking_model_name = args.tracking

    ckpt_number = "3"
    if args.checkpoint is not None:
        ckpt_number = args.checkpoint

    filename = ""
    if args.file is not None:
        filename = args.file

    image_enhancement = "None"
    if args.img_enh is not None and args.img_enh in image_enhancement_methods:
        image_enhancement = args.img_enh

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

    model = Detection_Model(model_name, classes, paths, ckpt_number)
    tracker_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
    evaluater = Evaluate_Incidents(classes)
    pe = Evaluate_Performance("Video", datasets, classes, model, tracker_model)

    if args.file != None or args.mode == "analysis":
        cap = cv2.VideoCapture(sourceFile)
        if not cap.isOpened():
            print("Error: Could not open video")
        else:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        if args.mode == "analysis":
            output_video_path = os.path.join(baseOutputDir, video)
        else:
            output_video_path = os.path.join(baseOutputDir, (args.file + ".mp4"))
        frame_rate = 20
        frame_size = (int(width), int(height))
        print(frame_size)

        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    frame_number = 0
    while True:
        ret, frame, new_video, mask = pe.read(resize)
        frame_number += 1
        if frame_number % skip_frames != 0:
            continue

        if ret:
            frame = pe.image_enhancement(frame, image_enhancement, mask)
        else:
            print('Video has ended!')
            break

        if new_video:
            new_tracking_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
            pe.tracking_model = new_tracking_model
        
        pe.detect_and_track(frame)

        evaluater.purge(frame_number)

        for track in pe.get_tracks():
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            color, text, current_point, next_point = evaluater.evaluate(track, frame_number)

            # pe.performance(track, text)
            draw_rectangle(frame, track, color)
            draw_text(frame, track, text)
            if current_point and next_point:
                draw_line(frame, current_point, next_point)
        
        # pe.status()

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if args.file != None or args.mode == "analysis":
            out.write(result)

        if args.show == 1:
            cv2.imshow("Output Video", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    summary = pe.summary()
    print(summary)
    if filename != '':
        output_file = "./data/output/" + filename + ".txt"
        with open(output_file, "w") as file:
            output = f"Image enhancement: {image_enhancement}\n"
            output += f"Detection: {model_name}\n"
            output += f"Tracking: {tracking_model_name}\n"
            output += summary
            file.write(output)

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
            runConfig = [extractJSONFile(config)]
            dirName = runConfig['dir']

            i = 1
            while os.path.exists(os.path.join(r'.\\data\\output', dirName)):
                dirName = runConfig['dir'] + str(i)
                i += 1
            os.mkdir(os.path.join(r'.\\data\\output', dirName))
        
        print(runConfig)

        # Runs the main program for each video listed in the txt file
        # for video in videoList:
        for file in runConfig:
            for video in file['data']:

                if file['configurations']['argOverride']:
                    argReplacement(file['configurations']['args'])
                else:
                    args = argsStorage

                if video != None or '':
                    try:
                        # print(video)
                        dirNames = {"sessionDir": None if config['type'] == 'RunConfig' else config['dir'], "runDir": runConfig['dir']}

                        main(video, dirNames)
                    except:
                        continue # TODO: This is a little hacky for full implementation use specific exception. Exception can be found in the root directory
                # print(";)")
    
    else:
        main(args.source, args.source.split(".")[0] if args.mode == "analysis" else args.file)