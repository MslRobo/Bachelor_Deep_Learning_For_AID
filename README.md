# Bachelor
I will provide a quick guide below on how to get started with this repository. README files are also provided in all relevant folders, which give a short description and explanation of each file in the respective folders.

ROOT or . is defined as the same folder where this readme file is located.


# Installation

## Prerequisites

- [Python 3.8](https://www.python.org/)
- [TensorFlow 2.0](https://www.tensorflow.org/)
- [Nvidia toolkit](https://developer.nvidia.com/cuda-toolkit)

Install dependencies:

    pip install requirements.txt

## TensorFlowAPI:
1. Download protoc and add to PATH:
    - Download [Protoc](https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip).

    - Add protoc to Windows PATH: "*_PATH TO DIRECTORY_*\protoc\protoc-3.15.6-win64\bin"

2. Clone [this repository](https://github.com/tensorflow/models) into the "./training/tensorflowapi/" folder.

3. Install dependencies:
   
        cd ./training/tensorflowapi/research/slim && pip install -e .

4. Download pre-trained models [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and put them in the "./training/pre-trained-models" folder. The models used in this thesis are: ssd_mobnet, efficientdet and faster_rcnn.


## YOLOv5:
Follow the installation steps provided [here](https://github.com/ultralytics/yolov5) by Ultralytics. Clone the repository into the "./training/yolov5/" folder. (Keep the `dataset.yaml` file. It is used when training the yolov5 model)
   

# Prepare the data

## Datasets

1. Download the datasets from [here](https://drive.google.com/drive/folders/1hNMBL2MNyz5dWZdi1BR1o1X-3yypdCkJ?usp=sharing).
2. Move the datasets into the folder "./data/" (Folder structure is given in the README file located in the folder "./data")

## TensorFlowAPI:

Create .tfrecords by running this command from ROOT:

    cd ./training/scripts
    python generate_records.py


## YOLOv5:

Prepare the training and validation dataset for YOLOv5 by running this command from ROOT:

    cd ./training/scripts
    python generate_yolo.py


# Train

## TensforFlowAPI:
Run this command from ROOT to train a model from TensorFlowAPI:

    cd training
    python tensorflowapi/research/object_detection/model_main_tf2.py --model_dir=models/*MODEL_NAME* --pipeline_config_path=models/*MODEL_NAME*/pipeline.config --num_train_steps=*NUMBER_OF_STEPS*

The different models used are: 
- Model name: **ssd_mobnet**, number of steps: **10 000**
- Model name: **efficientdet**, number of steps: **15 000**
- Model name: **faster_rcnn**, number of steps: **20 000**

## YOLOv5:

    python ./yolov5/train.py --img 640 --batch 4 --epoch 30 --data ./dataset.yaml --cfg ./yolov5/models/yolov5x.yaml --weights ./yolov5x.pt --name ./yolov5x_trained

## Trained models:

Checkpoints from the trained model used in the thesis can be downloaded from [here](https://drive.google.com/drive/folders/1hNMBL2MNyz5dWZdi1BR1o1X-3yypdCkJ?usp=sharing). The checkpoints are stored in the "Checkpoints" folder on the Google Drive folder. These checkpoints should have a similar folder structure to this Github. I.e. the checkpoints must be placed in the respective folders located under "./training/".


# Eval

## TensforFlowAPI:
Run this command from ROOT to train a model from TensorFlowAPI:

    cd training
    python tensorflowapi/research/object_detection/model_main_tf2.py --model_dir=models/*MODEL_NAME* --pipeline_config_path=models/*MODEL_NAME*/pipeline.config --checkpoint_dir=models/*MODEL_NAME*

The different models used are: 
- Model name: **ssd_mobnet**
- Model name: **efficientdet**
- Model name: **faster_rcnn**


# Tensorboard

You can use Tensorboard to get info regarding the fine-tuned models. Navigate to the model folders and run:

    tensorboard --logdir=.


# Test the models on the test dataset:

Run this command from ROOT to test the models:

    python run.py -m *model* -c *checkpoint* -r *number* -t *tracking_model* -i *image_enhancement_method* -f *filename* -s *number* -p *bool*

**Description of flags**:
- -m: Specify which object detection model to use. 
  - **Options**: yolov5, yolov5_trained, ssd_mobnet, efficientdet, faster_rcnn
  - **Default**: yolov5
- -c: Specify the checkpoint to use. Only relevant for the models from TensorFlowAPI.
- -r: Resize the input frame. Specify a number that should be used as the scale factor.
- -t: Specify which tracking model to use.
  - **Options**: DeepSort, simple
  - **Default**: DeepSort
- -i: Specify which image enhancement method to use:
  - **Options**: none, gray_linear, gray_nonlinear, he, retinex_ssr, retinex_msr, mask
  - **Default**: none
- -f: Specify a filename to save the results to a file.
- -s: Specify how many frames should be skipped.
- -p: If given, a pre-trained model will be used. The model name must correspond to the folder name in the "pre-trained-models" directory. 
- *It is possible to simply run `python run.py` without any flags. Then pre-trained YOLOv5x, deep SORT, and no image enhancement will be applied. No files will be saved.*


# References:

- Tensorflow models:
  - [TensorFlow Model Garden](https://github.com/tensorflow/models)

- Reused source code:
  - [TFODCourse](https://github.com/nicknochnack/TFODCourse), Nicholas Renotte
  - [Retinex Image Enhancement](https://github.com/aravindskrishnan/Retinex-Image-Enhancement), Aravind S Krishnan
  - [The AI Guy Code](https://github.com/theAIGuysCode/yolov4-deepsort), The AI Guy
  - [Deep SORT](https://github.com/nwojke/deep_sort), Nicolai Wojke

- Datasets:
  - [Master thesis](https://github.com/BerntA/CVEET), Bernt Andreas Eide
  - [YouTube video used as test dataset](https://www.youtube.com/watch?v=IOxxEJpXZGU&ab_channel=RedDFilm)

References are also cited in the files where code has been used.