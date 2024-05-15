# Setup
## This file was borrowed from Aleksander Vedvik, was originally named README.md and has been edited to serve as an installation guide

ROOT or . is defined as the same folder where this readme file is located.


# Installation

## Prerequisites

- [Python 3.8](https://www.python.org/)
- [TensorFlow 2.0](https://www.tensorflow.org/)
- [Nvidia toolkit](https://developer.nvidia.com/cuda-toolkit)

Install dependencies:

    pip install requirements.txt

## YOLOv5:
Follow the installation steps provided [here](https://github.com/ultralytics/yolov5) by Ultralytics. Clone the repository into the "./training/yolov5/" folder. (Keep the `dataset.yaml` file. It is used when training the yolov5 model)
   

# Prepare the data

## Datasets

1. Download the datasets from [here](https://drive.google.com/drive/folders/1hNMBL2MNyz5dWZdi1BR1o1X-3yypdCkJ?usp=sharing).
2. Move the datasets into the folder "./data/" (Folder structure is given in the README file located in the folder "./data")

## YOLOv5:

Prepare the training and validation dataset for YOLOv5 by running this command from ROOT:

    cd ./training/scripts
    python generate_yolo.py


# Train

## YOLOv5:

    python ./yolov5/train.py --img 640 --batch 4 --epoch 30 --data ./dataset.yaml --cfg ./yolov5/models/yolov5x.yaml --weights ./yolov5x.pt --name ./yolov5x_trained

## Trained models:

Checkpoints from the trained model used in the thesis can be downloaded from [here](https://drive.google.com/drive/folders/1hNMBL2MNyz5dWZdi1BR1o1X-3yypdCkJ?usp=sharing). The checkpoints are stored in the "Checkpoints" folder on the Google Drive folder. These checkpoints should have a similar folder structure to this Github. I.e. the checkpoints must be placed in the respective folders located under "./training/".
