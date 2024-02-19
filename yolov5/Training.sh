#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=pytorch_mnist
#SBATCH --output=mnist_test_01.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate training_env
# Run the training script
python -u train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt --nosave --cache