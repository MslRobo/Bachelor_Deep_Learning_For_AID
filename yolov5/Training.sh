#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=training
#SBATCH --output=traing_01.out
 
# Activate environment
uenv verbose cuda-11.4.0 cudnn-11.4-8.2.4
uenv miniconda3-py39
conda activate Training_env
# Run the training script
python -u train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt --nosave --cache