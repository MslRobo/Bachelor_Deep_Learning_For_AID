#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=training_setup
#SBATCH --output=training_setup.out
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-py39
conda create -n Training_env -c pytorch pytorch torchvision numpy -y
conda activate Training_env
pip install -r requirements.txt