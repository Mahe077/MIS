#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate /hdd/2019CS077/TestCode/2019CS077_Test

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1

# Run your Python script
python ../train.py --dataroot /hdd/2019CS077/TestCode/v0/datasets/maps --name maps_cyclegan_rgb_v2 --model cycle_gan --pool_size 50 --no_dropout --checkpoints_dir  /hdd/2019CS077/TestCode/v0/checkpoints/ &>  train_rgb_v2.txt

# Deactivate the conda environment
conda deactivate