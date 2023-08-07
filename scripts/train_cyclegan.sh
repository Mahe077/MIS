#!/bin/bash
source /home/antpc/anaconda3/etc/profile.d/conda.sh

conda activate /hdd/2019CS077/TestCode/2019CS077_Test
export CUDA_VISIBLE_DEVICES=1

python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout --checkpoints_dir  ./checkpoints/

conda deactivate