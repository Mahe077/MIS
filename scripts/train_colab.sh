#!/bin/bash

# Optimized training script for Colab

# Note: You will need to upload your dataset to Colab and update the --dataroot path.
# You may also want to save checkpoints to your Google Drive; update --checkpoints_dir accordingly.

python ../train.py \
    --dataroot ./datasets/maps \
    --name maps_cyclegan_colab \
    --model cycle_gan \
    --pool_size 10 \
    --no_dropout \
    --checkpoints_dir ./checkpoints \
    --loadSize 143 \
    --fineSize 128 \
    --ngf 32 \
    --ndf 32
