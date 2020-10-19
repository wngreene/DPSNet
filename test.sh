#!/usr/bin/env bash

# # GTA-SfM
# python stereo_dataset_test.py ~/Projects/data/gta_sfm \
#        --pretrained-dps ~/Projects/DPSNet/checkpoints/gta_sfm-15epochs-b1/07-31-01_41/dpsnet_14_checkpoint.pth.tar \
#        --mindepth 0.1 --maxdepth 30 --nlabel 64 --normalize_depths --stereo_dataset --output-print --roll_right_image_180

# DeMoN data
python test.py ~/Projects/data/demon/test/ --sequence-length 2 --output-print --pretrained-dps ./pretrained/dpsnet.pth.tar
