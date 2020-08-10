#!/usr/bin/env bash

# GTA-SfM
python test.py ~/Projects/data/gta_sfm \
       --pretrained-dps ~/Projects/DPSNet/checkpoints/gta_sfm-15epochs-b1/07-31-01_41/dpsnet_14_checkpoint.pth.tar \
       --mindepth 0.1 --maxdepth 30 --nlabel 64 --normalize_depths --stereo_dataset

# DeMoN data
# python test.py ./dataset/train --mindepth 0.5 --maxdepth 10 --nlabel 64
