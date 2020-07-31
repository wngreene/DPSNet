#!/usr/bin/env bash

# GTA-SfM
python train.py ~/Projects/data/gta_sfm --log-output -b 1 --mindepth 0.1 --nlabel 64 --normalize_depths --stereo_dataset --epochs 15

# DeMoN data
# python train.py ./dataset/train --log-output -b 1 --mindepth 0.5 --nlabel 64
