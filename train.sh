#!/usr/bin/env bash

python train.py ~/Projects/data/gta_sfm --log-output -b 1 --mindepth 0.1 --nlabel 64 --normalize_depths
