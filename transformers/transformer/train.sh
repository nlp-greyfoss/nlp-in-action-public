#!/bin/sh
timestamp=$(date +"%Y%m%d%H%M")
nohup python train_multi_gpus.py > train_$timestamp.log 2>&1 &
tail -f train_$timestamp.log
