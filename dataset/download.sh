#!/usr/bin/env bash

## increase the max open file limit allowed by OS
# ulimit -n 4096

# Dataset URIs
# s3://argoverse/datasets/av2/sensor/
# s3://argoverse/datasets/av2/lidar/
# s3://argoverse/datasets/av2/motion-forecasting/
# s3://argoverse/datasets/av2/tbv/

export DATASET_NAME="motion-forecasting"  # sensor, lidar, motion_forecasting or tbv.
export TARGET_DIR="./dataset/Argoverse2/motion_forecasting/"  # Target directory on your machine.

# modify the number of workers according to your CPU
s5cmd --no-sign-request --numworkers 30 cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR
