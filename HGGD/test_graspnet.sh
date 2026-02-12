#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python test_graspnet.py \
--center-num 48 \
--anchor-num 7 \
--anchor-k 6 \
--anchor-w 50 \
--anchor-z 20 \
--grid-size 8 \
--scene-l 174 \
--scene-r 190 \
--all-points-num 25600 \
--group-num 512 \
--local-k 10 \
--ratio 8 \
--input-h 360 \
--input-w 640 \
--local-thres 0.01 \
--heatmap-thres 0.01 \
--num-workers 1 \
--dataset-path './data/6dto2drefine_realsense' \
--checkpoint './realsense_checkpoint' \
--scene-path './data/graspnet' \
--dump-dir 'pred_grasps' \
--description 'realsense_seen'
