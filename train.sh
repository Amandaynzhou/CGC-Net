#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --input_feature_dim=2 --feature 'c' --assign-ratio=0.1 --batch-size=2 --num_workers=2