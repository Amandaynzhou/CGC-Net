#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --resume='/media/bialab/28A67CC4A67C93D2/visiting-student/experiments/gcnn/result/nuclei_soft-assign_l3x1_ar25_h20_o20_fca/model_best.pth.tar'\
 --visualization=True --skip_train=True --assign-ratio=0.25 --num-pool=1 --input_feature_dim 18 --feature 'ca'
