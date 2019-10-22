#!/usr/bin/env bash
python train.py --cv=1 --name='fuse' --n=8 --sample_ratio=0.5 --load_data_list --input_feature_dim=18 --feature 'ca' --assign-ratio=0.10 --batch-size=4\
 --num_workers=4 --norm_adj --method='soft-assign' --lr=0.001 --step_size=10 --gcn_name='SAGE' --sampling_method='fuse' --g='knn' --drop=0.2 --jk
