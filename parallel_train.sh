#!/usr/bin/env bash
python train.py --cv=1 --name='fix_fuse_2' --n=8 --sample_ratio=0.5 --load_data_list --input_feature_dim=18 --feature 'ca' --assign-ratio=0.10 --batch-size=40 --sita=0\
 --num_workers=16 --norm_adj --beta=0 --alpha=0 --method='soft-assign-jk' --lr=0.001 --step_size=10 --gcn_name='SAGE' --sampling_method='fuse' --g='knn' --drop=0.2 --jk
