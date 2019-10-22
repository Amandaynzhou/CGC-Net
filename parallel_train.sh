#!/usr/bin/env bash
python train.py --cv=1 --name='fix_fuse_2' --n=8 --sample_ratio=0.5 --load_data_list --input_feature_dim=18 --feature 'ca' --assign-ratio=0.20 --batch-size=4 --sita=0\
 --num_workers=4 --norm_adj --method='soft-assign' --lr=0.001 --step_size=10 --gcn_name='SAGE' --sampling_method='fuse' --g='knn' --drop=0.2 --jk\
  --resume='nuclei_soft-assign_l3x1_ar20_h20_o20_fca_%0.5_a0.0_b0.0fix_fuse_list_adj0.4_sr0.5_d0.2_jkknn' --skip_train --visualization
