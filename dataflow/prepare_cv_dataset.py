
import os.path as osp
import os
import torch
from multiprocessing import Pool
from torch_geometric.data.dataset import to_list
from torch_geometric.data import Dataset
from torch_geometric.data import Data, DataListLoader,DataLoader
from torch_cluster import neighbor_sampler
from torch_geometric.nn import radius_graph
from torch_geometric.nn import GCNConv
import copy
import random
import threading
import numpy as np
import cv2
import glob
import scipy.sparse as sp
from tqdm import tqdm
from common.transform import get_composed_augmentations
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.transforms import RadiusGraph, NormalizeFeatures,normalize_scale
from sklearn import preprocessing
from common.utils import mkdirs,FarthestSampler,filter_sampled_indice
# from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import sparse_to_dense, dense_to_sparse
from setting import DataSetting,ICIARSetting,CrossValidSetting
from torchvision import transforms
from dataflow.graph_sampler import radius_graph_with_least_k,prob_distance_graph,random_sample_graph2
from torch.utils.data.distributed import DistributedSampler
from common.utils import plt_graphs
import pdb
from torch_geometric.datasets import MNISTSuperpixels
FEATURE_NAMES = ['max_gray','min_gray','mean_im_out','diff','mean_h','mean_s','mean_v','mean_l','mean_a','mean_b','var_im'
                ,'skew_im','mean_ent','glcm_dissimilarity','glcm_homogeneity','glcm_energy','glcm_ASM','hull_area',
                'eccentricity','equiv_diameter','extent','area','majoraxis_length','minoraxis_length','perimeter',
                'solidity','orientation','radius','aspect_ratio']

# _appearance_subset_idx = [2,3,10,11,12,13,14,15,16,18,21,22,23,24,25,26 ]
#_label_subset_idx = [30,31,32,33,34,35]
#_coordinate_idx = [36,37]

_CROSS_VAL = {1:{'train':['fold_1', 'fold_2'], 'valid': ['fold_3']},
              2:{'train':['fold_1', 'fold_3'], 'valid': ['fold_2']},
              3:{'train':['fold_2', 'fold_3'], 'valid': ['fold_1']},

}

_MEAN_CIA = {1:[ 1.44855589e+02,  1.50849152e+01,  4.16993829e+02, -9.89115031e-02,
         4.29073361e+00,  7.03308534e+00,  1.50311764e-01,  1.20372119e-01,
         1.99874447e-02,  7.24825770e-01,  1.28062193e+02,  1.71914904e+01,
         9.00313323e+00,  4.29522533e+01,  8.76540101e-01,  8.06801284e+01, 3584,3584],
         2:[ 1.45949547e+02,  1.53704952e+01,  4.39127922e+02, -1.10080479e-01,
         4.30617772e+00,  7.27624697e+00,  1.45825849e-01,  1.21214980e-01,
         2.03645262e-02,  7.28225987e-01,  1.27914898e+02,  1.72524907e+01,
         8.96012595e+00,  4.30067152e+01,  8.76016742e-01,  8.09466370e+01,3584,3584],
         3:[ 1.45649518e+02,  1.52438912e+01,  4.30302592e+02, -1.07054163e-01,
         4.29877990e+00,  7.13800092e+00,  1.47971754e-01,  1.20517868e-01,
         2.00830612e-02,  7.24701226e-01,  1.26430193e+02,  1.71710396e+01,
         8.94070628e+00,  4.27421136e+01,  8.74665450e-01,  8.02611304e+01,3584,3584]}

_STD_CIA = {1:[3.83891570e+01, 1.23159786e+01, 3.74384781e+02, 5.05079918e-01,
        1.91811771e-01, 2.95460595e+00, 7.31040425e-02, 7.41484835e-02,
        2.84762625e-02, 2.47544275e-01, 1.51846534e+02, 5.96200235e+01,
        6.00087195e+00, 2.85961395e+01, 1.95532620e-01, 5.49411936e+01,3584,3584],
            2:[3.86514982e+01, 1.25207234e+01, 3.87362858e+02, 5.02515226e-01,
        1.89045551e-01, 3.05856764e+00, 7.22404102e-02, 7.53090608e-02,
        2.90460236e-02, 2.46734916e-01, 1.53743958e+02, 6.34661492e+01,
        6.02575043e+00, 2.88403590e+01, 1.94214810e-01, 5.49984596e+01,3584,3584],
            3:[3.72861596e+01, 1.23840868e+01, 3.87834784e+02, 5.02444847e-01,
        1.86722327e-01, 2.99248449e+00, 7.20327363e-02, 7.45553798e-02,
        2.87285660e-02, 2.49195190e-01, 1.50986869e+02, 6.56370060e+01,
        6.00008814e+00, 2.86376250e+01, 1.97764021e-01, 5.54134874e+01,3584,3584]}



#
# class PreprocessData(object):
#     def __init__(self):
#         super(PreprocessData, self).__init__()
#         self.setting = CrossValidSetting()
#         self.folds = ['fold_1', 'fold_2', 'fold_3']
#         self.original_files = []
#         self.sampling_ratio = 0.5
#         self.sample_method = 'fuse'
#         self.mask = 'cia'
#         self.graph_sampler = 'knn'
#         self.max_edge_distance =100
#         self.max_neighbours = 8
#         self.sampler = FarthestSampler()
#         for fold in self.folds:
#             for f in glob.iglob(self.setting.root + '/proto/feature/shaban/' + fold +'/**/*', recursive=True):
#                 if '.npy' in f:
#                     self.original_files.append(f.strip('.npy'))
#         self.processed_dir =os.path.join( self.setting.root , 'proto', 'fix_%s_%s_%s'%(self.sample_method,self.mask,self.graph_sampler))
#         for f in self.folds:
#             mkdirs(os.path.join(self.processed_dir, f))
#
#         self.process(50)

def _read_one_raw_graph( raw_file_path):
    # import pdb;pdb.set_trace()
    nodes_features = np.load( raw_file_path + '.npy')
    coordinates = np.load(raw_file_path.replace('feature', 'coordinate') + '.npy')
    nodes_features = np.concatenate((nodes_features, coordinates), axis= -1)
    coordinates = torch.from_numpy(coordinates).to(torch.float)
    nodes_features = torch.from_numpy(nodes_features ).to(torch.float)
    if '1_normal' in raw_file_path:
        label = 0
    elif '2_low_grade' in raw_file_path:
        label = 1
    else:
        label = 2
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x = nodes_features, pos = coordinates, y = y)
    return data




def gen(raw_path):
    epoch = 50
    graph_sampler = 'knn'
    mask = 'cia'
    sample_method= 'fuse'
    setting = CrossValidSetting()
    processed_dir = os.path.join(setting.root, 'proto',
                                 'fix_%s_%s_%s' % (sample_method, mask, graph_sampler))

     # Read data from `raw_path`
    data =_read_one_raw_graph(raw_path)
     # sample epoch time
    num_nodes = data.x.shape[0]
    num_sample = num_nodes
    distance_path = os.path.join(setting.root, 'proto', 'distance', 'shaban',
                                  raw_path.split('/')[-3], raw_path.split('/')[-1] + '.pt')
    if mask == 'hvnet':
        distance_path = distance_path.replace('distance', 'distance_s')
    distance = np.load(distance_path.replace('.pt', '.npy'))
    for i in range(epoch):
       subdata = copy.deepcopy(data)
       choice, num_subsample = _sampling(num_sample,0.5,distance)
       # sample_num_node = int(self.sampling_ratio * num_nodes)
       # choice = np.random.choice(num_nodes,sample_num_node , replace=False)
       for key, item in subdata:
           if torch.is_tensor(item) and item.size(0) == num_nodes:
               subdata[key] = item[choice]
       # generate the graph
       if graph_sampler == 'knn':
           edge_index = radius_graph(subdata.pos, 100, None, True, 8)
       else:
           edge_index = random_sample_graph2(choice, distance, 100, True,
                                     n_sample=8,sparse=True)
       subdata.edge_index=edge_index
       torch.save(data, osp.join(processed_dir,str(i),
                                 raw_path.split('/')[-3],
                                  raw_path.split('/')[-1].split('.')[0] + '.pt'))

def _sampling( num_sample, ratio, distance = None):


    num_subsample = int(num_sample * ratio)



    if sample_method == 'farthest':


        indice = sampler(distance, num_subsample)

    elif sample_method == 'fuse':
        # 70% farthest, 30% random
        far_num =int( 0.7 * num_subsample)
        rand_num = num_subsample - far_num

        far_indice = sampler(distance, far_num)
        remain_item = filter_sampled_indice(far_indice, num_sample)
        rand_indice = np.asarray(random.sample(remain_item, rand_num))

        indice = np.concatenate((far_indice, rand_indice),0)

    else:
        # random
        indice = np.random.choice(num_subsample, num_sample, replace=False)

    return  indice, num_subsample

if __name__ == '__main__':
    setting = CrossValidSetting()
    folds = ['fold_1', 'fold_2', 'fold_3']
    original_files = []
    sampling_ratio = 0.5
    sample_method = 'fuse'
    mask = 'cia'
    graph_sampler = 'knn'
    max_edge_distance = 100
    max_neighbours = 8
    epoch = 50
    sampler = FarthestSampler()
    for fold in folds:
        for f in glob.iglob(setting.root + '/proto/feature/shaban/' + fold + '/**/*', recursive=True):
            if '.npy' in f:
                original_files.append(f.strip('.npy'))
    processed_dir = os.path.join(setting.root, 'proto',
                                      'fix_%s_%s_%s' % (sample_method, mask, graph_sampler))
    for f in folds:

        for j in range(epoch):
            mkdirs(osp.join(processed_dir, '%d' % epoch, f))

    p = Pool(32)
    arr = p.map(gen,original_files )
    p.close()
    p.join()
