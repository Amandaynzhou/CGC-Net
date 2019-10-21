import os.path as osp
import os
import torch
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
from dataflow.graph_sampler import radius_graph_with_least_k,prob_distance_graph,random_sample_graph
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


def prepare_train_val_loader(args):
    if args.task == 'colon':
        setting = DataSetting()
    elif args.task == 'ICIAR':
        setting = ICIARSetting()
    else:
        setting = CrossValidSetting()
    # pdb.set_trace()
    # if  len(os.listdir('/research/dept6/ynzhou/gcnn/data/proto/cross_val/fold_1'))==0:
    # PreprocessData()

    if args.load_data_list and (not args.full_test_graph):
        # if torch.cuda.device_count() > 1:
        #     sampler_type = DistributedSampler(
        #         NucleiDatasetBatchOutput(root=setting.root, feature_type=args.feature_type,
        #                                  split='train', sampling_time=setting.sample_time,
        #                                  sampling_ratio=setting.sample_ratio,
        #                                  normalize=args.normalize, dynamic_graph=args.dynamic_graph,
        #                                  sampling_method=args.sampling_method))
        # else:aggregator
        #     sampler_type = None
        sampler_type = None
        train_dataset_loader = DataListLoader( NucleiDatasetBatchOutput( root = setting.root, feature_type = args.feature_type,
                                                          split = 'train' if args.task=='colon' else 'train_aug', sampling_time=setting.sample_time,
                                                          sampling_ratio=args.sample_ratio,
                                                          normalize= args.normalize,dynamic_graph=args.dynamic_graph,
                                                          sampling_method =args.sampling_method,
                                                                         sampling_by_number=args.sampling_by_number,
                                                                         datasetting = setting, mask = args.mask,
                                                                         neighbour = args.neighbour, add_noise = args.add_noise,
                                                                         graph_sampler = args.graph_sampler,
                                                                         crossval=args.cross_val),
                                                          sampler=sampler_type,
                                                          batch_size=args.batch_size,
                                                          shuffle=True if sampler_type is None else False,
                                                          num_workers=args.num_workers,

                                                         )

        validset = NucleiDatasetBatchOutput(root=setting.root,
                                    feature_type=args.feature_type,
                                    split='valid',
                                    sampling_time=setting.sample_time if not args.valid_full else 1,
                                    sampling_ratio=args.sample_ratio if not args.valid_full else 1,
                                    normalize=args.normalize,
                                    dynamic_graph=args.dynamic_graph,
                                    sampling_method=args.sampling_method,
                                    sampling_by_number=args.sampling_by_number,
                                    datasetting=setting,
                                    mask=args.mask,
                                    neighbour=args.neighbour,
                                            graph_sampler=args.graph_sampler,crossval = args.cross_val)

        val_dataset_loader =  DataListLoader(
            validset,
            batch_size= args.batch_size if not args.valid_full else 1,  # setting.batch_size,
            shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

        # val_dataset_loader = DataListLoader(
        #     NucleiDatasetBatchOutput(root=setting.root, feature_type = args.feature_type, split = 'valid', sampling_time=setting.sample_time,
        #                   sampling_ratio=args.sample_ratio, normalize= args.normalize,dynamic_graph=args.dynamic_graph,
        #                   sampling_method=args.sampling_method,
        #                              sampling_by_number=args.sampling_by_number,datasetting = setting,
        #                              mask = args.mask,
        #                              neighbour=args.neighbour,least_n = 0),
        #                   batch_size=args.batch_size,#setting.batch_size,
        #                   shuffle=False,
        #                   num_workers=args.num_workers, pin_memory=True
        #                   )

        if args.task == 'colon':
            test_dataset_loader = DataListLoader(
                NucleiDatasetBatchOutput(root=setting.root, feature_type = args.feature_type, split = 'valid', sampling_time=setting.sample_time,
                              sampling_ratio=setting.sample_ratio, normalize= args.normalize,dynamic_graph=args.dynamic_graph,
                              sampling_method=args.sampling_method,
                                         sampling_by_number=args.sampling_by_number,datasetting = setting
                                         , mask=args.mask,neighbour = args.neighbour ,graph_sampler = args.graph_sampler,crossval = args.cross_val),
                              batch_size=args.batch_size if not args.full_test_graph else 1,#setting.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers, pin_memory=True
                              )
        else:
            test_dataset_loader = None
    elif args.load_data_sparse:
        # if  torch.cuda.device_count() > 1:
        #     sampler_type = DistributedSampler( NucleiDatasetBatchOutput( root = setting.root, feature_type = args.feature_type,
        #                                                              split='train', sampling_time=setting.sample_time,
        #                                                   sampling_ratio=setting.sample_ratio,
        #                                                   normalize= args.normalize,
        #                                                   dynamic_graph=args.dynamic_graph,
        #                                                   sampling_method=args.sampling_method))
        # else:
        sampler_type = None
        train_dataset_loader = DataLoader( NucleiDatasetBatchOutput( root = setting.root, feature_type = args.feature_type,
                                                                     split='train'if args.task=='colon' else 'train_aug', sampling_time=setting.sample_time,
                                                          sampling_ratio=args.sample_ratio,
                                                          normalize= args.normalize,
                                                          dynamic_graph=args.dynamic_graph,
                                                          sampling_method=args.sampling_method,
                                                                     sampling_by_number=args.sampling_by_number,
                                                                     datasetting=setting,
                                                                     mask = args.mask,
                                                                     neighbour=args.neighbour,add_noise = args.add_noise,
                                                                     graph_sampler = args.graph_sampler),
                                                          batch_size=args.batch_size,
                                                          sampler=sampler_type,
                                           shuffle=True if sampler_type is None else False,
                                                          num_workers=args.num_workers,
                                           )

        val_dataset_loader = DataLoader(
            NucleiDatasetBatchOutput(root=setting.root, feature_type = args.feature_type, split = 'valid',
                                     sampling_time=setting.sample_time,
                                     sampling_ratio=args.sample_ratio,
                                     normalize= args.normalize,
                                     dynamic_graph=args.dynamic_graph,
                                     sampling_method = args.sampling_method,
                                     sampling_by_number=args.sampling_by_number,
                                     datasetting=setting
                                     , mask=args.mask,
                                     neighbour=args.neighbour,graph_sampler = args.graph_sampler),
                                     batch_size=args.batch_size,#setting.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True
                )
        if args.task == 'colon':
            test_dataset_loader = DataLoader(
                NucleiDatasetBatchOutput(root=setting.root,
                                         feature_type = args.feature_type,
                                         split = 'test',
                                         sampling_time=setting.sample_time,
                              sampling_ratio=args.sample_ratio,
                                         normalize= args.normalize,
                                         dynamic_graph=args.dynamic_graph,
                                         sampling_method = args.sampling_method,
                                         sampling_by_number=args.sampling_by_number,
                                         datasetting=setting,
                                         mask = args.mask,
                                         neighbour=args.neighbour,graph_sampler = args.graph_sampler),
                              batch_size=args.batch_size,#setting.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers, pin_memory=True
                    )
        else:
            test_dataset_loader = None
    else:
        # if torch.cuda.device_count() > 1:
        #     sampler_type = DistributedSampler(NucleiDataset( root = setting.root, feature_type = args.feature_type, split = 'train', sampling_time=setting.sample_time,
        #                   sampling_ratio=setting.sample_ratio, normalize= args.normalize,dynamic_graph=args.dynamic_graph,sampling_method = args.sampling_method
        #                    ))
        # else:
        sampler_type = None
        train_dataset_loader =torch.utils.data.DataLoader(
            NucleiDataset( root = setting.root, feature_type = args.feature_type,
                           split = 'train' if args.task=='colon' else 'train_aug',
                           sampling_time=setting.sample_time,
                          sampling_ratio=args.sample_ratio,
                           normalize= args.normalize,
                           dynamic_graph=args.dynamic_graph,
                           sampling_method = args.sampling_method,
                           sampling_by_number= args.sampling_by_number,
                           datasetting=setting,
                           mask = args.mask,
                           neighbour=args.neighbour, add_noise = args.add_noise, graph_sampler = args.graph_sampler,crossval = args.cross_val
                           ),
            batch_size=args.batch_size,
            shuffle=True if sampler_type is None else False,
            sampler=sampler_type,
            num_workers=args.num_workers,
            )

        valset = NucleiDatasetTest(root=setting.root,
                                    feature_type=args.feature_type,
                                    split='valid',
                                    sampling_time=setting.sample_time if not args.valid_full else  1,
                                    sampling_ratio=args.sample_ratio if not args.valid_full else 1,
                                    normalize=args.normalize,
                                    dynamic_graph=args.dynamic_graph,
                                    sampling_method=args.sampling_method,
                                    sampling_by_number=args.sampling_by_number,
                                    datasetting=setting,
                                    mask=args.mask,
                                    neighbour=args.neighbour, graph_sampler = args.graph_sampler,crossval = args.cross_val)

        val_dataset_loader =  torch.utils.data.DataLoader(
                valset,
                batch_size=  args.batch_size if not args.sample_ratio else 1,#setting.batch_size,
                shuffle=False,
                num_workers=args.num_workers, pin_memory=True
                )
        #
        # val_dataset_loader = torch.utils.data.DataLoader(
        #     NucleiDataset(root=setting.root, feature_type = args.feature_type,
        #                   split = 'valid',
        #                   sampling_time=setting.sample_time,
        #                   sampling_ratio=args.sample_ratio,
        #                   normalize= args.normalize,
        #                   dynamic_graph=args.dynamic_graph,
        #                   sampling_method =args.sampling_method,
        #                   sampling_by_number=args.sampling_by_number,
        #                   datasetting=setting,
        #                   mask = args.mask,
        #                   neighbour=args.neighbour,least_n = args.least_n
        #                   ),
        #     batch_size=args.batch_size,#setting.batch_size,
        #     shuffle=False,
        #     num_workers=args.num_workers, pin_memory=True
        #     )
        if args.task == 'colon':
            if args.full_test_graph:
                testset = NucleiDatasetTest(root=setting.root,
                              feature_type = args.feature_type,
                              split = 'valid',
                              sampling_time=1,
                              sampling_ratio=1,
                              normalize= args.normalize,
                              dynamic_graph=args.dynamic_graph,
                              sampling_method =args.sampling_method,
                                            sampling_by_number=args.sampling_by_number,
                                            datasetting=setting,
                                            mask = args.mask,
                                            neighbour=args.neighbour,graph_sampler = args.graph_sampler,crossval = args.cross_val)
            else:
                testset = NucleiDataset(root=setting.root,
                              feature_type = args.feature_type,
                              split = 'valid',
                              sampling_time=setting.sample_time,
                              sampling_ratio=args.sample_ratio,
                              normalize= args.normalize,
                              dynamic_graph=args.dynamic_graph,
                              sampling_method =args.sampling_method,
                                        sampling_by_number=args.sampling_by_number,
                                        datasetting=setting,
                                        mask = args.mask,
                                        neighbour=args.neighbour,
                                        graph_sampler=args.graph_sampler,crossval = args.cross_val
                                        )


            test_dataset_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=args.batch_size if not args.full_test_graph else 1,#setting.batch_size,
                shuffle=False,
                num_workers=args.num_workers, pin_memory=False
                )
        else:
            test_dataset_loader = None

    return train_dataset_loader, val_dataset_loader, test_dataset_loader


class PreprocessData(object):
    def __init__(self):
        super(PreprocessData, self).__init__()
        self.setting = CrossValidSetting()
        self.folds = ['fold_1', 'fold_2', 'fold_3']
        self.original_files = []
        self.sampling_ratio = 0.5
        self.sample_method = 'fuse'
        self.mask = 'cia'
        self.graph_sampler = 'knn'
        self.max_edge_distance =100
        self.max_neighbours = 8
        self.sampler = FarthestSampler()
        for fold in self.folds:
            for f in glob.iglob(self.setting.root + '/proto/feature/shaban/' + fold +'/**/*', recursive=True):
                if '.npy' in f:
                    self.original_files.append(f.strip('.npy'))
        self.processed_dir =os.path.join( self.setting.root , 'proto', 'fix_%s_%s_%s'%(self.sample_method,self.mask,self.graph_sampler))
        for f in self.folds:
            mkdirs(os.path.join(self.processed_dir, f))

        self.process(50)

    def _read_one_raw_graph(self, raw_file_path):
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

    def process(self,epoch):
        i = 0
        print('pre-process the graph dataset...')
        for j in range(epoch):
            mkdirs( osp.join(self.processed_dir , '_%d'%epoch))


    def gen(self):
        epoch = 50
        for raw_path in tqdm(self.original_files):
             # Read data from `raw_path`
             data = self._read_one_raw_graph(raw_path)
             # sample epoch time
             num_nodes = data.x.shape[0]
             num_sample = num_nodes
             distance_path = os.path.join(self.setting.root, 'proto', 'distance', 'shaban',
                                          raw_path.split('/')[-3], raw_path.split('/')[-1] + '.pt')
             if self.mask == 'hvnet':
                 distance_path = distance_path.replace('distance', 'distance_s')
             distance = np.load(distance_path.replace('.pt', '.npy'))
             for i in range(epoch):
                subdata = copy.deepcopy(data)
                choice, num_subsample = self._sampling(num_sample,self.sampling_ratio,distance)
                # sample_num_node = int(self.sampling_ratio * num_nodes)
                # choice = np.random.choice(num_nodes,sample_num_node , replace=False)
                for key, item in subdata:
                    if torch.is_tensor(item) and item.size(0) == num_nodes:
                        subdata[key] = item[choice]
                # generate the graph
                if self.graph_sampler == 'knn':
                    edge_index = radius_graph(subdata.pos, self.max_edge_distance, None, True, self.max_neighbours)

                # elif self.graph_sampler == 'nearest':
                #     edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True,
                #                                            self.max_neighbours)
                #     adj = sparse_to_dense(edge_index)

                else:
                    edge_index = random_sample_graph(choice, distance, self.max_edge_distance, True,
                                              n_sample=self.max_neighbours,sparse=True)
                subdata.edge_index=edge_index
                torch.save(data, osp.join(self.processed_dir,
                                          raw_path.split('/')[-3],
                                           raw_path.split('/')[-1].split('.')[0] + '.pt'))

    def _sampling(self, num_sample, ratio, distance = None):


        num_subsample = int(num_sample * ratio)



        if self.sample_method == 'farthest':


            indice = self.sampler(distance, num_subsample)

        elif self.sample_method == 'fuse':
            # 70% farthest, 30% random
            far_num =int( 0.7 * num_subsample)
            rand_num = num_subsample - far_num

            far_indice = self.sampler(distance, far_num)
            remain_item = filter_sampled_indice(far_indice, num_sample)
            rand_indice = np.asarray(random.sample(remain_item, rand_num))

            indice = np.concatenate((far_indice, rand_indice),0)

        else:
            # random
            indice = np.random.choice(num_subsample, num_sample, replace=False)

        return  indice, num_subsample

class NucleiDataset(Dataset):

    def __init__(self, root,  feature_type, transform=None, pre_transform=None, split = 'train',
                 sampling_time = 10, sampling_ratio = 0.5, normalize = False, dynamic_graph = False, sampling_method = 'farthest',
                 sampling_by_number = True, datasetting = None, mask = 'CIA',neighbour = 8, add_noise = False, graph_sampler = 'knn',crossval = 1):
        super(NucleiDataset, self).__init__( root, transform, pre_transform)
        setting = datasetting#DataSetting()
        self.epoch = 0
        self.val_epoch = 0
        self.graph_sampler = graph_sampler
        self.task = setting.name
        self.mask = mask
        self.setting = setting
        self.add_noise = add_noise
        self.dynamic_graph = dynamic_graph
        self.sample_method = sampling_method
        self.sampling_by_ratio = not sampling_by_number
        if self.dynamic_graph and sampling_method !='random':
            self.sampler = FarthestSampler()
        self.normalize = normalize
        self.sampling_time = sampling_time
        self.sampling_ratio = sampling_ratio
        self.max_edge_distance = setting.max_edge_distance
        self.max_num_nodes = int(setting.max_num_nodes * sampling_ratio) + 1
        self.max_neighbours = neighbour
        self.cross_val = crossval
        if mask == 'hvnet':
            setting.dataset = [f.strip('_s') +'_s' for f in setting.dataset]
        # self.use_polar = setting.use_polar
        self.split = split
        if self.split!='train':
            np.random.seed(1024)
        else:
            np.random.seed(None)
        # self.img_transform = transforms.Compose([
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.raw_dir_list = [os.path.join(setting.root, 'proto', 'feature', dataset, self.split) for dataset in setting.dataset]
        if setting.name == 'colon':
            _process_name = 'cross_val'
            # _process_name = 'processed_100_s' if mask == 'hvnet' else 'processed_100'
        else:
            _process_name = 'processed_iciar_aug'
        # _process_name = 'old_processed'
        # pdb.set_trace()
        self.processed_dir = []
        for fold in _CROSS_VAL[self.cross_val][split]:
            self.processed_dir.append(osp.join(self.root, 'proto', _process_name, fold))
        self.processed_root = os.path.join(self.root, 'proto', _process_name)
        self.processed_fix_data_root =  os.path.join(self.root, 'proto', 'fix_fuse_cia_knn')
        self.original_files = []
        # for raw_dir in self.raw_dir_list:
        #    self.original_files.extend([ osp.join(raw_dir, f) for f in os.listdir(raw_dir)])
        # add file name to label

        self.original_files = [ f.split('.npy')[0] for f in self.original_files ]
        # todo
        if self.setting.name =='colon':
            if mask=='cia':
                self.mean =  _MEAN_CIA[crossval]
                self.std = _STD_CIA[crossval]
               #  self.mean = [ 1.46221439e+02,  1.50919032e+01,  4.18933968e+02, -1.06722111e-01,
               #  4.28864839e+00,  1.43240486e+03,  2.77574693e+01,  1.33524025e+01,
               #  2.48558933e+02,  7.25978426e-01,  1.29617862e+02,  1.74934388e+01,
               #  8.96863659e+00,  4.31483840e+01,  8.74269732e-01,  7.99509656e+01, 1500,1500]
               #  self.std =[3.71039805e+01, 1.23434769e+01, 3.75002701e+02, 5.02443639e-01,
               # 1.91420502e-01, 1.79165845e+03, 3.47563458e+01, 8.38285628e+00,
               # 5.09069578e+02, 2.52385522e-01, 1.54908971e+02, 6.92765483e+01,
               # 6.09798648e+00, 2.89411105e+01, 2.00044592e-01, 5.46217527e+01, 1500, 1500]
            else:
                self.mean = [1.49234878e+02, 3.64214992e+01, 4.71188346e+02, 9.96754800e-02,
       4.33956586e+00, 2.26289686e+03, 4.31617967e+01, 1.75380005e+01,
       4.12394044e+02, 6.66048696e-01, 2.06727586e+02, 2.44344821e+01,
       1.22555443e+01, 5.38665971e+01, 8.56713621e-01, 7.60878663e+01,1500,1500]
                self.std =[2.89986135e+01, 2.81694989e+01, 3.43457305e+02, 4.66909809e-01,
       1.62459095e-01, 2.35831703e+03, 4.84423850e+01, 1.02378017e+01,
       6.12117537e+02, 2.89339729e-01, 2.58059180e+02, 1.23206879e+02,
       9.45993973e+00, 3.67439242e+01, 2.32119301e-01, 5.72582486e+01,1500,1500]
        else:
            self.mean =[1.42303737e+02, 4.45724086e+01, 5.12881523e+02, 1.17443112e-01,
        4.37827100e+00, 3.26860159e+03, 5.32320136e+01, 2.05980427e+01,
        6.00409991e+02, 6.69971418e-01, 2.77649096e+02, 2.58330915e+01,
        1.39345598e+01, 6.22481437e+01, 8.55130003e-01, 7.93717597e+01,768,1024]
            self.std =[1.94340848e+01, 2.31672883e+01, 3.05939918e+02, 4.07708236e-01,
        1.38779800e-01, 3.41345131e+03, 6.72855345e+01, 1.32714215e+01,
        1.52072208e+03, 2.71090807e-01, 3.72361045e+02, 1.15149545e+02,
        1.03044603e+01, 4.42820630e+01, 2.29416214e-01, 5.63687483e+01,768,1024]
        if feature_type == 'c':
            self.mean = self.mean[-2:]
            self.std = self.std[-2:]
        elif feature_type =='a':
            self.mean = self.mean[:-2]
            self.std = self.std[:-2]
        self.mean = torch.Tensor(self.mean)
        self.std = torch.Tensor(self.std)
        # elif feature_type == 'cl':n

        #     self.mean = self.mean[-8:]
        #     self.std = self.std[-8:]
        # elif feature_type == 'ca':
        #     self.mean = self.mean[:16]+ self.mean[-2:]
        #     self.std = self.std[:16] + self.std[-2:]
            #self.mean = [ 1.48644626e+02,  1.50220432e+01, -9.74926628e-02,  2.30577711e+02,
            #7.69403592e-01,  2.02606213e+01,  1.44985469e+02,  1.11584660e+01,
            #5.38992658e+01,1972,1972]
            #self.std = [3.01260501e+01 ,1.23251095e+01, 5.04242229e-01 ,2.80437427e+02,
            #1.54578013e-01 ,1.23724232e+01 ,3.86174444e+01 ,6.28408725e+00,
             #3.82686658e+01,1972,1972]
        # mkdirs(self.processed_dir)
        self.label_name = {'High': 2, 'Low': 1, 'Normal': 0, 'grade_2': 1, 'grade_1': 0, 'grade_3': 2}
        # import pdb;pdb.set_trace()
        # if len(os.listdir(self.processed_dir)) == 0:
        #    self.process()
        self.idxlist = []
        for folder in self.processed_dir:
            self.idxlist.extend([os.path.join(folder.split('/')[-1],f) for f in os.listdir(folder)])
        self.feature_type = feature_type


    @property
    def raw_file_names(self):
        return self.original_files

    @property
    def raw_paths(self):
        pass
        # r"""The filepaths to find in order to skip the download.
        # return is img name, we have different npy file _feature.npy, _distance.npy, _polar.npy
        # """arr
        #
        # files = to_list(self.raw_file_names)
        # for f in files:
        #     if 'Grade' in f:
        #         path = osp.join(self.raw_dir_list[0], f)
        # return [osp.join(self.raw_dir, 'feature-new', f) for f in files]

    @property
    def processed_file_names(self):
        processed_file_names = self.idxlist
        return processed_file_names

    def __len__(self):
        return len(self.processed_file_names)

    def _download(self):
        # Download to `self.raw_dir`.
        pass

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_val_epoch(self,epoch):
        self.val_epoch = epoch

    def _process(self):
        pass

    def _prepare_transformer(self):
        aug_dict = {'eraser':0.8}
        self.augments = get_composed_augmentations(aug_dict)

    def _prepare_img(self, idx):
        img = cv2.imread(os.path.join(self.root, 'raw', 'ICIAR', self.split, self.idxlist[idx].replace('.npy', '.tif')))
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.augments(img, None)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = self.img_transform(img)
        return img

    def _sampling(self, num_sample, ratio, distance_path = None):
        if self.mask == 'hvnet':
            distance_path = distance_path.replace('distance','distance_s')
        if self.sampling_by_ratio:
            num_subsample = int(num_sample * ratio)
            if self.task != 'colon':
                if num_sample < 100:
                    num_subsample = num_sample

        else:
            num_subsample = min(self.max_num_nodes, num_sample)
        if self.sample_method == 'farthest':

            distance = np.load(distance_path.replace('.pt','.npy'))
            indice = self.sampler(distance, num_subsample)

        elif self.sample_method == 'fuse':
            # 70% farthest, 30% random
            far_num =int( 0.7 * num_subsample)
            rand_num = num_subsample - far_num
            distance = np.load(distance_path.replace('.pt','.npy'))
            far_indice = self.sampler(distance, far_num)
            remain_item = filter_sampled_indice(far_indice, num_sample)
            rand_indice = np.asarray(random.sample(remain_item, rand_num))

            indice = np.concatenate((far_indice, rand_indice),0)

        else:
            # random
            indice = np.random.choice( num_sample,num_subsample, replace=False)

        return  indice, num_subsample


    def process(self):
        i = 0

        print('pre-process the graph dataset...')
        for raw_path in tqdm(self.original_files):
             # Read data from `raw_path`.

             for j in range(self.sampling_time):
                 data = self._read_one_raw_graph(raw_path)
                 # todo : visualize

                 if self.pre_filter is not None and not self.pre_filter(data):
                     continue

                 if self.pre_transform is not None:
                     data = self.pre_transform(data)
                 if self.sampling_time>1:
                    torch.save(data, osp.join(self.processed_dir, raw_path.split('/')[-1].split('.')[0] + '_' + str(j)+ '.pt'))
                 # torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                 else:
                     torch.save(data, osp.join(self.processed_dir,
                                               raw_path.split('/')[-1].split('.')[0] + '.pt'))
                 i += 1



    def _read_one_raw_graph(self, raw_file_path):
        # import pdb;pdb.set_trace()
        # the feature are [29 * appe
        # arance, hard-label, soft-label * 6, coordinate * 2]
        nodes_features = np.load( raw_file_path + '.npy')
        coordinates = np.load(raw_file_path.replace('feature', 'coordinate') + '.npy')
        # softlabel = np.load(raw_file_path.replace('feature-new', 'label')+'.npy')
        # softlabel = softlabel[:,2:]
        # nodes_features = coordinates
        nodes_features = np.concatenate((nodes_features, coordinates), axis= -1)
        # nodes_features = np.concatenate((nodes_features[:,:3], nodes_features[:, 4:]), -1)
        coordinates = torch.from_numpy(coordinates).to(torch.float)
        nodes_features = torch.from_numpy(nodes_features ).to(torch.float)
        if self.setting.name == 'colon':
            if 'tialab' in raw_file_path:
                if 'High' in raw_file_path:
                    label = 2
                elif 'Low' in raw_file_path:
                    label = 1
                else:
                    label = 0
            else:

                label = int(raw_file_path.split('_')[-2])-1
        else:
            if 'iv' in raw_file_path:
                label = 3
            elif 'is' in raw_file_path:
                label = 2
            elif 'b' in raw_file_path:
                label = 1
            else:
                label = 0
        y = torch.tensor([label], dtype=torch.long)
        if not self.dynamic_graph:
            if self.sampling_ratio!=1:
                num_nodes = nodes_features.shape[0]
                num_sample = int(num_nodes * self.sampling_ratio)
                choice = np.random.choice(num_nodes, num_sample, replace=False)
                coordinates = coordinates[choice]
                nodes_features = nodes_features[choice]
            edge_index = radius_graph(coordinates, self.max_edge_distance, None, True, self.max_neighbours)
            edge_index_dense = sparse_to_dense(edge_index)
            # todo: normlize
            if self.normalize:

                sqrt_deg = torch.diag(1.0/torch.sqrt(torch.sum(edge_index_dense,dim=0).squeeze()))
                edge_attribute = torch.matmul(torch.matmul(sqrt_deg, edge_index_dense), sqrt_deg).float()
            else:
                edge_attribute = edge_index_dense
            _, edge_attribute_sparse = dense_to_sparse(edge_attribute)
            data_after_norm = Data(x = nodes_features, pos=coordinates, y = y, edge_index= _, edge_attr=edge_attribute_sparse)
        else:
            data_after_norm = Data(x = nodes_features, pos = coordinates, y=y)
        return data_after_norm

    def _build_edges_from_coordinate(self, coordinates):
        # coordinate [num_nodes, xy-dimension]
        x_coor = coordinates[:, 1, None]
        y_coor = coordinates[:, 0, None]
        distance_x = (x_coor - x_coor.T)^2
        distance_y = (y_coor - y_coor.T)^2
        distance_matrix = np.sqrt((distance_x  + distance_y))
        distance_matrix[distance_matrix > self.max_edge_distance] = 0
        distance_matrix[distance_matrix <= self.max_edge_distance] = 1
        return distance_matrix


    def get(self, idx):

        # data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        name = self.idxlist[idx].split('/')[-1]
        # data = torch.load(osp.join(self.processed_dir, self.idxlist[idx]))
        adj_all = torch.zeros((self.max_num_nodes, self.max_num_nodes), dtype=torch.float)
        # transform
        if self.dynamic_graph:
            data = torch.load(osp.join(self.processed_root, self.idxlist[idx]))
            num_nodes = data.num_nodes
            if self.setting.name == 'colon':
                # todo change distance path
                dist_path = os.path.join(self.root, 'proto', 'distance','shaban', self.idxlist[idx])
                # dist_path  = os.path.join(self.root, 'proto', 'distance', self.split, self.idxlist[idx])
            else:
                dist_path = os.path.join(self.root, 'proto', 'distance/ICIAR', self.split, self.idxlist[idx]) # /media/amanda/HDD2T_1/warwick-research/data/ICIAR/coordinate

            choice, sample_num_node = self._sampling(num_nodes,
                                                     self.sampling_ratio,dist_path
                                                     )
            # sample_num_node = int(self.sampling_ratio * num_nodes)
            # choice = np.random.choice(num_nodes,sample_num_node , replace=False)
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) == num_nodes:
                    data[key] = item[choice]
            # generate the graph
            if self.graph_sampler == 'knn':
                edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
                adj = sparse_to_dense(edge_index)
            elif self.graph_sampler == 'nearest':
                edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True,
                                                       self.max_neighbours)
                adj = sparse_to_dense(edge_index)
            elif self.graph_sampler =='d_prob':
                if self.mask == 'hvnet':
                    dist_path = dist_path.replace('distance', 'distance_s')
                adj = prob_distance_graph(choice, dist_path,self.max_edge_distance, True,)
            else:
                adj = random_sample_graph(choice,dist_path,self.max_edge_distance, True, n_sample = self.max_neighbours)

            # if self.distance_prob_graph:
            #     if self.mask == 'hvnet':
            #         dist_path = dist_path.replace('distance', 'distance_s')
            #     adj = prob_distance_graph(choice, dist_path,self.max_edge_distance, True,)
            # else:
            #     if self.least_n == 0:
            #         edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
            #     else:
            #         edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
            #     adj = sparse_to_dense(edge_index)

            adj[adj>0] = 1
            #if self.feature_type == 'a':
                # pure appearance information
            #   adj[:,:] = 1.
            adj_all[: sample_num_node, :sample_num_node] = adj
        else:
            data = torch.load(osp.join(self.processed_fix_data_root, str(self.epoch), self.idxlist[idx]))
            if self.graph_sampler == 'knn':
                edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
                adj = sparse_to_dense(edge_index)
            elif self.graph_sampler == 'nearest':
                edge_index = data.edge_index#radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True,
                                                       #self.max_neighbours)
                adj = sparse_to_dense(edge_index)



            # adj = sparse_to_dense(data.edge_index)
            num_nodes = adj.shape[0]
            adj_all[: num_nodes, :num_nodes] = adj
        #sqrt_deg = torch.diag(1.0/torch.sqrt(torch.sum(adj,dim=0).squeeze()))
        #adj = torch.matmul(torch.matmul(sqrt_deg, adj), sqrt_deg).float()
        # construct to same shape as the biggest graph


        feature = data.x
        if self.feature_type == 'c':
            feature = feature[:, -2:]
        elif self.feature_type =='a':
            feature = feature[:, :-2]
        #
        # elif self.feature_type == 'ca':
        #     feature = torch.cat((feature[:, :16], feature[:, -2:]), -1)
        #feature = feature[ :, self.used_feature_idx]
        #feature = torch.cat([feature[:,:3], feature[:,4:]], -1)
        num_feature = feature.shape[1]
        feature = (feature - self.mean)/self.std

        if self.add_noise and (self.split == 'train'or self.split =='train_aug'):
            gaussian = np.random.normal(loc = 0, scale= 0.1, size = feature.shape)
            gaussian = torch.from_numpy(gaussian).to(torch.float)
            feature = feature + gaussian


        feature_all = torch.zeros((self.max_num_nodes, num_feature), dtype=torch.float)
        if self.dynamic_graph:
            feature_all[:sample_num_node] = feature
        else:
            feature_all[:num_nodes] = feature
        label = data.y
        # print(adj.dtype)
        if self.setting.name != 'colon':
            img = self._prepare_image(idx)

        # adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        # adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)
            return {'adj':adj_all,
                'feats':feature_all,
                'label':label,
                'num_nodes': sample_num_node if self.dynamic_graph else num_nodes,
                'img':img}
        else:
            idx = torch.tensor(idx)
            return {'adj':adj_all,
                'feats':feature_all,
                'label':label,
                'num_nodes': sample_num_node if self.dynamic_graph else num_nodes,
                'patch_idx': idx}

class NucleiDatasetTest(NucleiDataset):
    def __init__(self, root, feature_type, transform=None, pre_transform=None, split = 'train',
                 sampling_time=1, sampling_ratio=1, normalize=False, dynamic_graph=False,
                 sampling_method='farthest',  sampling_by_number= True,datasetting = None, mask = 'CIA',
                 neighbour = 8, graph_sampler = 'knn',crossval = 1):
        super(NucleiDatasetTest, self).__init__(root, feature_type, transform=transform,
                                                       pre_transform=pre_transform, split = split,
                                                       sampling_time=1, sampling_ratio=1,
                                                       normalize=normalize, dynamic_graph=False,
                                                       sampling_method=sampling_method, sampling_by_number=sampling_by_number,
                                                       datasetting = datasetting,mask=mask,neighbour=neighbour,graph_sampler =graph_sampler,crossval=crossval )
    def get(self, idx):
        # only support batch size = 1
        name = self.idxlist[idx].split('/')[-1]
        data = torch.load(osp.join(self.processed_root, self.idxlist[idx]))
        # generate the graph
        if self.setting.name == 'colon':
            dist_path = os.path.join(self.root, 'proto', 'distance', 'shaban', self.idxlist[idx])
            # dist_path = os.path.join(self.root, 'proto', 'distance', self.split, self.idxlist[idx])
        else:
            dist_path = os.path.join(self.root, 'proto', 'distance/ICIAR', self.split, self.idxlist[idx])
            dist_path = dist_path.replace('distance', 'distance_s')

        if self.graph_sampler == 'knn':
            edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
            adj_all = sparse_to_dense(edge_index)
        elif self.graph_sampler == 'nearest':
            edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True,
                                                   self.max_neighbours)
            adj_all = sparse_to_dense(edge_index)
        elif self.graph_sampler == 'd_prob':
            adj_all = prob_distance_graph(None, dist_path, self.max_edge_distance, True, )
        else:
            adj_all = random_sample_graph(None, dist_path, self.max_edge_distance, True, n_sample=self.max_neighbours)


        #
        # if self.least_n==0:
        #     edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
        # else:
        #     edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
        # adj_all = sparse_to_dense(edge_index)
        adj_all[adj_all>0]= 1
        # adj_all[:,:] = 1
        if self.feature_type == 'a':
            adj_all[adj_all>0] = 1
        num_nodes = adj_all.shape[0]
        feature = data.x
        if self.feature_type == 'c':
            feature = feature[:, -2:]
        elif self.feature_type == 'a':
            coor = feature[:,-2:]
            coor = (coor -1500)/1500.
            feature = feature[:, :-2]

        num_feature = feature.shape[1]
        feature = (feature - self.mean)/self.std
        label = data.y
        idx = torch.tensor([idx])
        return {'adj':adj_all,
                'feats':feature,
                'label':label,
                'num_nodes': num_nodes if self.dynamic_graph else num_nodes,
                'patch_idx': idx
                }




class NucleiDatasetBatchOutput(NucleiDataset):
    def __init__(self, root, feature_type, transform=None, pre_transform=None, split = 'train',
                 sampling_time=10, sampling_ratio=0.5, normalize=False, dynamic_graph = False, sampling_method = 'farthest',
                 sampling_by_number= True,datasetting = None, mask = 'CIA',neighbour = 8, add_noise = False,graph_sampler ='knn',crossval = 1):
        super(NucleiDatasetBatchOutput, self).__init__( root, feature_type, transform=transform, pre_transform=pre_transform, split = split,
                 sampling_time=sampling_time, sampling_ratio=sampling_ratio, normalize=normalize, dynamic_graph=dynamic_graph, sampling_method  = sampling_method
                                                        , sampling_by_number=sampling_by_number,datasetting=datasetting,mask=mask,neighbour=neighbour,
                                                        add_noise=add_noise,graph_sampler =graph_sampler,crossval = crossval )

    def get(self, idx):
        name = self.idxlist[idx].split('/')[-1]
        epoch = self.epoch if self.split=='train' else self.val_epoch
        if self.dynamic_graph:
            data = torch.load(osp.join(self.processed_root, self.idxlist[idx]))
        else:
            data = torch.load(osp.join(self.processed_fix_data_root, str(epoch), self.idxlist[idx]))
        if self.feature_type == 'c':
            data.x = data.x[:,-2:]
        elif self.feature_type =='a':
            data.x = data.x[:, :-2]
        if self.dynamic_graph:

            num_nodes = data.num_nodes
            if self.setting.name == 'colon':
                dist_path = os.path.join(self.root, 'proto', 'distance', 'shaban', self.idxlist[idx])
                # dist_path  = os.path.join(self.root, 'proto', 'distance', self.split, self.idxlist[idx])
            else:
                dist_path = os.path.join(self.root, 'proto', 'distance/ICIAR', self.split, self.idxlist[idx])


            choice, _ = self._sampling(num_nodes, self.sampling_ratio,
                                   dist_path)

            # choice = np.random.choice(num_nodes, int(self.sampling_ratio * num_nodes), replace=False)
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) == num_nodes:
                    data[key] = item[choice]
            if self.graph_sampler == 'knn':
                edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)

            elif self.graph_sampler == 'nearest':
                edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True,
                                                       self.max_neighbours)

            elif self.graph_sampler == 'd_prob':
                edge_index = prob_distance_graph(choice, dist_path, self.max_edge_distance, True, True)
            else:
                edge_index = random_sample_graph(choice, dist_path, self.max_edge_distance, True,
                                              n_sample=self.max_neighbours,sparse=True)
            #
            # if self.distance_prob_graph:
            #     if self.mask == 'hvnet':
            #         dist_path = dist_path.replace('distance', 'distance_s')
            #     edge_index = prob_distance_graph(choice,dist_path,self.max_edge_distance, True,True)
            # else:
            #     if self.least_n == 0:
            #         edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
            #     else:
            #         edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True,
            #                                                self.max_neighbours)
            data.edge_index = edge_index
        else:
            if self.graph_sampler == 'knn':
                edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours)
                data.edge_index = edge_index
            elif self.graph_sampler == 'nearest':
                edge_index = radius_graph_with_least_k(data.pos, self.max_edge_distance, None, True,
                                                       self.max_neighbours)
                data.edge_index = edge_index

        data.patch_idx = torch.tensor([idx])
        data.x = (data.x - self.mean) / self.std
        if self.add_noise and self.split == 'train' or self.split =='train_aug':
            gaussian = np.random.normal(loc=0, scale=0.1, size=data.x.shape)
            gaussian =  torch.from_numpy(gaussian).to(torch.float)
            data.x = data.x + gaussian

        if self.setting.name != 'colon':
            img = self._prepare_image(idx)
            data.img = img
        return data

class NucleiDatasetListOutput(NucleiDataset):
    def __init__(self, root, feature_type, transform=None, pre_transform=None, split = 'train',
                 sampling_time=10, sampling_ratio=0.5, normalize=False, dynamic_graph=False,
                 sampling_method='farthest',  sampling_by_number= True,datasetting = None, mask = 'CIA',neighbour = 8,graph_sampler  = 'knn',crossval = 1):
        super(NucleiDatasetListOutput, self).__init__(root, feature_type, transform=transform,
                                                       pre_transform=pre_transform, split = split,
                                                       sampling_time=sampling_time, sampling_ratio=sampling_ratio,
                                                       normalize=normalize, dynamic_graph=dynamic_graph,
                                                       sampling_method=sampling_method,  sampling_by_number=sampling_by_number,
                                                       datasetting = datasetting,
                                                      mask=mask,
                                                      neighbour=neighbour,
                                                      graph_sampler =graph_sampler,
                                                      crossval= crossval )
################ rewrite cross val dataset class #############
