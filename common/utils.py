import os
import numpy as np
from torch_geometric.utils import sparse_to_dense, dense_to_sparse, to_networkx
from torch_geometric.nn import radius_graph
import matplotlib.pyplot as plt
import networkx as nx
import torch
import cv2
from scipy import io as sio
import shutil
from torch_geometric.data import Data
from torch.nn import Parameter
from scipy.spatial.distance import euclidean
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageOps,ImageCms
import random
import torchvision.transforms.functional as tf


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count>0:
            self.avg = self.sum / self.count


#graph read write function
def pt_to_gexf(numpyfile, savepath, sample = [0.1, 0.2, 0.3, 0.4 ,0.5 , 0.6, 0.7 ,0.9, 1]):
    # data = torch.load(torchfile)
    # feature = data.x.numpy()
    coordinates = np.load(numpyfile)
    coordinates = coordinates/1500.
    coordinates = torch.from_numpy(coordinates).to(torch.float)
    num_nodes = coordinates.shape[0]
    for sampling_ratio in sample:
        num_sample = int(num_nodes * sampling_ratio)
        choice = np.random.choice(num_nodes, num_sample, replace=True)
        coordinates_new = coordinates[choice]

        edge_index = radius_graph(coordinates_new, 100, None, True, 8)
        adj = sparse_to_dense(edge_index)
        adj = adj.numpy()
        coordinates_new = coordinates_new.numpy()
        G = nx.from_numpy_matrix(adj)
        x = dict(enumerate(coordinates_new[:, 0].tolist(), 0))
        y = dict(enumerate(coordinates_new[:, 1].tolist(), 0))
        nx.set_node_attributes(G, x, 'x')
        nx.set_node_attributes(G, y, 'y')
        nx.write_gexf(G, os.path.join( savepath, str(sampling_ratio) + numpyfile.split('/')[-1].replace('.npy','.gexf')))


def output_to_gexf(coordinate, adj, assign_matrix_list, path):
    '''
        Visualize the assignment matrix
    '''
    x = dict(enumerate(coordinate[:,0].tolist(), 0))
    y = dict(enumerate(coordinate[:,1].tolist(), 0))
    assign_matrix_dicts = []
    for idx, assign_matrix in enumerate(assign_matrix_list):
        assign_matrix = np.argmax(assign_matrix, axis=1)
        assign_matrix = dict(enumerate(assign_matrix.flatten(),0))
        assign_matrix = {k: int(v)  for k, v in assign_matrix.items()}
        assign_matrix_dicts.append(assign_matrix)
    mapping_assign_dicts = {}
    mapping_assign_dicts['assign_1'] = assign_matrix_dicts[0].copy()
    if len(assign_matrix_dicts)>1:
        for idx, deeper_assign_matrix in enumerate(assign_matrix_dicts, 1):
            if idx == 1:
                first_assign = assign_matrix_dicts[0].copy()
                continue
            map_assign_matrix = dict([(k, deeper_assign_matrix[v]) for (k,v) in first_assign.items()])
            mapping_assign_dicts['assign_%d'%idx] = map_assign_matrix.copy()
            first_assign = map_assign_matrix.copy()
            # not sparse format
    assert  adj.shape[0] == adj.shape[1], 'the adjacent matrix should have same row and col'
    G = nx.from_numpy_matrix(adj)
    # import pdb;pdb.set_trace()
    for k,v in mapping_assign_dicts.items():
    #     name = namestr[idx]
        nx.set_node_attributes(G, v, k)
    nx.set_node_attributes(G, x, 'x')
    nx.set_node_attributes(G, y, 'y')
    nx.write_gexf(G, path)

# load save functions
def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdirs(os.path.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'model_best.pth.tar'))

def load_checkpoint(fpath):
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model
# optimizer

def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))


# other temp things
def dataset_analysis(keys):
    train_root_list = ['/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_1/1_normal',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_1/2_low_grade',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_1/3_high_grade',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_2/1_normal',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_2/2_low_grade',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_2/3_high_grade',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_3/1_normal',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_3/2_low_grade',
                       '/research/dept6/ynzhou/gcnn/data/proto/feature/shaban/fold_3/3_high_grade',
                       ]
    # test_root_list =['/media/amanda/HDD2T_1/warwick-research/data/proto/feature/ICIAR/valid']
    # train_root_list  =  ['/media/amanda/HDD2T_1/warwick-research/data/proto/feature/tialab/train','/media/amanda/HDD2T_1/warwick-research/data/proto/feature/extra/train']
    # test_root_list = ['/media/amanda/HDD2T_1/warwick-research/data/proto/feature/tialab/valid','/media/amanda/HDD2T_1/warwick-research/data/proto/feature/extra/valid']
    if 'max_node' in keys:
        trainall, train_max_node_number= max_nodes_in_dataset(train_root_list)
        # testall, test_max_node_number = max_nodes_in_dataset(test_root_list)
        print(train_max_node_number, )
    elif'normalized' in keys:
        norm1,std1 = normalization(train_root_list)
        # norm2,std2 = normalization(test_root_list)
        return norm1,std1

def mat2np(path):
    f = sio.loadmat(path)
    f = f['predicted_map']
    f = f.astype(np.int16)
    np.save(path.replace('predicted_map.mat', '.npy'), f)


def max_nodes_in_dataset(pathlist):
    all_number = []
    for path in pathlist:
        lists = os.listdir(path)

        for f in lists:
            g = np.load(os.path.join(path, f))
            nodes = g.shape[0]
            all_number.append(nodes)
    maxn = max(all_number)
    return all_number, maxn


def normalization(pathlist):
    all_number = []
    for path in pathlist:
        lists = os.listdir(path)
        for f in lists:
            g = np.load(os.path.join(path, f)).astype(np.double)
            if g.shape[1] == 10:
                print(f)
            all_number.append(g)
            # print(g.shape)
    allins = np.vstack(all_number)
    # allins = np.concatenate((allins[:,:3], allins[:,4:]),-1)
    norm = np.mean(allins, 0)
    std = np.std(allins, 0)
    return norm, std

def plt_graphs(graphlist):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)
    import pdb;pdb.set_trace()
    for i, graph in enumerate(graphlist):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = graph.x.shape[0]
        adj_matrix = sparse_to_dense(graph.edge_index, graph.edge_attr ).numpy()
        G =  nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=30,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()
    fig.savefig('graph.png')

def plt_graph_on_img(name = '/media/bialab/98d97ea6-026f-4815-8c11-72e20967da7f/proto/graph/train/coordinate/H08-19835_A1H&E_1_3_grade_1_0001_1793.npy',distance=80 ,number =10,bbox=[500,2000,500,2000],):
    coordinate =  np.load(name)
    img_path = '/media/bialab/98d97ea6-026f-4815-8c11-72e20967da7f/Datasets/train/1_normal/H08-19835_A1H&E_1_3_grade_1_0001_1793.png'
    # filter
    img = cv2.imread(img_path)


    top, bottom, left, right = bbox[0], bbox[1], bbox[2], bbox[3]
    img = img[ int(left/2):int(right/2), int(top/2):int(bottom/2)]
    img = cv2.resize(img, (right-left,bottom-top))
    # import pdb;pdb.set_trace()
    y_range = coordinate[coordinate[:,0]>top]
    y_range = y_range[y_range[:,0]<bottom]
    x_range = y_range[y_range[:,1]>left ]
    x_range = x_range[x_range[:,1]<right]
    x_range = torch.from_numpy(x_range).float()
    x_range = x_range - top
    graph = radius_graph(x_range, distance, None, True, number)
    data= Data(x = x_range, pos=x_range, edge_index= graph)
    G = to_networkx(data)
    dpi = 300
    figsize  =  (bottom-top)/ float(dpi), (right-left) / float(dpi)
    plt.switch_backend('agg')
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(img)

    pos = {}
    x_range = x_range.numpy()
    for i in range(x_range.shape[0]):
        pos[i] = (x_range[i,1], x_range[i,0])
    nx.draw(G ,pos=pos,node_color='#336699',width=0.2, node_size=2,
                edge_color='grey',arrows = False)
    # draw a circle
    circle1 = plt.Circle(pos[120], 100, color='b', fill=False )
    circle2 = plt.Circle(pos[120], 80, color='g', fill=False )
    circle3 = plt.Circle(pos[120], 60, color='r', fill=False )
    circle4 = plt.Circle(pos[120], 40, color='black', fill=False )
    plt.gcf().gca().add_artist(circle1)
    plt.gcf().gca().add_artist(circle2)
    plt.gcf().gca().add_artist(circle3)
    plt.gcf().gca().add_artist(circle4)
    # ax.add_artist(circle)
    # nx.draw_networkx_edges(G,pos=nx.spring_layout(G))
    # plt.tight_layout()
    # fig.canvas.draw()
    fig.savefig('graph.png')

def get_bbox(mask):
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    return [y1,y2,x1,x2]
    # return x1,x2,y1,y2


# def dist_ponto_cj(ponto,lista):
#     return [euclidean(ponto, lista[j]) for j in range(len(lista))]
#
# def ponto_mais_longe(pts, lista_ds):
#     ds_max = max(lista_ds)
#     idx = lista_ds.index(ds_max)
#     return pts[idx]
#
# def farthest_sampling(pts, K):
#     N = pts.shape[0]
#     farthest_pts = [0] * K
#     P0 = pts[np.random.randint(0, N)]
#     farthest_pts[0] = P0
#     ds0 = dist_ponto_cj(P0, pts)
#
#     ds_tmp = ds0
#     for i in range(1, K):
#         farthest_pts[i] = ponto_mais_longe(pts, ds_tmp)
#         ds_tmp2 = dist_ponto_cj(farthest_pts[i], pts)
#         ds_tmp = [min(ds_tmp[j], ds_tmp2[j]) for j in range(len(ds_tmp))]

    # return farthest_pts

class FarthestSampler2:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts = np.zeros((k, 2), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts


class FarthestSampler:
    def __init__(self):
        pass
    def __call__(self, arr, k):
        farthest_pts = np.zeros((k), dtype=np.int32)
        farthest_pts[0] = np.random.randint(arr.shape[0])
        farthest_distance = arr[farthest_pts[0]]
        for i in range(1,k):
            farthest_pts[i] = np.argmax(farthest_distance)
            farthest_distance = np.minimum(farthest_distance, arr[farthest_pts[i]])
        return farthest_pts

def filter_sampled_point(ori, sampled):
    sample_hash = map(tuple, sampled)
    sample_hash = set(sample_hash)
    ori_hash = map(tuple, ori)
    ori_hash = set(ori_hash)
    remain = ori_hash - sample_hash
    remain = np.vstack(list(remain))
    return remain

def filter_sampled_indice(indice, num):
    total = [i for i in range(num)]
    a = list(filter(lambda x:x not in indice, total))
    return a

