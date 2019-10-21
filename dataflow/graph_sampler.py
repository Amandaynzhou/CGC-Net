import networkx as nx
import numpy as np
import torch
import torch.utils.data

# class GraphSampler(torch.utils.data.Dataset):
#     ''' Sample graphs and nodes in graph
#     '''
#     def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
#         self.adj_all = []
#         self.len_all = []
#         self.feature_all = []
#         self.label_all = []
#
#         self.assign_feat_all = []
#
#         if max_num_nodes == 0:
#             self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
#         else:
#             self.max_num_nodes = max_num_nodes
#
#         #if features == 'default':
#         self.feat_dim = G_list[0].node[0]['feat'].shape[0]
#
#         for G in G_list:
#             adj = np.array(nx.to_numpy_matrix(G))
#             if normalize:
#                 sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
#                 adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
#             self.adj_all.append(adj)
#             self.len_all.append(G.number_of_nodes())
#             self.label_all.append(G.graph['label'])
#             # feat matrix: max_num_nodes x feat_dim
#             if features == 'default':
#                 f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
#                 for i,u in enumerate(G.nodes()):
#                     f[i,:] = G.node[u]['feat']
#                 self.feature_all.append(f)
#             elif features == 'id':
#                 self.feature_all.append(np.identity(self.max_num_nodes))
#             elif features == 'deg-num':
#                 degs = np.sum(np.array(adj), 1)
#                 degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
#                                       axis=1)
#                 self.feature_all.append(degs)
#             elif features == 'deg':
#                 self.max_deg = 10
#                 degs = np.sum(np.array(adj), 1).astype(int)
#                 degs[degs>max_deg] = max_deg
#                 feat = np.zeros((len(degs), self.max_deg + 1))
#                 feat[np.arange(len(degs)), degs] = 1
#                 feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
#                         'constant', constant_values=0)
#
#                 f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
#                 for i,u in enumerate(G.nodes()):
#                     f[i,:] = G.node[u]['feat']
#
#                 feat = np.concatenate((feat, f), axis=1)
#
#                 self.feature_all.append(feat)
#             elif features == 'struct':
#                 self.max_deg = 10
#                 degs = np.sum(np.array(adj), 1).astype(int)
#                 degs[degs>10] = 10
#                 feat = np.zeros((len(degs), self.max_deg + 1))
#                 feat[np.arange(len(degs)), degs] = 1
#                 degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
#                         'constant', constant_values=0)
#
#                 clusterings = np.array(list(nx.clustering(G).values()))
#                 clusterings = np.expand_dims(np.pad(clusterings,
#                                                     [0, self.max_num_nodes - G.number_of_nodes()],
#                                                     'constant'),
#                                              axis=1)
#                 g_feat = np.hstack([degs, clusterings])
#                 if 'feat' in G.node[0]:
#                     node_feats = np.array([G.node[i]['feat'] for i in range(G.number_of_nodes())])
#                     node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
#                                         'constant')
#                     g_feat = np.hstack([g_feat, node_feats])
#
#                 self.feature_all.append(g_feat)
#
#             if assign_feat == 'id':
#                 self.assign_feat_all.append(
#                         np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
#             else:
#                 self.assign_feat_all.append(self.feature_all[-1])
#
#         self.feat_dim = self.feature_all[0].shape[1]
#         self.assign_feat_dim = self.assign_feat_all[0].shape[1]
#
#     def __len__(self):
#         return len(self.adj_all)
#
#     def __getitem__(self, idx):
#         adj = self.adj_all[idx]
#         num_nodes = adj.shape[0]
#         adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
#         adj_padded[:num_nodes, :num_nodes] = adj
#
#         # use all nodes for aggregation (baseline)
#
#         return {'adj':adj_padded,
#                 'feats':self.feature_all[idx].copy(),
#                 'label':self.label_all[idx],
#                 'num_nodes': num_nodes,
#                 'assign_feats':self.assign_feat_all[idx].copy()}
from torch_cluster import radius
from torch_geometric.nn import knn
def radius_graph_with_least_k(x,
                 r,
                 batch=None,
                 loop=False,
                 max_num_neighbors=32,
                 flow='source_to_target',
                              least_neighbour = 0):
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import radius_graph

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.tensor([0, 0, 0, 0])
        >>> edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    row, col = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    row2, col2 = knn(x, x, 2, batch, batch)
    if not loop:
        mask = row != col
        mask2 = row2 != col2
        row2, col2 = row2[mask2], col2[mask2]
        row, col = row[mask], col[mask]
        row = torch.cat([row,row2],dim=0)
        col = torch.cat([col,col2], dim = 0)
    return torch.stack([row, col], dim=0)
EPS = 0.00001

def prob_distance_graph(choice = None, distance_path= None, max_edge_distance= 100, loop=False, sparse = False):
    '''

    :param distance: [n x n]
    :param max_edge_distance: float
    :param batch:
    :param loop:
    :return:
    '''
    distance = np.load(distance_path.replace('.pt', '.npy'))
    # print(choice)
    # print(distance.shape)
    if choice is not None:
        distance = distance[choice]
        distance = distance[:,choice]
    # print(distance.shape)
    distance = 2/np.sqrt(distance + EPS)
    distance[distance<2/np.sqrt(max_edge_distance+EPS)] = 0
    distance[distance>1] = 1
    edges = np.random.binomial(1,distance)
    if sparse:
        row,col = np.nonzero(edges)
        row = torch.from_numpy(row)
        col = torch.from_numpy(col)
        if not loop:
            mask = row != col
            row,col = row[mask],col[mask]
        return torch.stack([row, col], dim=0)
    else:
        if not loop:
            edges[np.eye(distance.shape[0])] = 0
        edges = torch.from_numpy(edges)
        return edges


def random_sample_graph(choice = None,distance_path = None, max_edge_distance = 100, loop=False, sparse = False, n_sample = 4):
    '''

    :param distance: [n x n]
    :param max_edge_distance: float
    :param batch:
    :param loop:
    :return:
    '''
    distance = np.load(distance_path.replace('.pt', '.npy'))
    # print(choice)
    # print(distance.shape)
    if choice is not None:
        distance = distance[choice]
        distance = distance[:, choice]
    # print(distance.shape)
    distance[distance==0] =1
    distance[distance>max_edge_distance]=0
    distance[distance > 0] = 1
    distance = distance/distance.sum(-1)[:,None]
    # print(distance)
    # print(distance.sum(-1))
    cul_d = distance.cumsum(-1)
    adj = np.zeros_like(distance,dtype=float)

    for i in range(n_sample):
        edges = np.random.rand(distance.shape[0], 1)
        choice = (edges<cul_d).argmax(1)
        adj[np.arange(adj.shape[0]), choice] = 1
        adj[choice, np.arange(adj.shape[1])] = 1
    if sparse:
        row, col = np.nonzero(adj)
        row = torch.from_numpy(row)
        col = torch.from_numpy(col)
        if not loop:
            mask = row != col
            row, col = row[mask], col[mask]
        return torch.stack([row, col], dim=0)
    else:
        if not loop:
            adj[np.eye(distance.shape[0])] = 0
        adj = torch.from_numpy(adj)
        return adj


def random_sample_graph2(choice = None,distance=None, max_edge_distance = 100, loop=False, sparse = False, n_sample = 4):
    '''

    :param distance: [n x n]
    :param max_edge_distance: float
    :param batch:
    :param loop:
    :return:
    '''

    if choice is not None:
        distance = distance[choice]
        distance = distance[:, choice]
    # print(distance.shape)
    distance[distance==0] =1
    distance[distance>max_edge_distance]=0
    distance[distance > 0] = 1
    distance = distance/distance.sum(-1)[:,None]
    # print(distance)
    # print(distance.sum(-1))
    cul_d = distance.cumsum(-1)
    adj = np.zeros_like(distance,dtype=float)

    for i in range(n_sample):
        edges = np.random.rand(distance.shape[0], 1)
        choice = (edges<cul_d).argmax(1)
        adj[np.arange(adj.shape[0]), choice] = 1
        adj[choice, np.arange(adj.shape[1])] = 1
    if sparse:
        row, col = np.nonzero(adj)
        row = torch.from_numpy(row)
        col = torch.from_numpy(col)
        if not loop:
            mask = row != col
            row, col = row[mask], col[mask]
        return torch.stack([row, col], dim=0)
    else:
        if not loop:
            adj[np.eye(distance.shape[0])] = 0
        adj = torch.from_numpy(adj)
        return adj