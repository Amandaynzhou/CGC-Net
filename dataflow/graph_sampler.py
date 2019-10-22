import numpy as np
import torch
import torch.utils.data

def random_sample_graph2(choice = None, distance=None, max_edge_distance = 100, loop=False, sparse = False, n_sample = 4):
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