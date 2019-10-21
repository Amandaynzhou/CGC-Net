import torch
import torch.nn.functional as F
import pdb
def row_wise_entropy_regularization(assign_matrix, batch_num_nodes=None):
    if batch_num_nodes is None:
        print('Warning: calculating entropy_regularization without masking')
    loss = torch.log(assign_matrix)



def balanced_cluster_loss(assign_matrix, mask = None):
    batch_size, num_nodes, num_clusters = assign_matrix.size()
    # pdb.set_trace()
    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(assign_matrix.dtype)
        num_nodes =  torch.sum(mask, 1)
        assign_matrix = assign_matrix * mask
    mean = num_nodes/(num_clusters -1)
    cluster_size = assign_matrix.sum(1)
    loss = 1/(num_clusters-1) * torch.norm(cluster_size-mean, p = 2, dim = 1)
    return loss