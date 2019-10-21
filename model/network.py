import torch
import torch.nn as nn
from torch.nn import init
from model.losses import balanced_cluster_loss
import torch.nn.functional as F
import numpy as np
from setting import ModelSetting
from torch_scatter import scatter_max,scatter_mul
from torch_geometric.nn import DenseSAGEConv,dense_diff_pool,GCNConv,GraphConv, GATConv, SAGEConv,GINConv,DenseGINConv,JumpingKnowledge
import torch
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.data import Batch
from torch_geometric.utils import sparse_to_dense,to_dense_batch
from model.utils import to_dense_adj
from torch.nn import Linear, LSTM
from torchvision.models.resnet import resnet18

EPS = 1e-15
import pdb
from torch_geometric.nn import JumpingKnowledge

# from torchvision.models import ResNet
# nn.SyncBatchNorm




class DenseJK(nn.Module):
    def __init__(self, mode, channels=None, num_layers=None):
        super(DenseJK, self).__init__()
        self.channel = channels
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None
            assert num_layers is not None
            self.lstm = LSTM(
                channels,
                channels  * num_layers // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * channels * num_layers // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        r"""Aggregates representations across different layers.

        Args:
            xs  [batch, nodes, featdim*3]
        """

        xs = torch.split(xs, self.channel, -1)# list of batch, node, featdim
        xs = torch.stack(xs,2)#[batch, nodes, num_layers, num_channels]
        shape = xs.shape
        x = xs.reshape((-1,shape[2],shape[3]))  # [ngraph * num_nodes , num_layers, num_channels]
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [ngraph * num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        x =  (x * alpha.unsqueeze(-1)).sum(dim=1)
        x = x.reshape((shape[0],shape[1],shape[3]))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)



class SGNN_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim,embedding_dim, bias, bn, add_loop, lin = True,gcn_name='SAGE', sync = False,
                 activation = 'relu'):
        super(SGNN_Module, self).__init__()
        self.add_loop = add_loop
        self.gcn1 = self._gcn(gcn_name, input_dim, hidden_dim,
                              bias,activation)  # DenseSAGEConv(input_dim, hidden_dim, normalize= True, bias= bias)
        self.active1 = self._activation(activation)
        self.gcn2 = self._gcn(gcn_name, hidden_dim, embedding_dim,
                              bias,activation)  # DenseSAGEConv(hidden_dim, embedding_dim, normalize=True, bias= bias)
        self.active2 = self._activation(activation)
        if bn:
            if sync:
                self.bn1 = nn.SyncBatchNorm(hidden_dim)
            else:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(embedding_dim)
        if lin is True:
            self.lin = torch.nn.Linear(hidden_dim + embedding_dim,
                                       embedding_dim)
        else:
            self.lin = None

    def _activation(self, name = 'relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name =='leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def _gcn(self, name, input_dim, hidden_dim, bias, activation = 'relu'):
        if name == 'SAGE':
            return DenseSAGEConv(input_dim, hidden_dim, normalize=True, bias=bias)
        else:
            nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), self._activation(activation),
                                nn.Linear(hidden_dim, hidden_dim))
            return DenseGINConv(nn1)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()l

        x0 = x
        x1 = self.bn(1, self.active1(self.gcn1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, self.active2(self.gcn2(x1, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2], dim=-1)

        if mask is not None:
            x = x * mask
        if self.lin is not None:
            x = self.lin(x)
            if mask is not None:
                x = x * mask
        return x



class GNN_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim,embedding_dim, bias, bn, add_loop, lin = True,gcn_name='SAGE', sync = False
                 ,activation = 'relu', jk = False):
        super(GNN_Module, self).__init__()
        self.jk = jk
        self.add_loop = add_loop
        self.gcn1 = self._gcn(gcn_name,input_dim, hidden_dim,bias, activation)#DenseSAGEConv(input_dim, hidden_dim, normalize= True, bias= bias)
        self.active1 = self._activation(activation)
        self.gcn2 = self._gcn(gcn_name,hidden_dim, hidden_dim,bias, activation)#DenseSAGEConv(hidden_dim, hidden_dim, normalize= True, bias= bias)
        self.active2 = self._activation(activation)
        self.gcn3 = self._gcn(gcn_name,hidden_dim, embedding_dim,bias, activation)#DenseSAGEConv(hidden_dim, embedding_dim, normalize=True, bias= bias)
        self.active3 = self._activation(activation)
        if bn:
            if sync:
                self.bn1 = nn.SyncBatchNorm(hidden_dim)
            else:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                self.bn3 = nn.BatchNorm1d(embedding_dim)
        # if self.jk:
        #     self.jk_layer = JumpingKnowledge('lstm')
        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_dim + embedding_dim,
                                       embedding_dim)
        else:
            self.lin = None

    def _activation(self, name = 'relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name =='leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def _gcn(self,name, input_dim, hidden_dim,bias, activation='relu'):
        if name == 'SAGE':
            return DenseSAGEConv(input_dim, hidden_dim, normalize= True, bias= bias)
        else:
            nn1 = nn.Sequential(nn.Linear(input_dim,  hidden_dim), self._activation(activation),
                                nn.Linear( hidden_dim,  hidden_dim))
            return DenseGINConv(nn1)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()l


        x0 = x
        x1 = self.bn(1,self.active1(self.gcn1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2,self.active2(self.gcn2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3,self.active3(self.gcn3(x2, adj, mask, self.add_loop)))
        # if not self.jk:
        x = torch.cat([x1, x2, x3], dim=-1) #batch , node, (feat-dim * 3)
        # else:
        #     x =

        if mask is not None:
            x = x * mask
        if self.lin is not None:
            x = self.lin(x)
            if mask is not None:
                x = x * mask
        return x



class SoftOnePoolingGcnEncoder(nn.Module):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, bias, bn, assign_hidden_dim,label_dim,
                 assign_ratio=0.25,  pred_hidden_dims=[50], linkpred=True, concat = True,gcn_name='SAGE'):
        super(SoftOnePoolingGcnEncoder, self).__init__()

        assign_dim = int(max_num_nodes * assign_ratio)
        self.linkpred = linkpred
        self.GCN_embed_1 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn, add_loop= False, lin=False,gcn_name=gcn_name)
        self.GCN_pool_1 = GNN_Module(input_dim, assign_hidden_dim, assign_dim, bias, bn, add_loop= False,gcn_name=gcn_name)
        if concat:
            input_dim = hidden_dim * 2 + embedding_dim
        else:
            input_dim = embedding_dim

        assign_dim = int(assign_dim * assign_ratio)
        self.GCN_embed_2 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn, add_loop= False,lin=False,gcn_name=gcn_name)
        self.pred_model = self.build_readout_module(input_dim * 2, pred_hidden_dims,
                                                    label_dim, )
    def build_readout_module(self,pred_input_dim, pred_hidden_dims, label_dim, ):
        pred_input_dim = pred_input_dim
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(nn.ReLU(True))
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def forward(self,  x, adj, batch_num_nodes, label = None):

        out_all = []
        # stage 1
        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)

        out, _ = torch.max(embed_feature, dim = 1)
        out_all.append(out)
        assign = self.GCN_pool_1(x, adj, embedding_mask)

        x, adj, link_loss1, ent_loss1 = self._diff_pool(embed_feature, adj, assign, embedding_mask, self.linkpred)
        # stage 2
        embed_feature = self.GCN_embed_2(x, adj, None)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)

        output = torch.cat(out_all, 1)
        del out_all
        output = self.pred_model(output)
        if self.training:
            if self.linkpred:
                link_loss = link_loss1
            else:
                link_loss = torch.zeros((1)).cuda()
            ent_loss = ent_loss1
            cls_loss = F.cross_entropy(output, label, size_average=True)
            return output, link_loss, ent_loss, cls_loss
        return output


    def construct_mask(self, max_nodes, batch_num_nodes):
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def _diff_pool(self, x, adj, s, mask, linkpred):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()
        if  not linkpred:
            return  out, out_adj, None, ent_loss
        else:
            link_loss = adj - torch.matmul(s, s.transpose(1, 2))
            link_loss = torch.norm(link_loss, p=2)
            link_loss = link_loss / adj.numel()
            return out, out_adj, link_loss, ent_loss


class SoftPoolingGcnEncoder(nn.Module):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, bias, bn, assign_hidden_dim,label_dim,
                 assign_ratio=0.25,  pred_hidden_dims=[50], linkpred=True, concat = True, gcn_name='SAGE',
                 collect_assign = False, balanced_cluster = False, load_data_sparse = False,norm_adj=False,
                 activation = 'relu', readout = 'max', drop_out = 0., noise = False,jk = False, task = 'colon'):


        super(SoftPoolingGcnEncoder, self).__init__()

        if task == 'ICIAR':
            self.resnet18 = resnet18(True, num_classes=4)
        self.jk = jk
        self.drop_out = drop_out
        self.norm_adj = norm_adj
        self.readout = readout
        self.load_data_sparse = load_data_sparse
        self.balanced_cluster = balanced_cluster
        self.collect_assign = collect_assign
        self.assign_matrix = []
        assign_dim = int(max_num_nodes * assign_ratio)
        self.linkpred = linkpred
        self.GCN_embed_1 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False, lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk1 = DenseJK('lstm',  hidden_dim, 3)
        self.GCN_pool_1 = GNN_Module(input_dim, assign_hidden_dim, assign_dim, bias, bn,
                                     add_loop= False, gcn_name=gcn_name,activation=activation, jk = jk)

        if concat and not jk:
            input_dim = hidden_dim * 2 + embedding_dim
        else:
            input_dim = embedding_dim

        assign_dim = int(assign_dim * assign_ratio)
        self.GCN_embed_2 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk2 = DenseJK('lstm', hidden_dim , 3)
        self.GCN_pool_2 = GNN_Module(input_dim, assign_hidden_dim, assign_dim, bias, bn,
                                     add_loop= False, gcn_name=gcn_name,activation=activation, jk = jk)

        self.GCN_embed_3 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk3 = DenseJK('lstm', hidden_dim, 3)

        if readout =='max':
            pred_input = input_dim * 3
        else :
            pred_input = input_dim * 6

        self.pred_model = self.build_readout_module(pred_input, pred_hidden_dims,
                                                    label_dim, activation)


    @staticmethod
    def construct_mask( max_nodes, batch_num_nodes):
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()


    def _re_norm_adj(self,adj,p, mask = None):
        # pdb.set_trace()
        idx = torch.arange(0, adj.shape[1],out=torch.LongTensor())
        adj[:,idx,idx] = 0
        new_adj =  torch.div(adj,adj.sum(-1)[...,None] + EPS)*(1-p)
        new_adj[:,idx,idx] = p
        if mask is not None:
            new_adj = new_adj * mask
        return new_adj

    def _diff_pool1(self, x, adj, s, mask, linkpred):
        balanced_loss = None
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
            # ent_entries = 1
            # entries = adj.numel()
            # pdb.set_trace()
            ent_entries = torch.sum(mask, 1)
            entries = torch.sum(torch.pow(ent_entries, 2))
        else:
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = adj.shape[1]
            entries = adj.numel()
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean() / ent_entries
        if self.balanced_cluster:
            balanced_loss = balanced_cluster_loss(s, mask)

        if not linkpred:
            return out, out_adj, None, ent_loss, balanced_loss
        else:
            # pdb.set_trace()

            link_loss = adj - torch.matmul(s, s.transpose(1, 2))
            link_loss = torch.norm(link_loss, p=2)

            link_loss = link_loss / entries  # adj.numel()

            return out, out_adj, link_loss, ent_loss, balanced_loss

    def _diff_pool2(self, x, adj, s, mask, linkpred):
        balanced_loss = None
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        # pdb.set_trace()
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
            # ent_entries = 1
            # entries = adj.numel()

            ent_entries = torch.sum(mask, 1)[:,0]
            entries = torch.pow(ent_entries, 2)
        else:
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = torch.tensor([adj.shape[1]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
            entries = torch.tensor([adj.numel()/adj.shape[0]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
        # pdb.set_trace()
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        ent_loss = torch.div((-s * torch.log(s + EPS)).sum(dim=-1).mean(-1),ent_entries)
        if self.balanced_cluster:
            balanced_loss = balanced_cluster_loss(s, mask)

        if  not linkpred:
            return  out, out_adj, None, ent_loss, balanced_loss
        else:
            # pdb.set_trace()

            link_loss = adj - torch.matmul(s, s.transpose(1, 2))
            link_loss = torch.norm(link_loss, p=2,dim=(1,2))
            link_loss = link_loss / entries #adj.numel()
            return out, out_adj, link_loss, ent_loss, balanced_loss

    def _diff_pool(self, x, adj, s, mask, linkpred):
        balanced_loss = None
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        # pdb.set_trace()
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = torch.sum(mask, 1)[:,0]
            entries = torch.pow(ent_entries, 2)
        else:
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = torch.tensor([adj.shape[1]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
            entries = torch.tensor([adj.numel()/adj.shape[0]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
        # pdb.set_trace()
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        ent_loss = torch.div((-s * torch.log(s + EPS)).sum(dim=-1).mean(-1),ent_entries)
        if self.balanced_cluster:
            balanced_loss = balanced_cluster_loss(s, mask)

        if  not linkpred:
            return  out, out_adj, None, ent_loss, balanced_loss
        else:
            # pdb.set_trace()
            pred_adj = torch.matmul(s, s.transpose(1, 2))
            pred_adj =torch.clamp_max(pred_adj,1.)
            link_loss = -adj * torch.log(pred_adj+ EPS) - (1-adj) * torch.log(1-pred_adj+EPS)


            link_loss = link_loss.sum((1,2)) / entries #adj.numel()

            return out, out_adj, link_loss, ent_loss, balanced_loss

    def _activation(self, name = 'relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name =='leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def build_readout_module(self,pred_input_dim, pred_hidden_dims, label_dim, activation ):
        pred_input_dim = pred_input_dim
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self._activation(activation))
                pred_input_dim = pred_dim
                if self.drop_out>0:
                    pred_layers.append(nn.Dropout(self.drop_out))
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model


    def _sparse_to_dense_input(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        label = data.y
        edge_index = to_dense_adj(edge_index, batch)
        x ,batch_num_node= to_dense_batch(x, batch)
        return x, edge_index,batch_num_node,label

    def forward(self,  data):

        # def forward(self, x, adj, batch_num_nodes, label=None):
        out_all = []
        mean_all = []
        self.assign_matrix = []
        # stage 1
        if self.load_data_sparse:
            # if self.task =='ICIAR':
            #     x, adj, batch_num_nodes, label, img = self._sparse_to_dense_input(data)
            # else:
                x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            # if self.task == 'ICIAR':

            x, adj, batch_num_nodes= data[0], data[1], data[2]
            if self.training:
                label = data[3]

        # if
        #     img_feature = self.resnet18(img)

        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        # pdb.set_trace()
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4, embedding_mask)
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)
        # pdb.set_trace()
        if self.jk:
            embed_feature = self.jk1(embed_feature)
        out, _ = torch.max(embed_feature, dim = 1)
        out_all.append(out)

        if self.readout == 'mix':
            mean= torch.mean(embed_feature, dim = 1)
            mean_all.append(mean)
        assign = self.GCN_pool_1(x, adj, embedding_mask)
        x, adj, link_loss1, ent_loss1, b_loss1 = self._diff_pool(embed_feature, adj, assign, embedding_mask, self.linkpred)
        # stage 2
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_2(x, adj, None)
        if self.jk:
            embed_feature = self.jk2(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        if self.readout == 'mix':
            mean = torch.mean(embed_feature, dim = 1)
            mean_all.append(mean)
        assign = self.GCN_pool_2(x, adj, None)
        x, adj, link_loss2, ent_loss2, b_loss2 = self._diff_pool(embed_feature, adj, assign, None, self.linkpred)

        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_3(x, adj, None)
        if self.jk:
            embed_feature = self.jk3(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        if self.readout == 'mix':
            mean = torch.mean(embed_feature, dim = 1)
            mean_all.append(mean)
        output = torch.cat(out_all, 1)
        if self.readout == 'mix':
            mean_all = torch.cat(mean_all,1)
            output = torch.cat([mean_all,output], 1)
        # pdb.set_trace()
        output = self.pred_model(output)
        if self.training:
            if self.linkpred:
                link_loss = link_loss1 + link_loss2
            else:
                link_loss = torch.zeros((1)).cuda()
            ent_loss = ent_loss1 + ent_loss2
            if self.balanced_cluster:
                b_loss = b_loss1 + b_loss2
            else:
                b_loss =torch.zeros((1)).cuda()

            cls_loss = F.cross_entropy(output, label, size_average=True)
            return output, link_loss, ent_loss, cls_loss, b_loss
        return output

class SoftPoolingGcnEncoderJK(nn.Module):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, bias, bn, assign_hidden_dim,label_dim,
                 assign_ratio=0.25,  pred_hidden_dims=[50], linkpred=True, concat = True, gcn_name='SAGE',
                 collect_assign = False, balanced_cluster = False, load_data_sparse = False,norm_adj=False,
                 activation = 'relu', readout = 'max', drop_out = 0., noise = False,jk = False, task = 'colon'):


        super(SoftPoolingGcnEncoderJK, self).__init__()

        if task == 'ICIAR':
            self.resnet18 = resnet18(True, num_classes=4)
        self.jk = jk
        self.drop_out = drop_out
        self.norm_adj = norm_adj
        self.readout = readout
        self.load_data_sparse = load_data_sparse
        self.balanced_cluster = balanced_cluster
        self.collect_assign = collect_assign
        self.assign_matrix = []
        assign_dim = int(max_num_nodes * assign_ratio)
        self.linkpred = linkpred
        self.GCN_embed_1 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False, lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk1 = DenseJK('lstm',  hidden_dim, 3)
        self.GCN_pool_1 = GNN_Module(input_dim, assign_hidden_dim, assign_hidden_dim, bias, bn,
                                     add_loop= False, lin=False,gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.pool_jk1 =  DenseJK('lstm',  assign_hidden_dim, 3)
            self.lin1 = nn.Linear(assign_hidden_dim, assign_dim)
        else:
            self.lin1 = nn.Linear( 3* assign_hidden_dim , assign_dim)
        if concat and not jk:
            input_dim = hidden_dim * 2 + embedding_dim
        else:
            input_dim = embedding_dim

        assign_dim = int(assign_dim * assign_ratio)
        self.GCN_embed_2 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk2 = DenseJK('lstm', hidden_dim , 3)
        self.GCN_pool_2 = GNN_Module(input_dim, assign_hidden_dim, assign_hidden_dim, bias, bn,
                                     add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.pool_jk2 =  DenseJK('lstm',  assign_hidden_dim,assign_hidden_dim, 3)
            self.lin2 = nn.Linear(assign_hidden_dim, assign_dim)
        else:
            self.lin2 = nn.Linear(3 * assign_hidden_dim, assign_dim)
        self.GCN_embed_3 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk3 = DenseJK('lstm', hidden_dim, 3)

        if readout =='max':
            pred_input = input_dim * 3
        else :
            pred_input = input_dim * 6

        self.pred_model = self.build_readout_module(pred_input, pred_hidden_dims,
                                                    label_dim, activation)


    @staticmethod
    def construct_mask( max_nodes, batch_num_nodes):
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()


    def _re_norm_adj(self,adj,p, mask = None):
        # pdb.set_trace()
        idx = torch.arange(0, adj.shape[1],out=torch.LongTensor())
        adj[:,idx,idx] = 0
        new_adj =  torch.div(adj,adj.sum(-1)[...,None] + EPS)*(1-p)
        new_adj[:,idx,idx] = p
        if mask is not None:
            new_adj = new_adj * mask
        return new_adj

    def _diff_pool1(self, x, adj, s, mask, linkpred):
        balanced_loss = None
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
            # ent_entries = 1
            # entries = adj.numel()
            # pdb.set_trace()
            ent_entries = torch.sum(mask, 1)
            entries = torch.sum(torch.pow(ent_entries, 2))
        else:
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = adj.shape[1]
            entries = adj.numel()
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean() / ent_entries
        if self.balanced_cluster:
            balanced_loss = balanced_cluster_loss(s, mask)

        if not linkpred:
            return out, out_adj, None, ent_loss, balanced_loss
        else:
            # pdb.set_trace()

            link_loss = adj - torch.matmul(s, s.transpose(1, 2))
            link_loss = torch.norm(link_loss, p=2)

            link_loss = link_loss / entries  # adj.numel()

            return out, out_adj, link_loss, ent_loss, balanced_loss

    def _diff_pool2(self, x, adj, s, mask, linkpred):
        balanced_loss = None
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        # pdb.set_trace()
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
            # ent_entries = 1
            # entries = adj.numel()

            ent_entries = torch.sum(mask, 1)[:,0]
            entries = torch.pow(ent_entries, 2)
        else:
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = torch.tensor([adj.shape[1]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
            entries = torch.tensor([adj.numel()/adj.shape[0]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
        # pdb.set_trace()
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        ent_loss = torch.div((-s * torch.log(s + EPS)).sum(dim=-1).mean(-1),ent_entries)
        if self.balanced_cluster:
            balanced_loss = balanced_cluster_loss(s, mask)

        if  not linkpred:
            return  out, out_adj, None, ent_loss, balanced_loss
        else:
            # pdb.set_trace()

            link_loss = adj - torch.matmul(s, s.transpose(1, 2))
            link_loss = torch.norm(link_loss, p=2,dim=(1,2))
            link_loss = link_loss / entries #adj.numel()
            return out, out_adj, link_loss, ent_loss, balanced_loss

    def _diff_pool(self, x, adj, s, mask, linkpred):
        balanced_loss = None
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        # pdb.set_trace()
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = torch.sum(mask, 1)[:,0]
            entries = torch.pow(ent_entries, 2)
        else:
            # ent_entries = 1
            # entries = adj.numel()
            ent_entries = torch.tensor([adj.shape[1]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
            entries = torch.tensor([adj.numel()/adj.shape[0]],dtype = s.dtype).to(s.device).repeat(adj.shape[0])
        # pdb.set_trace()
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        ent_loss = torch.div((-s * torch.log(s + EPS)).sum(dim=-1).mean(-1),ent_entries)
        if self.balanced_cluster:
            balanced_loss = balanced_cluster_loss(s, mask)

        if  not linkpred:
            return  out, out_adj, None, ent_loss, balanced_loss
        else:
            # pdb.set_trace()
            pred_adj = torch.matmul(s, s.transpose(1, 2))
            pred_adj =torch.clamp_max(pred_adj,1.)
            link_loss = -adj * torch.log(pred_adj+ EPS) - (1-adj) * torch.log(1-pred_adj+EPS)


            link_loss = link_loss.sum((1,2)) / entries #adj.numel()

            return out, out_adj, link_loss, ent_loss, balanced_loss

    def _activation(self, name = 'relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name =='leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def build_readout_module(self,pred_input_dim, pred_hidden_dims, label_dim, activation ):
        pred_input_dim = pred_input_dim
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self._activation(activation))
                pred_input_dim = pred_dim
                if self.drop_out>0:
                    pred_layers.append(nn.Dropout(self.drop_out))
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model


    def _sparse_to_dense_input(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        label = data.y
        edge_index = to_dense_adj(edge_index, batch)
        x ,batch_num_node= to_dense_batch(x, batch)
        return x, edge_index,batch_num_node,label

    def forward(self,  data):

        # def forward(self, x, adj, batch_num_nodes, label=None):
        out_all = []
        mean_all = []
        self.assign_matrix = []
        # stage 1
        if self.load_data_sparse:
            # if self.task =='ICIAR':
            #     x, adj, batch_num_nodes, label, img = self._sparse_to_dense_input(data)
            # else:
                x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            # if self.task == 'ICIAR':

            x, adj, batch_num_nodes= data[0], data[1], data[2]
            if self.training:
                label = data[3]

        # if
        #     img_feature = self.resnet18(img)

        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        # pdb.set_trace()
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4, embedding_mask)
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)
        # pdb.set_trace()
        if self.jk:
            embed_feature = self.jk1(embed_feature)
        out, _ = torch.max(embed_feature, dim = 1)
        out_all.append(out)

        if self.readout == 'mix':
            mean= torch.mean(embed_feature, dim = 1)
            mean_all.append(mean)
        assign = self.GCN_pool_1(x, adj, embedding_mask)
        if self.jk:
            assign = self.pool_jk1(assign)
        assign = self.lin1(assign)

        x, adj, link_loss1, ent_loss1, b_loss1 = self._diff_pool(embed_feature, adj, assign, embedding_mask, self.linkpred)
        # stage 2
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_2(x, adj, None)
        if self.jk:
            embed_feature = self.jk2(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        if self.readout == 'mix':
            mean = torch.mean(embed_feature, dim = 1)
            mean_all.append(mean)
        assign = self.GCN_pool_2(x, adj, None)
        if self.jk:
            assign = self.pool_jk2(assign)
        assign = self.lin2(assign)
        x, adj, link_loss2, ent_loss2, b_loss2 = self._diff_pool(embed_feature, adj, assign, None, self.linkpred)

        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_3(x, adj, None)
        if self.jk:
            embed_feature = self.jk3(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        if self.readout == 'mix':
            mean = torch.mean(embed_feature, dim = 1)
            mean_all.append(mean)
        output = torch.cat(out_all, 1)
        if self.readout == 'mix':
            mean_all = torch.cat(mean_all,1)
            output = torch.cat([mean_all,output], 1)
        # pdb.set_trace()
        output = self.pred_model(output)
        if self.training:
            if self.linkpred:
                link_loss = link_loss1 + link_loss2
            else:
                link_loss = torch.zeros((1)).cuda()
            ent_loss = ent_loss1 + ent_loss2
            if self.balanced_cluster:
                b_loss = b_loss1 + b_loss2
            else:
                b_loss =torch.zeros((1)).cuda()

            cls_loss = F.cross_entropy(output, label, size_average=True)
            return output, link_loss, ent_loss, cls_loss, b_loss
        return output


class DeeperSoftPoolingGcnEncoder(SoftPoolingGcnEncoder):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, bias, bn, assign_hidden_dim,label_dim,
                 assign_ratio=0.25,  pred_hidden_dims=[50], linkpred=True, concat = True, gcn_name='SAGE',
                 collect_assign = False, balanced_cluster = False, load_data_sparse = False, norm_adj = True, GNN = 'SGNN',
                 activation ='relu', readout = 'max',drop_out = 0.):
        super(DeeperSoftPoolingGcnEncoder, self).__init__( max_num_nodes, input_dim, hidden_dim, embedding_dim, bias,
                                                           bn, assign_hidden_dim,label_dim,
                                                           assign_ratio,  pred_hidden_dims, linkpred, concat , gcn_name,
                                                           collect_assign , balanced_cluster , load_data_sparse, norm_adj,
                                                           activation, readout, drop_out= drop_out)

        assign_dim = 500
        self.drop_out = drop_out
        self.readout = readout
        self.GNN =GNN
        self.norm_adj = norm_adj
        self.linkpred = linkpred
        pred_input = 0
        self.GCN_embed_1 = self._gnn(input_dim, hidden_dim, embedding_dim, bias, bn,
                                     add_loop= False, lin=False, gcn_name=gcn_name, activation=activation)
        self.GCN_pool_1 = self._gnn(input_dim, assign_hidden_dim, assign_dim, bias, bn,
                                    add_loop= False, gcn_name=gcn_name, activation=activation)
        if concat:
            if self.GNN =='GNN':
                input_dim = hidden_dim * 2 + embedding_dim
            else:
                input_dim = hidden_dim + embedding_dim
        else:
            input_dim = embedding_dim

        pred_input +=input_dim
        assign_dim = 125
        self.GCN_embed_2 = self._gnn(input_dim, hidden_dim * 2, embedding_dim*2, bias, bn,
                                     add_loop= False,lin=False, gcn_name=gcn_name, activation=activation)
        self.GCN_pool_2 = self._gnn(input_dim, assign_hidden_dim*2, assign_dim, bias, bn,
                                    add_loop= False, gcn_name=gcn_name, activation=activation)

        if concat:
            if self.GNN == 'GNN':
                input_dim = hidden_dim * 4 + embedding_dim*2
            else:
                input_dim = hidden_dim * 2 + embedding_dim *2
        else:
            input_dim = embedding_dim*2
        pred_input += input_dim
        assign_dim = 30
        self.GCN_embed_3 = self._gnn(input_dim, hidden_dim*4, embedding_dim*4, bias, bn,
                                     add_loop= False,lin=False, gcn_name=gcn_name, activation=activation)
        self.GCN_pool_3 = self._gnn(input_dim, assign_hidden_dim * 4, assign_dim, bias, bn,
                                    add_loop=False, gcn_name=gcn_name, activation=activation)

        if concat:
            if self.GNN == 'GNN':
                input_dim = hidden_dim * 8+ embedding_dim*4
            else:
                input_dim = hidden_dim * 4 + embedding_dim * 4
        else:
            input_dim = embedding_dim*4

        pred_input += input_dim
        self.GCN_embed_4 = self._gnn(input_dim, hidden_dim * 8, embedding_dim * 8, bias, bn,
                                     add_loop=False, lin=False, gcn_name=gcn_name, activation = activation)
        if self.GNN == 'GNN':
            pred_input += hidden_dim * 16 + embedding_dim * 8
        else:
            pred_input += hidden_dim * 8 + embedding_dim * 8

        if readout !='max':
            pred_input = int(2 * pred_input)
        self.pred_model = self.build_readout_module(pred_input, pred_hidden_dims,
                                                    label_dim, activation = activation)

    def _gnn(self, input_dim, hidden_dim, embedding_dim, bias, bn, add_loop= False, lin=False,
             gcn_name = 'SAGE', activation='relu'):
        if self.GNN == 'GNN':
            return GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                              add_loop=add_loop, lin=lin, gcn_name=gcn_name, activation=activation)
        else:
            return SGNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                               add_loop=add_loop, lin=lin, gcn_name=gcn_name, activation=activation)

    def forward(self,  data):
        # def forward(self, x, adj, batch_num_nodes, label=None):
        out_all = []
        mean_all = []
        self.assign_matrix = []
        # stage 1
        if self.load_data_sparse:
            x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            x, adj, batch_num_nodes= data[0], data[1], data[2]



        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        if self.norm_adj:
            adj = self._re_norm_adj(adj,0.4,embedding_mask)
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)
        out, _ = torch.max(embed_feature, dim = 1)
        out_all.append(out)
        if self.readout == 'mix':
            mean, _ = torch.mean(embed_feature,dim=1)
            mean_all.append(mean)
        assign = self.GCN_pool_1(x, adj, embedding_mask)
        x, adj, link_loss1, ent_loss1, b_loss1 = self._diff_pool(embed_feature, adj, assign, embedding_mask, self.linkpred)
        # stage 2
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_2(x, adj, None)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        if self.readout == 'mix':
            mean, _ = torch.mean(embed_feature,dim=1)
            mean_all.append(mean)
        assign = self.GCN_pool_2(x, adj, None)
        x, adj, link_loss2, ent_loss2, b_loss2 = self._diff_pool(embed_feature, adj, assign, None, self.linkpred)

        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_3(x, adj, None)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        if self.readout == 'mix':
            mean, _ = torch.mean(embed_feature,dim=1)
            mean_all.append(mean)
        assign = self.GCN_pool_3(x, adj, None)
        x, adj, link_loss3, ent_loss3, b_loss3 = self._diff_pool(embed_feature, adj, assign, None, self.linkpred)

        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_4(x, adj, None)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        if self.readout == 'mix':
            mean, _ = torch.mean(embed_feature,dim=1)
            mean_all.append(mean)

        output = torch.cat(out_all, 1)
        if self.readout == 'mix':
            output = torch.cat([mean_all,output], dim=1)
        output = self.pred_model(output)
        if self.training:
            if not self.load_data_sparse:
                label = data[3]
            if self.linkpred:
                link_loss = link_loss1 + link_loss2 + link_loss3
            else:
                link_loss = torch.zeros((1)).cuda()
            ent_loss = ent_loss1 + ent_loss2 + ent_loss3
            if self.balanced_cluster:
                b_loss = b_loss1 + b_loss2 + b_loss3
            else:
                b_loss =torch.zeros((1)).cuda()

            cls_loss = F.cross_entropy(output, label, size_average=True)
            return output, link_loss, ent_loss, cls_loss, b_loss
        return output

class SelfAttentionGraphNetwork(nn.Module):
    def __init__(self,  input_dim, hidden_dim, embedding_dim, bias, bn, label_dim,
                 ratio, pred_hidden_dims=[50], name = 'SAGE'):
        super(SelfAttentionGraphNetwork, self).__init__()
        self.bn = bn
        self.GCN_1_1 = self._gcn(name, input_dim, hidden_dim, normalize = True, bias = bias)#SAGEConv(input_dim,  hidden_dim, normalize=True, bias = True)
        self.GCN_1_2 = self._gcn(name, hidden_dim, hidden_dim, normalize = True, bias = bias)#SAGEConv( hidden_dim,  hidden_dim, normalize=True, bias = True)
        self.GCN_1_3 = self._gcn(name, hidden_dim, embedding_dim, normalize = True, bias = bias)#SAGEConv( hidden_dim, embedding_dim, normalize=True, bias= True)
        self.sag_pool_1 = SAGPooling(embedding_dim, ratio=ratio, gnn=name)
        self.GCN_2_1 = self._gcn(name, embedding_dim, hidden_dim, normalize = True, bias = bias)#SAGEConv(embedding_dim, hidden_dim, normalize=True, bias = True)
        self.GCN_2_2 = self._gcn(name, hidden_dim, hidden_dim, normalize = True, bias = bias)#SAGEConv( hidden_dim,  hidden_dim, normalize=True, bias = True)
        self.GCN_2_3 = self._gcn(name, hidden_dim, embedding_dim, normalize = True, bias = bias)#SAGEConv(hidden_dim, embedding_dim, normalize= True, bias=True)
        self.sag_pool_2 = SAGPooling(embedding_dim, ratio=ratio, gnn=name)
        self.GCN_3_1 = self._gcn(name, embedding_dim, hidden_dim, normalize = True, bias = bias)#SAGEConv(embedding_dim, hidden_dim, normalize=True, bias = True)
        self.GCN_3_2 = self._gcn(name, hidden_dim, hidden_dim, normalize = True, bias = bias)#SAGEConv( hidden_dim,  hidden_dim, normalize=True, bias = True)
        self.GCN_3_3 = self._gcn(name, hidden_dim, embedding_dim, normalize = True, bias = bias)#SAGEConv(hidden_dim, embedding_dim, normalize= True, bias=True)

        if bn:
            self.bn_1_1 = nn.BatchNorm1d(hidden_dim)
            self.bn_1_2 = nn.BatchNorm1d(hidden_dim)
            self.bn_1_3 = nn.BatchNorm1d(embedding_dim)
            self.bn_2_1 = nn.BatchNorm1d(hidden_dim)
            self.bn_2_2 = nn.BatchNorm1d(hidden_dim)
            self.bn_2_3 = nn.BatchNorm1d(embedding_dim)
            self.bn_3_1 = nn.BatchNorm1d(hidden_dim)
            self.bn_3_2 = nn.BatchNorm1d(hidden_dim)
            self.bn_3_3 = nn.BatchNorm1d(embedding_dim)

        self.pred_module = self.build_readout_module(embedding_dim*3, pred_hidden_dims, label_dim)

    def _gcn(self, name, input_dim, hidden_dim, normalize, bias, **kwargs):
        if name == 'SAGE':
            return SAGEConv(input_dim,  hidden_dim, normalize, bias)
        elif name == 'GIN':
            nn1 =  nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            return GINConv(nn1)
        elif name == 'GAT':
            return GATConv(input_dim, hidden_dim, **kwargs)
        else:
            return GraphConv(input_dim, hidden_dim, **kwargs)

    def build_readout_module(self,pred_input_dim, pred_hidden_dims, label_dim, ):
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(nn.ReLU(True))
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def forward(self,  data):

        self.perms = []
        x, edge_index, batch = data.x, data.edge_index, data.batch
        out_all = []
        x = F.relu(self.GCN_1_1.forward(x,edge_index))
        if self.bn:
            x = self.bn_1_1(x)
        x = F.relu(self.GCN_1_2.forward(x, edge_index))
        if self.bn:
            x = self.bn_1_2(x)
        x = F.relu(self.GCN_1_3.forward(x,edge_index))
        if self.bn:
            x =self.bn_1_3(x)
        out, _ = scatter_max(x,batch,dim=0)
        out_all.append(out)
        x, edge_index, edge_attr, batch, perm = self.sag_pool_1.forward(x, edge_index, batch = batch)
        self.perms.append(perm.detach())
        x = F.relu(self.GCN_2_1.forward(x,edge_index))
        if self.bn:
            x = self.bn_2_1(x)
        x = F.relu(self.GCN_2_2.forward(x, edge_index))
        if self.bn:
            x = self.bn_2_2(x)
        x = F.relu(self.GCN_2_3.forward(x,edge_index))
        if self.bn:
            x = self.bn_2_3(x)
        out, _ = scatter_max(x,batch,dim=0)
        out_all.append(out)
        x, edge_index, edge_attr, batch, perm = self.sag_pool_2.forward(x, edge_index, batch=batch)
        self.perms.append(perm.detach())
        x = F.relu(self.GCN_3_1.forward(x,edge_index))
        if self.bn:
            x = self.bn_3_1(x)
        x = F.relu(self.GCN_3_2.forward(x, edge_index))
        if self.bn:
            x = self.bn_3_1(x)
        x = F.relu(self.GCN_3_3.forward(x,edge_index))
        if self.bn:
            x = self.bn_3_3(x)
        out, _ = scatter_max(x,batch,dim=0)
        out_all.append(out)
        out = torch.cat(out_all, 1)
        out = self.pred_module(out)
        if self.training:
            loss = F.cross_entropy(out, data.y, size_average=True)
            return loss
        return out

    def get_perm(self):
        return self.perms

class DeepSelfAttentionGraphNetwork(SelfAttentionGraphNetwork):
    def __init__(self,  input_dim, hidden_dim, embedding_dim, bias, bn, label_dim,
                 ratio, pred_hidden_dims=[50], name = 'SAGE'):
        super(DeepSelfAttentionGraphNetwork, self).__init__( input_dim, hidden_dim, embedding_dim, bias, bn, label_dim,
                 ratio, pred_hidden_dims, name )
        self.sag_pool_3 = SAGPooling(embedding_dim, ratio=ratio, gnn=name)

    def forward(self,  data):
        self.perms = []
        x, edge_index, batch = data.x, data.edge_index, data.batch
        out_all = []
        x = F.relu(self.GCN_1_1.forward(x,edge_index))
        if self.bn:
            x = self.bn_1_1(x)
        x = F.relu(self.GCN_1_2.forward(x, edge_index))
        if self.bn:
            x = self.bn_1_2(x)
        x = F.relu(self.GCN_1_3.forward(x,edge_index))
        if self.bn:
            x =self.bn_1_3(x)
        out, _ = scatter_max(x,batch,dim=0)
        out_all.append(out)
        x, edge_index, edge_attr, batch, perm = self.sag_pool_1.forward(x, edge_index, batch = batch)
        self.perms.append(perm.detach())
        x = F.relu(self.GCN_2_1.forward(x,edge_index))
        if self.bn:
            x = self.bn_2_1(x)
        x = F.relu(self.GCN_2_2.forward(x, edge_index))
        if self.bn:
            x = self.bn_2_2(x)
        x = F.relu(self.GCN_2_3.forward(x,edge_index))
        if self.bn:
            x = self.bn_2_3(x)
        out, _ = scatter_max(x,batch,dim=0)
        out_all.append(out)
        x, edge_index, edge_attr, batch, perm = self.sag_pool_2.forward(x, edge_index, batch=batch)
        self.perms.append(perm.detach())
        x = F.relu(self.GCN_3_1.forward(x,edge_index))
        if self.bn:
            x = self.bn_3_1(x)
        x = F.relu(self.GCN_3_2.forward(x, edge_index))
        if self.bn:
            x = self.bn_3_1(x)
        x = F.relu(self.GCN_3_3.forward(x,edge_index))
        if self.bn:
            x = self.bn_3_3(x)
        x, edge_index, edge_attr, batch, perm = self.sag_pool_3.forward(x, edge_index, batch=batch)
        self.perms.append(perm.detach())
        out, _ = scatter_max(x,batch,dim=0)
        out_all.append(out)
        out = torch.cat(out_all, 1)
        out = self.pred_module(out)
        if self.training:
            loss = F.cross_entropy(out, data.y, size_average=True)
            return loss
        return out

class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_  paper
    .. math::
        \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})
        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})
        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}
        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},
    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.
    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            (default: :obj:`0.5`)
        gnn (string, optional): Specifies which graph neural network layer to
            use for calculating projection scores (one of
            :obj:`"GCN"`, :obj:`"GAT"` or :obj:`"SAGE"`). (default: :obj:`GCN`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """

    def __init__(self, in_channels, ratio=0.5, gnn='GCN', **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn_name = gnn

        assert gnn in ['GCN', 'GAT', 'SAGE', 'GIN']
        if gnn == 'GCN':
            self.gnn = GraphConv(self.in_channels, 1, **kwargs)
        elif gnn == 'GAT':
            self.gnn = GATConv(self.in_channels, 1, **kwargs)
        elif gnn == 'GIN':
            nn1 = nn.Sequential(nn.Linear(self.in_channels, self.in_channels), nn.ReLU(), nn.Linear(self.in_channels, 1))
            self.gnn = GINConv(nn1)
        else:
            self.gnn = SAGEConv(self.in_channels, 1, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        score = torch.tanh(self.gnn(x, edge_index).view(-1))
        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

    def __repr__(self):
        return '{}({}, {}, ratio={})'.format(self.__class__.__name__,
                                             self.gnn_name, self.in_channels,
                                             self.ratio)

# Help functions #
