import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv,DenseGINConv
import torch
from torch_geometric.utils import to_dense_batch
from model.utils import to_dense_adj
from torch.nn import Linear, LSTM
EPS = 1e-15
import pdb

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
        if mask is not None:
            x = x * mask
        if self.lin is not None:
            x = self.lin(x)
            if mask is not None:
                x = x * mask
        return x

class SoftPoolingGcnEncoder(nn.Module):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, bias, bn, assign_hidden_dim,label_dim,
                 assign_ratio=0.25,  pred_hidden_dims=[50], concat = True, gcn_name='SAGE',
                 collect_assign = False, load_data_sparse = False,norm_adj=False,
                 activation = 'relu',  drop_out = 0.,jk = False):


        super(SoftPoolingGcnEncoder, self).__init__()

        self.jk = jk
        self.drop_out = drop_out
        self.norm_adj = norm_adj
        self.load_data_sparse = load_data_sparse
        self.collect_assign = collect_assign
        self.assign_matrix = []
        assign_dim = int(max_num_nodes * assign_ratio)
        assign_dim = int(max_num_nodes * assign_ratio)
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
        pred_input = input_dim * 3
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


    def _diff_pool(self, x, adj, s, mask):
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
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        return  out, out_adj


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
        if self.load_data_sparse:
                x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            x, adj, batch_num_nodes= data[0], data[1], data[2]
            if self.training:
                label = data[3]
        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4, embedding_mask)
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)
        if self.jk:
            embed_feature = self.jk1(embed_feature)
        out, _ = torch.max(embed_feature, dim = 1)
        out_all.append(out)

        assign = self.GCN_pool_1(x, adj, embedding_mask)
        x, adj = self._diff_pool(embed_feature, adj, assign, embedding_mask)
        # stage 2
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_2(x, adj, None)
        if self.jk:
            embed_feature = self.jk2(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        assign = self.GCN_pool_2(x, adj, None)
        x, adj = self._diff_pool(embed_feature, adj, assign, None)
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_3(x, adj, None)
        if self.jk:
            embed_feature = self.jk3(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        output = torch.cat(out_all, 1)
        output = self.pred_model(output)
        if self.training:
            cls_loss = F.cross_entropy(output, label, size_average=True)
            return output, cls_loss
        return output
