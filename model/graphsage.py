import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from setting import  ModelSetting

class SupervisedGraphSage(nn.Module):
    ''' GraphSage embeddings
    '''

    def __init__(self, enc):
        super(SupervisedGraphSage, self).__init__()
        setting =  ModelSetting()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(enc.embed_dim, setting.graphsage_num_class))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = embeds.mm(self.weight)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(nn.softmax(scores), labels.squeeze())

