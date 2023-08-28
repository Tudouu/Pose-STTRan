import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN.graphy import Graph
from GCN.tgcn import ConvTemporalGraphical
from GCN.GCn import Gcn
import numpy as np
import operator



class Model(nn.Module):

    def __init__(self, in_channels,graph_args,edge_importance_weighting):
        super(Model,self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)#(3,18,18)
        self.register_buffer('A', A)
        self.dev='cuda:0'
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))#3*18
        self.gcn_networks = nn.ModuleList((# in_channel=3 kernel_size=(9,3)
            Gcn(in_channels, 64, kernel_size, 1, residual=False),#(3,64,(9,3),1)
            Gcn(64,64,kernel_size, 1),
            Gcn(64, 64, kernel_size, 1),
            Gcn(64, 128, kernel_size, 1),
            Gcn(128,128, kernel_size, 1),
            Gcn(128, 128, kernel_size, 1)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.gcn_networks
            ])
            self.edge_importance=self.edge_importance.cuda()
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
            self.edge_importance = self.edge_importance.cuda()

        # fcn for prediction
        #print("numclass:",num_class)   400
        #self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, fre_ske_idx):
        i=0
        t=[]
        _,__,rel_num=x.shape
        x=x.float().to(self.dev)
        x=self.data_bn(x)
        x=x.view(1,18,3,rel_num)#
        x=x.permute(0,2,3,1).contiguous()#[1,3,rel_num,18]

        # forwad
        for gcn, importance in zip(self.gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        feature=x.permute(2,1,3,0).cpu().detach().numpy().tolist()#[num,128,18,1]

        for ll in range(len(fre_ske_idx)):
            for k in range(fre_ske_idx[ll]):
                t.append(feature[i])
            i=i+1

        t=torch.from_numpy(np.array(t)).cuda()

        return t

