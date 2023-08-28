import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN.graphy import Graph
from GCN.tgcn import ConvTemporalGraphical




class Model(nn.Module):

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        #print('afc',**kwargs)
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)#(3,18,18)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))#3*18
        self.data_bn.cuda()
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.gcn_networks = nn.ModuleList((# in_channel=3 kernel_size=(9,3)
            Gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),#(3,64,(9,3),1)
            Gcn(64, 128, kernel_size, 1, **kwargs),
            Gcn(128, 128, kernel_size, 1, **kwargs),
            Gcn(128, 256, kernel_size, 1, **kwargs),
            Gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        self.gcn_networks.cuda()

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

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        #[1,1,图数量,num_joint(从第一个关节点开始),x/y/score]
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        #print(x.shape)
        #[1, 3, 图片数量, 18]

        # forwad
        for gcn, importance in zip(self.gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()

        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)  # [1, 256, 85, 18, 1]

        # prediction
        x = self.fcn(x)
        # print('xshape:',x.shape)  [400, 85, 18, 1]
        #一共26组
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)  # [1, 400, 85, 18, 1] TODO 真正有用的output

        return output,feature




class Gcn(nn.Module):

    def __init__(self,
                 in_channels,#3
                 out_channels,#64
                 kernel_size,#(9,3)
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        #padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        #print('gcnxsssssss',x.shape)
        #print('Assssssss',A.shape)
        #x = self.tcn(x) + res
        #print('tcnxsssssss', x.shape)
        #x=x+res

        return self.relu(x), A