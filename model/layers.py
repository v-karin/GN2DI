# src/models/pre_weight.py
import torch
from torch import nn

class PreWeight(nn.Module):
    def __init__(self, num_weights, hidden_dim, num_lay, dropout_pre_weight=0.2):
        super(PreWeight, self).__init__()

        self.mlp = nn.Sequential()
        
        for idx in range(num_lay-1):
            if idx != 0:
                num_weights = hidden_dim
                
            temp = nn.Sequential(
                nn.Linear(num_weights, hidden_dim),
                nn.Dropout(p=dropout_pre_weight),
                nn.ReLU()
            )
            
            self.mlp = nn.Sequential(*(list(self.mlp)+list(temp))) 

        temp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(*(list(self.mlp)+list(temp))) 

    def forward(self, w):
        return self.mlp(w)

# src/models/graph_learner.py
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import GraphConv

class GraphLearner(nn.Module):
    def __init__(self,
                 in_channel,
                 num_conv_lay,
                 hidden_dim_conv,
                 hidden_dim_readout, 
                 hidden_dim_updater,
                 dropout_gl=0.2):
        super(GraphLearner, self).__init__()
        
        self.hidden_dim_readout = hidden_dim_readout
        self.conv = []
        self.read_out = []
        self.weight_updater = []
        self.dropout_gl = dropout_gl

        for idx in range(num_conv_lay):
            if idx != 0:
                in_channel = hidden_dim_conv

            self.conv.append(
                GraphConv(
                    in_channels=in_channel,
                    out_channels=hidden_dim_conv,
                    aggr='mean'
                ).to(device="cuda")
            )
            
            self.read_out.append(
                nn.Sequential(
                    nn.Linear(hidden_dim_conv, hidden_dim_readout),
                    nn.Dropout(p=dropout_gl),
                    nn.ReLU()
                ).to(device="cuda")
            )

            self.weight_updater.append(
                nn.Sequential(
                    nn.Linear(hidden_dim_readout*3, hidden_dim_updater),
                    nn.Dropout(p=dropout_gl),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_updater, 1),
                    nn.Sigmoid()
                ).to(device="cuda")
            )
    
    def update_edge_weight(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i, x_j, edge_attr.view(-1,1).repeat(1,self.hidden_dim_readout)), dim=-1))
        return edge_attr

    def forward(self, static_node_features, edge_index, edge_weight):
        h = static_node_features
        for cnv, read_out, upd in zip(self.conv, self.read_out, self.weight_updater):
            h = F.relu(F.dropout(read_out(cnv(h, edge_index, edge_weight))))
            edge_weights = self.update_edge_weight(h, edge_weight, edge_index, upd)
        
        return edge_weight

# src/models/imputation.py
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import GraphConv

class Imputation(nn.Module):
    def __init__(self, 
                 in_channel,
                 num_conv_lay,
                 hidden_dim_conv,
                 hidden_dim_readout, 
                 dropout_imp=0.2):
        super(Imputation, self).__init__()
        self.conv = []
        self.read_out = []
        for idx in range(num_conv_lay):
            if idx != 0:
                in_channel = hidden_dim_conv

            self.conv.append(
                GraphConv(
                    in_channels=in_channel,
                    out_channels=hidden_dim_conv,
                    aggr='mean'
                )
            )
            
            self.read_out.append(
                nn.Sequential(
                    nn.Linear(hidden_dim_conv, hidden_dim_readout),
                    nn.Dropout(p=dropout_imp),
                    nn.ReLU(),
                )
            )

        self.prd = nn.Linear(hidden_dim_readout, 1)

    def forward(self, dynamic_node_features, edge_index, edge_weight):
        h = dynamic_node_features
        for cnv, read_out in zip(self.conv, self.read_out):
            h = F.relu(F.dropout(cnv(h[:, 0, :], edge_index, edge_weight)[:, None, :]))        
            h = read_out(h)
    
        x_hat = self.prd(h)
        return x_hat
