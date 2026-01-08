import torch
from torch import nn

from .layers import PreWeight, GraphLearner,Imputation

class GN2DI(nn.Module):
    def __init__(self, 
                 num_weights, 
                 hidden_dim_pre_weight, 
                 num_lay_pre_weight, 
                 in_channel_gl,
                 num_conv_lay_gl,
                 hidden_dim_conv_gl,
                 hidden_dim_gl, 
                 in_channel_imp,
                 num_conv_lay_imp,
                 hidden_dim_conv_imp,
                 hidden_dim_readout_imp,
                 hidden_dim_updater,
                 dropout_pre_weight=0.2,
                 dropout_gl=0.2,
                 dropout_imp=0.2):
        super(GN2DI, self).__init__()

        self.edge_index = None
        self.edge_weight = None

        self.pre_weighting = PreWeight(
            num_weights, 
            hidden_dim_pre_weight, 
            num_lay_pre_weight, 
            dropout_pre_weight
        )
        
        self.gl_module = GraphLearner(
            in_channel_gl,
            num_conv_lay_gl,
            hidden_dim_conv_gl,
            hidden_dim_gl, 
            hidden_dim_updater,
            dropout_gl
        )
        
        self.imp_module = Imputation(
            in_channel_imp,
            num_conv_lay_imp,
            hidden_dim_conv_imp,
            hidden_dim_readout_imp,
            dropout_imp
        )

    def impute(self, dynamic_node_features):
        x_hat = self.imp_module(
            dynamic_node_features,
            self.edge_index,
            self.edge_weight
        )
        return x_hat

    def forward(self, static_node_features, dynamic_node_features, edge_index, edge_weight):
        edge_weight = self.pre_weighting(edge_weight)
        edge_weight = self.gl_module(static_node_features, edge_index, edge_weight)
        
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        x_hat = self.imp_module(dynamic_node_features, edge_index, edge_weight)
        return x_hat
