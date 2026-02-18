import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear
from torch_scatter import scatter



class EdgeConv(nn.Module):
    def __init__(self, node_channels, edge_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.mlp = Seq(Linear(2*node_channels+edge_channels, out_channels),
                       torch.nn.ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        W = torch.cat([x_i, x_j, edge_attr], dim=1)
        edge_out = self.mlp(W)
        node_out = scatter(
            edge_out, edge_index[0], dim=0, dim_size=x.shape[0], reduce="mean")
        return node_out, edge_out


class MPNN(nn.Module):
    def __init__(self, num_layers, num_features=32, activation=torch.relu, mlp=[32, 32]):
        super().__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.ac = activation

        self.conv1 = EdgeConv(3, 3, self.num_features)
        self.conv2 = EdgeConv(self.num_features, self.num_features+3, self.num_features)

        self.convs = nn.ModuleList()
        for _ in range(self.num_layers-2):
            layer = EdgeConv(2*self.num_features,
                               2*self.num_features, self.num_features)
            self.convs.append(layer)
        
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(num_features, mlp[0]))
        self.mlp.append(nn.ReLU())
        for i in range(1, len(mlp)-1):
            self.mlp.append(nn.Linear(mlp[i], mlp[i+1]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(mlp[-1], 3))
        
    
    def forward(self, node_attr, edge_attr, edge_index):
        x1, edge_x1 = self.conv1(
            node_attr, edge_index, edge_attr)
        x1 = self.ac(x1)
        edge_x1 = self.ac(edge_x1)

        x2, edge_x2 = self.conv2(
            x1, edge_index, torch.cat([edge_x1, edge_attr], dim=1)) # [x1, node_attr]
        x2 = self.ac(x2)
        edge_x2 = self.ac(edge_x2)

        x_ll, x_l = x1, x2
        edge_ll, edge_l = edge_x1, edge_x2
        for conv in self.convs:
            x_cur, edge_cur = conv(
                torch.cat([x_l, x_ll], dim=1), edge_index, torch.cat([edge_l, edge_ll], dim=1))
        
            x_cur = self.ac(x_cur) 
            edge_cur = self.ac(edge_cur)

            x_ll, x_l = x_l, x_cur
            edge_ll, edge_l = edge_l, edge_cur

        x = self.mlp(x_l)
        return x, edge_l