import torch
import torch.nn as nn
import torch.nn.parallel
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class MESH_NET(nn.Module):
    '''
    baseline lifted from MESH2IR
    '''
    def __init__(self):
        super(MESH_NET,self).__init__()
        self.feature_dim = 3
        self.conv1 = GCNConv(self.feature_dim, 32)
        self.pool1 = TopKPooling(32, ratio=0.6)
        self.conv2 = GCNConv(32, 32) #(32, 64)
        self.pool2 = TopKPooling(32, ratio=0.6) #64, ratio=0.6)
        self.conv3 = GCNConv(32, 32) #(64, 128)
        self.pool3 = TopKPooling(32, ratio=0.6) #(128, ratio=0.6)
        # self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +1, embedding_dim=self.feature_dim)
        self.lin1 = torch.nn.Linear(64, 16) #(256, 128)
        self.lin2 = torch.nn.Linear(16, 8) #(128, 64)
        # self.lin3 = torch.nn.Linear(8, 1) #(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(16) #(128)
        self.bn2 = torch.nn.BatchNorm1d(8) #(64)
        self.act1 = torch.nn.ReLU()
        # self.act2 = torch.nn.ReLU()        
  
    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch
        # x = self.item_embedding(x)
        # x = x.squeeze(1)
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ ,_= self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _,_ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _,_ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        # x = self.lin2(x)
        # x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin2(x)).squeeze(1)
        return x
