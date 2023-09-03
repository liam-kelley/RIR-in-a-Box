import torch.nn as nn
from torch import cat
from torch.linalg import norm
from RIRMetricsExperiments import get_pytorch_rir
import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

class ShoeboxToRIR(nn.Module):
    def __init__(self,sample_rate=48000, max_order=10):
        super().__init__()
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order

    def forward(self, input):
        '''
        inputs
        input = an embedding kind of like room_dimensions, mic_position, source_position
        shape B*9

        outputs
        shoebox_rir_batch, shoebox_origin_batch
        shapes B * 24000, B
        '''
        room_dimensions = input[:, 0:3]*10 # ARBITRARY NUMBER, HOW TO DO THIS DIFFERENTLY? USED A SIGMOID INSTEAD OF EXP...
        mic_position = input[:, 3:6]*room_dimensions
        source_position = input[:, 6:9]*room_dimensions

        batch_size=room_dimensions.shape[0]
        # Forward pass through backpropagatable pytorch shoebox RIR calculation # Is there a way to do this in parallel?
        for i in range(batch_size):
            shoebox_rir=get_pytorch_rir(room_dimensions[i],mic_position[i],source_position[i],self.sample_rate, max_order=self.max_order)
            if i==0: shoebox_rir_batch=shoebox_rir[None,:]
            else: shoebox_rir_batch=cat((shoebox_rir_batch,shoebox_rir[None,:]),0)

        # Get torch origins
        torch_distances = norm(mic_position-source_position, dim=1)
        shoebox_origin_batch = 40 + (self.sample_rate*torch_distances/self.sound_speed)

        return shoebox_rir_batch, shoebox_origin_batch


class RoomToRIR(nn.Module):
    '''
    It's a simple encoder module.
    It takes in a batch of input features (mic_position (3d), src_position (3d), vertices(nx (8) + ny (8))) of shape = [batch_size, 22].
    Produces an intermediate shoebox representation ((9,1) tensor = room_dim, mic_pos, src_pos)
    And finally outputs shoebox_rir and shoebox_rir_origin.

    Here, 8 2d vertices are used, but conceivably any amount of vertices is ok. Please initialize the input length to the right size.
    '''
    def __init__(self, input_size=22, hidden_sizes=[30,20], sample_rate=48000, max_order=10):
        super().__init__()
        self.basic_NN = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], 9),
            nn.Sigmoid()
        )
        self.STRIR = ShoeboxToRIR(sample_rate=sample_rate, max_order=max_order)

    def forward(self, input):
        '''
        outputs
        shoebox_rir_batch, shoebox_origin_batch
        shapes B * 24000, B
        '''
        z = self.basic_NN(input)

        shoebox_rir_batch, shoebox_origin_batch = self.STRIR(z)

        return shoebox_rir_batch, shoebox_origin_batch
    

class GraphToShoeboxEncoder(nn.Module):
    '''
    used example https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py
    
    should not work very well
    '''
    def __init__(self,
                 sample_rate=48000, max_order=10):
        super().__init__()
        # TopK pooling and GCN until fixed size
        # GCN and 2 layer MLP Alternating
        # MLP
        #4 # 3d mic position, 3d source position, 8 2d vertices
        self.training=False
        input_node_features=4
        final_embedding_size=9

        self.conv1 = GraphConv(in_channels=input_node_features, out_channels=128)
        self.pool1 = TopKPooling(in_channels=128, ratio=0.25)
        self.conv2 = GraphConv(in_channels=128, out_channels=128)
        self.pool2 = TopKPooling(in_channels=128, ratio=0.25)
        self.conv3 = GraphConv(in_channels=128, out_channels=128)
        self.pool3 = TopKPooling(in_channels=128, ratio=0.25)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, final_embedding_size)

        self.SBtoRIR = ShoeboxToRIR()

    def forward(self, x, edge_index, batch=None):
        '''
        batches are managed by the batch vector YEAH!

        inputs
        x is the feature matrix of the nodes. X is of shape N(0) * F(0)
        edge_index is the edge index matrix. E is of shape 2 * E(0) (number of edges)
        mask is the mask matrix. M is of shape N(0)
        The batch vector , which assigns each node to a specific example is of shape N(0)

        outputs
        shoebox_rir_batch, shoebox_origin_batch
        shapes B * 24000, B
        '''

        # Convolutional layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Pooling layer
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Convolutional layer
        x = F.relu(self.conv2(x, edge_index))

        # Pooling layer
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Convolutional layer
        x = F.relu(self.conv3(x, edge_index))

        # Pooling layer
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # this is confusing
        x = x1 + x2 + x3

        # Cool, an MLP!
        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.sigmoid(self.lin3(x))

        return x


# class a_GNN_with_a_couple_convolutionnal_layers(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,
#                  normalize=False, lin=True, concatenate=True):
#         super().__init__()
#         self.concatenate=concatenate

#         self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
#         self.bn1 = nn.BatchNorm1d(hidden_channels)

#         self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
#         self.bn2 = nn.BatchNorm1d(hidden_channels)

#         self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
#         self.bn3 = nn.BatchNorm1d(out_channels)

#         if lin is True:
#             if concatenate:
#                 self.lin = nn.Linear(2 * hidden_channels + out_channels,
#                                        out_channels)
#             else:
#                 self.lin = nn.Linear(out_channels, out_channels)
#         else:
#             self.lin = None

#     def bn(self, i, x):
#         '''batch norm'''
#         batch_size, num_nodes, num_channels = x.size()

#         x = x.view(-1, num_channels) # flatten
#         x = getattr(self, f'bn{i}')(x) # choose the right batch norm layer
#         x = x.view(batch_size, num_nodes, num_channels) # unflatten
#         return x

#     def forward(self, x, adj, mask=None):
#         batch_size, num_nodes, in_channels = x.size()

#         x0 = x

#         x1=self.conv1(x0, adj, mask)
#         x1=x1.relu()
#         x1=self.bn(1, x1)

#         x2=self.conv2(x1, adj, mask)
#         x2=x2.relu()
#         x2=self.bn(2, x2)

#         x3=self.conv3(x2, adj, mask)
#         x3=x3.relu()
#         x3=self.bn(3, x3)

#         if self.concatenate: x = torch.cat([x1, x2, x3], dim=-1) # concatenate 

#         if self.lin is not None:
#             x = self.lin(x).relu()

#         return x