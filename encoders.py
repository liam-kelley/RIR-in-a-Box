import torch.nn as nn
from numpy import concatenate
from torch import stack
from torch.linalg import norm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional

from torch_geometric.nn import GCNConv, GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from compute_rir_v2 import torch_ism
from compute_batch_rir_v2 import batch_simulate_rir_ism

from pyLiam.LKTimer import LKTimer
timer=LKTimer(print_time=True)

class ShoeboxToRIR(nn.Module):
    def __init__(self,sample_rate=48000, max_order=10):
        super().__init__()
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order
        self.batch_size=None
        self.streams=None

    def forward(self, input : torch.Tensor, force_absorption : Optional[torch.Tensor] = None):
        '''
        Args:
            input (torch.Tensor) : shoebox parameters. shape B * 10. (Room_dimensions (3) [0.0,+inf], mic_position (3) [0.0,1.0], source_position (3) [0.0,1.0], absorption (3) [0.0,1.0])
            force_absorption

        Returns:
            shoebox_rir_batch (list of torch.Tensor): batch of rir. shape (batch_size, rir_length*)
            shoebox_origin_batch (tensor) : shape B
        '''
        batch_size=input.shape[0]

        # Get shoebox parameters
        room_dimensions = input[:, 0:3]  # (batch_size, 3)
        mic_position = input[:, 3:6]*room_dimensions  # (batch_size, 1, 3)
        source_position = input[:, 6:9]*room_dimensions  # (batch_size, 3)
        if force_absorption is not None: absorption = force_absorption  # (batch_size, 1, 6)
        else: absorption = input[:, 9] # (batch_size)

        # Maybe faster batch simulate rir
        shoebox_rir_batch_2=batch_simulate_rir_ism(room_dimensions,mic_position.unsqueeze(1),source_position,
                                                    absorption.unsqueeze(1).unsqueeze(2).expand(-1,-1,6),
                                                    self.max_order, self.sample_rate)        

        # Get torch origins
        torch_distances = norm(mic_position-source_position, dim=1)
        shoebox_origin_batch = 40 + (self.sample_rate*torch_distances/self.sound_speed) # 40 is delay_filter_length: int = 81 // 2

        return shoebox_rir_batch_2, shoebox_origin_batch


class GraphToShoeboxEncoder(nn.Module):
    '''
    used example https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py
    As of right now, very similar the mesh encoder from MESH2IR
    '''
    def __init__(self,training=False):
        super().__init__()
        # TopK pooling and GCN until fixed size
        # GCN and 2 layer MLP Alternating
        # MLP
        self.training=training

        # IN : mesh with features x, y, z, floor, wall, ceiling, platform, background, unknown (last six are boolean)
        self.conv1 = GCNConv(in_channels=9, out_channels=32) #
        self.pool1 = TopKPooling(in_channels=32, ratio=0.6)
        self.conv2 = GCNConv(in_channels=32, out_channels=32)
        self.pool2 = TopKPooling(in_channels=32, ratio=0.6)
        self.conv3 = GCNConv(in_channels=32, out_channels=32)
        self.pool3 = TopKPooling(in_channels=32, ratio=0.6)

        self.lin1 = torch.nn.Linear(192, 64)
        self.lin2 = torch.nn.Linear(64, 10)
        self.lin3 = torch.nn.Linear(16, 10) # Optionnal : only if oracle mic pos and src pos are given
        # OUT : 3d Shoebox dims, 3d mic position, 3d source position, 1 absorption

    def forward(self, x, edge_index, batch=None, batch_oracle_mic_pos=None, batch_oracle_src_pos=None):
        '''
        batches are managed by the batch vector

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
        x = F.relu(self.conv1(x, edge_index))

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
        x = torch.cat([x1 ,x2, x3], dim=1)

        # Cool, an MLP!
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        if batch_oracle_mic_pos is not None and batch_oracle_src_pos is not None:
            x = torch.cat([x, batch_oracle_mic_pos, batch_oracle_src_pos], dim=1)
            x = self.lin3(x)
        
        # This is always the final activation layer.
        x[:,0:3] = torch.exp(x[:,0:3])
        x[:,3:10] = torch.sigmoid(x[:,3:10])

        return x

    @staticmethod
    def plot_intermediate_shoeboxes(x, label_x, show=True):
        fig, ax = plt.subplots(1, figsize=(7, 7))
        boum=x.detach().cpu().numpy()[0]
        bing=label_x.detach().cpu().numpy()[0]
        room_dimensions = boum[0:3]
        label_room_dimensions = bing[0:3]
        room_dimensions = concatenate((room_dimensions,[room_dimensions[0]]), axis=0)
        label_room_dimensions = concatenate((label_room_dimensions,[label_room_dimensions[0]]), axis=0)
        ax.plot([0, room_dimensions[0], room_dimensions[0], 0, 0], [0, 0, room_dimensions[1], room_dimensions[1], 0], c='blue',label='intermediate shoebox')
        ax.plot([0, label_room_dimensions[0], label_room_dimensions[0], 0, 0], [0, 0, label_room_dimensions[1], label_room_dimensions[1], 0], c='darkorange',label='label shoebox')
        ax.add_patch(Rectangle((0, 0), room_dimensions[0], room_dimensions[1],
                                    alpha=0.2, facecolor = 'darkblue', fill=True))
        ax.add_patch(Rectangle((0, 0), label_room_dimensions[0], label_room_dimensions[1],
                                    alpha=0.3, facecolor = 'orange', fill=True))
        ax.text(-0.85, -0.35, 'Intermediate room height = ' + str(room_dimensions[2]),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
        ax.text(-0.85, -0.75, 'Label room height = ' + str(label_room_dimensions[2]),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
        mic_pos=boum[3:6]*room_dimensions[:3]
        label_mic_pos=bing[3:6]*label_room_dimensions[:3]
        ax.scatter(mic_pos[0], mic_pos[1], marker='x', c='darkblue',label='mic')
        ax.scatter(label_mic_pos[0], label_mic_pos[1], marker='x', c='darkorange',label='mic')
        source_pos=boum[6:9]*room_dimensions[:3]
        label_source_pos=bing[6:9]*label_room_dimensions[:3]
        ax.scatter(source_pos[0], source_pos[1],  c='darkblue', label='source')
        ax.scatter(label_source_pos[0], label_source_pos[1],  c='darkorange', label='source')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(-1, max(boum[0],bing[0])+1)
        ax.set_ylim(-1, max(boum[1],bing[1])+1)
        # ax.set_aspect('equal', 'box')
        ax.set_title('Intermediate shoebox loss')
        ax.grid(True, ls=':', alpha=0.5)
        ax.legend()

        if show: plt.show()


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
        self.lin1 = torch.nn.Linear(64, 16) #(256, 128)
        self.act1 = torch.nn.ReLU() 
        self.lin2 = torch.nn.Linear(16, 8) #(128, 64)

    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch

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
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin2(x)).squeeze(1)
        return x


from torch_geometric.nn import DenseSAGEConv
# SAGE CONV: MEAN
# GINConv : SUM
# GATConv : Attention
# GINConv : SUM
# GINConv : SUM
class little_CGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        # Input : Graph features
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        self.lin2 = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        x = self.lin1(x).relu()
        x = self.lin2(x).relu()

        return x

from torch_geometric.nn import dense_diff_pool

# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#         num_nodes = ceil(0.25 * max_nodes)
#         self.gnn1_pool = little_CGNN(dataset.num_features, 64, num_nodes)
#         self.gnn1_embed = little_CGNN(dataset.num_features, 64, 64, lin=False)

#         num_nodes = ceil(0.25 * num_nodes)
#         self.gnn2_pool = little_CGNN(3 * 64, 64, num_nodes)
#         self.gnn2_embed = little_CGNN(3 * 64, 64, 64, lin=False)

#         self.gnn3_embed = little_CGNN(3 * 64, 64, 64, lin=False)

#         self.lin1 = torch.nn.Linear(3 * 64, 64)
#         self.lin2 = torch.nn.Linear(64, dataset.num_classes)

#     def forward(self, x, adj, mask=None):
#         s = self.gnn1_pool(x, adj, mask)
#         x = self.gnn1_embed(x, adj, mask)

#         x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

#         s = self.gnn2_pool(x, adj)
#         x = self.gnn2_embed(x, adj)

#         x, adj, l2, e2 = dense_diff_pool(x, adj, s)

#         x = self.gnn3_embed(x, adj)

#         x = x.mean(dim=1)
#         x = self.lin1(x).relu()
#         x = self.lin2(x)
#         return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2