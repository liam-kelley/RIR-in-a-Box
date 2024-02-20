import torch.nn as nn
from torch.linalg import norm
import torch
import torch.nn.functional as F
from typing import Optional

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from backpropagatable_ISM.compute_batch_rir_v2 import batch_simulate_rir_ism

from mesh2ir_meshnet import MESH_NET, data_for_meshnet

class ShoeboxToRIR(nn.Module):
    def __init__(self,sample_rate=16000, max_order=10):
        super().__init__()
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order
        self.batch_size=None

    def forward(self, input : torch.Tensor, force_absorption : Optional[torch.Tensor] = None):
        '''
        This will generate a RIR from the RIR-in-a-Box latent space.

        Args:
            input (torch.Tensor) : shoebox parameters. shape B * 10. (Room_dimensions (3) [0.0,+inf], mic_position (3) [0.0,1.0], source_position (3) [0.0,1.0], absorption (3) [0.0,1.0])
            force_absorption

        Returns:
            shoebox_rir_batch (list of torch.Tensor): batch of rir. shape (batch_size, rir_length*)
            shoebox_toa_batch (tensor) : shape B
        '''
        batch_size=input.shape[0]

        # Get shoebox parameters from RIRBox latent space.
        room_dimensions = input[:, 0:3]  # (batch_size, 3)
        mic_position = input[:, 3:6]*room_dimensions  # (batch_size, 1, 3)
        source_position = input[:, 6:9]*room_dimensions  # (batch_size, 3)
        if force_absorption is not None: absorption = force_absorption  # (batch_size, 1, 6)
        else: absorption = input[:, 9] # (batch_size)

        # Maybe faster batch simulate rir
        shoebox_rir_batch_2=batch_simulate_rir_ism(room_dimensions,mic_position.unsqueeze(1),source_position,
                                                    absorption.unsqueeze(1).unsqueeze(2).expand(-1,-1,6),
                                                    self.max_order, self.sample_rate)        

        # Get origins (Time of first arrival)
        distances = norm(mic_position-source_position, dim=1)
        shoebox_toa_batch = 40 + (self.sample_rate*distances/self.sound_speed) # 40 is delay_filter_length: int = 81 // 2

        return shoebox_rir_batch_2, shoebox_toa_batch


class MeshToShoebox(nn.Module):
    '''
    MODEL 1: (GNN + 2-layer MLP)
     -> concatenate mic pos and src pos
     -> 2-layer MLP
     -> (3 room dim + 3 absorption) embedding.
     -> LOSS: Sample 4 random msconf in that room, and mean the losses on the 4 RIR.
    
    MODEL 2: (GNN + 2-layer MLP)
     -> concatenate mic pos and src pos
     -> 2-layer MLP
     -> (3 room dim + 3 mic pos + 3 src pos + 3 absorption) embedding.
     -> LOSS: on RIR

    MODEL 3: MODEL 1 (pre-trained?)
    -> Concatenate mic pos and src pos again 
    -> 3-layer MLP 
    -> (3 room dim + 3 mic pos + 3 src pos + 3 absorption) embedding.  
    -> LOSS: on RIR
    '''
    def __init__(self, meshnet=None, model=1):
        super().__init__()
        # Model type
        assert(model in [1,2,3])
        self.model=model

        # Load (pretrained) mesh net
        if meshnet == None: self.meshnet = MESH_NET()
        else: self.meshnet = meshnet # for loading a pretrained meshnet

        # Linear layers
        self.lin3 = torch.nn.Linear(8, 32)
        if self.model in [1,3] : self.lin4 = torch.nn.Linear(32, 6)
        elif self.model == 2 : self.lin4 = torch.nn.Linear(32, 12)
        if self.model == 3 :
            self.lin5 == torch.nn.Linear(12, 32)
            self.lin6 == torch.nn.Linear(32, 6)

        # Activation
        self.softplus = torch.nn.Softplus()

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos):
        data = data_for_meshnet(x, edge_index, batch) # the pretrained mesh_net we use uses a data struct for input data.
        x = self.meshnet(data)
        
        x = torch.cat(batch_oracle_mic_pos, batch_oracle_src_pos, dim=1)
        x = F.relu(self.lin3(x))
        x = self.lin4(x)

        softplus_output = self.softplus(x[:,0:3])
        if self.model in [1,3] : sigmoid_output = torch.sigmoid(x[:,3:6])
        if self.model==2 : sigmoid_output = torch.sigmoid(x[:,3:12])
        x = torch.cat((softplus_output, sigmoid_output), dim=1)

        if self.model in [1,2]: return(x)

        if self.model == 3 :
            x = torch.cat(batch_oracle_mic_pos, batch_oracle_src_pos, dim=1)
            x = F.relu(self.lin5(x))
            x = torch.sigmoid(self.lin6(x))
            return(x)


# class GraphToShoeboxEncoder(nn.Module):
#     '''
#     DEPRECATED
#     used example https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py
#     As of right now, very similar the mesh encoder from MESH2IR.
#     '''
#     def __init__(self,training=False):
#         super().__init__()
#         # TopK pooling and GCN until fixed size
#         # GCN and 2 layer MLP Alternating
#         # MLP
#         self.training=training

#         # IN : mesh with features x, y, z, floor, wall, ceiling, platform, background, unknown (last six are boolean)
#         self.conv1 = GCNConv(in_channels=9, out_channels=32) #
#         self.pool1 = TopKPooling(in_channels=32, ratio=0.6)
#         self.conv2 = GCNConv(in_channels=32, out_channels=32)
#         self.pool2 = TopKPooling(in_channels=32, ratio=0.6)
#         self.conv3 = GCNConv(in_channels=32, out_channels=32)
#         self.pool3 = TopKPooling(in_channels=32, ratio=0.6)

#         # self.lin1 = torch.nn.Linear(192, 64)
#         # self.lin2 = torch.nn.Linear(64, 10)
#         # self.lin3 = torch.nn.Linear(16, 10) # Optionnal : only if oracle mic pos and src pos are given
#         self.lin1 = torch.nn.Linear(64, 16)
#         self.lin2 = torch.nn.Linear(16, 10)
#         self.lin3 = torch.nn.Linear(16, 10) # Optionnal : only if oracle mic pos and src pos are given
#         self.softplus = torch.nn.Softplus()
#         # OUT : 3d Shoebox dims, 3d mic position, 3d source position, 1 absorption

#     def forward(self, x, edge_index, batch=None, batch_oracle_mic_pos=None, batch_oracle_src_pos=None):
#         '''
#         batches are managed by the batch vector

#         inputs
#         x is the feature matrix of the nodes. X is of shape N(0) * F(0)
#         edge_index is the edge index matrix. E is of shape 2 * E(0) (number of edges)
#         mask is the mask matrix. M is of shape N(0)
#         The batch vector , which assigns each node to a specific example is of shape N(0)

#         outputs
#         shoebox_rir_batch, shoebox_origin_batch
#         shapes B * 24000, B
#         '''
#         # Check for NaNs or infs in inputs
#         assert not torch.isnan(x).any(), "NaNs detected in input 'x'"
#         assert not torch.isinf(x).any(), "Infs detected in input 'x'"
#         assert not torch.isnan(edge_index).any(), "NaNs detected in input 'edge_index'"
#         assert not torch.isinf(edge_index).any(), "Infs detected in input 'edge_index'"
#         if batch is not None:
#             assert not torch.isnan(batch).any(), "NaNs detected in input 'batch'"
#             assert not torch.isinf(batch).any(), "Infs detected in input 'batch'"

#         # Convolutional layer
#         x = F.relu(self.conv1(x, edge_index))

#         # Pooling layer
#         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         # Convolutional layer
#         x = F.relu(self.conv2(x, edge_index))

#         # Pooling layer
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         # Convolutional layer
#         x = F.relu(self.conv3(x, edge_index))

#         # Pooling layer
#         x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
#         # this is confusing
#         x = torch.cat([x1 ,x2, x3], dim=1)

#         # Cool, an MLP!
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.lin2(x))
#         if batch_oracle_mic_pos is not None and batch_oracle_src_pos is not None:
#             x = torch.cat([x, batch_oracle_mic_pos, batch_oracle_src_pos], dim=1)
#             x = self.lin3(x)
        
#         # This is always the final activation layer, constraining the dimensions.
#         softplus_output = self.softplus(x[:,0:3])
#         sigmoid_output = torch.sigmoid(x[:,3:10])
#         x = torch.cat((softplus_output, sigmoid_output), dim=1)

#         return x
