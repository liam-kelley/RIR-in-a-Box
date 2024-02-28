import torch.nn as nn
from torch.linalg import norm
import torch
import torch.nn.functional as F
from typing import Optional

# from torch_geometric.nn import GCNConv, TopKPooling
# from torch_geometric.nn import global_max_pool as gmp
# from torch_geometric.nn import global_mean_pool as gap

from backpropagatable_ISM.compute_batch_rir_v2 import batch_simulate_rir_ism

from models.mesh2ir_models import MESH_NET, data_for_meshnet

class ShoeboxToRIR(nn.Module):
    def __init__(self,sample_rate=16000, max_order=15, rir_length=3968, force_absorption : Optional[torch.Tensor] = None):
        super().__init__()
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order
        self.rir_length=rir_length
        self.force_absorption = force_absorption
        print("ShoeboxToRIR initialized.")

    def forward(self, input : torch.Tensor):
        '''
        This will generate a RIR from the RIR-in-a-Box latent space.

        Args:
            input (torch.Tensor) : shoebox parameters. shape B * 12. (Room_dimensions (3) [0.0,+inf], mic_position (3) [0.0,1.0], source_position (3) [0.0,1.0], absorption (3) [0.0,1.0]) # Walls, floor, ceiling

        Returns:
            shoebox_rir_batch (list of torch.Tensor): batch of rir. shape (batch_size, rir_length*)
            shoebox_toa_batch (tensor) : shape B
        '''
        # Get shoebox parameters from RIRBox latent space.
        room_dimensions = input[:, 0:3]  # (batch_size, 3)
        mic_position = input[:, 3:6]*room_dimensions  # (batch_size, 1, 3)
        source_position = input[:, 6:9]*room_dimensions  # (batch_size, 3)
        if self.force_absorption is not None: absorption = self.force_absorption  # (batch_size, 3) # Walls, floor, ceiling
        else: absorption = input[:, 9:12] # (batch_size, 3) # Walls, floor, ceiling

        # Convert rirbox Absorption values to ISM format
        absorption = torch.cat((absorption[:, 9].unsqueeze(1).expand(-1,4),
                                absorption[:, 10].unsqueeze(1),
                                absorption[:, 11].unsqueeze(1)), dim=1)  # (batch_size, 6) # west, east, south, north, floor, ceiling
        absorption = absorption.unsqueeze(1)  # (batch_size, n_bands=1, n_walls=6) # n_bands=1 for now, because the backpropagatable ISM code does not support multiple bands yet.

        # Figure out how our RIR impulses should be LP filtered
        # (this is a tempory solution before we implement proper multi-band support)
        cutoff_frequency = 2000
        window_length = ShoeboxToRIR.get_window_length(self.sample_rate, center_frequency=cutoff_frequency/2)

        # Batch simulate rir
        shoebox_rir_batch_2=batch_simulate_rir_ism(room_dimensions,mic_position.unsqueeze(1),source_position, absorption,
                                                    self.max_order, self.sample_rate, output_length=self.rir_length,
                                                    window_length=window_length, lp_cutoff_frequency=cutoff_frequency)     

        # Get origins (Time of first arrival)
        distances = norm(mic_position-source_position, dim=1)
        shoebox_toa_batch = window_length//2 + (self.sample_rate*distances/self.sound_speed)

        return shoebox_rir_batch_2, shoebox_toa_batch
    
    @staticmethod
    def get_window_length(fs=16000, center_frequency=500):
        filter_transition_band = center_frequency / 5 # a good approximation.
        filter_order = fs / filter_transition_band # a good approximation.
        window_length = (filter_order // 2)*2  + 1
        return window_length

class MeshToShoebox(nn.Module):
    '''
    MODEL 1: MESH_NET (GNN + 2-layer MLP)
     -> concatenate mic pos and src pos
     -> 2-layer MLP
     -> (3 room dim + 3 absorption) embedding.
     -> LOSS: Sample 4 random msconf in that room, and mean the losses on the 4 RIR.
    
    MODEL 2: MESH_NET (GNN + 2-layer MLP)
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
        self.lin3 = torch.nn.Linear(14, 32)
        if self.model in [1,3] : self.lin4 = torch.nn.Linear(32, 6)
        elif self.model == 2 : self.lin4 = torch.nn.Linear(32, 12)
        if self.model == 3 :
            self.lin5 == torch.nn.Linear(12, 32)
            self.lin6 == torch.nn.Linear(32, 6)

        # Activation
        self.softplus = torch.nn.Softplus()
        print("MeshToShoebox model ", self.model, " initialized")

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos):
        data = data_for_meshnet(x, edge_index, batch) # the pretrained mesh_net we use uses a data struct for input data.
        x = self.meshnet(data)
        
        x = torch.cat((x, batch_oracle_mic_pos, batch_oracle_src_pos), dim=1)
        x = F.relu(self.lin3(x))
        x = self.lin4(x)

        softplus_output = self.softplus(x[:,0:3])
        if self.model in [1,3] : sigmoid_output = torch.sigmoid(x[:,3:6])
        if self.model==2 : sigmoid_output = torch.sigmoid(x[:,3:12])
        x = torch.cat((softplus_output, sigmoid_output), dim=1)

        if self.model in [1,2]: return(x)

        if self.model == 3 :
            x = torch.cat((batch_oracle_mic_pos, batch_oracle_src_pos), dim=1)
            x = F.relu(self.lin5(x))
            x = torch.sigmoid(self.lin6(x))
            return(x)

class RIRBox_FULL(nn.Module):
    '''
    combines both parts of the RIRBox model for simple evaluation or training.
    '''
    def __init__(self, mesh_to_sbox, sbox_to_rir):
        super(RIRBox_FULL, self).__init__()
        self.mesh_to_sbox = mesh_to_sbox
        self.sbox_to_rir = sbox_to_rir

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos):
        latent_shoebox_batch = self.mesh_to_sbox(x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos)
        shoebox_rir_batch, shoebox_origin_batch = self.sbox_to_rir(latent_shoebox_batch)
        return shoebox_rir_batch, shoebox_origin_batch
