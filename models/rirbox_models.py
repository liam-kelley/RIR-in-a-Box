import torch.nn as nn
from torch.linalg import norm
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import math

# from torch_geometric.nn import GCNConv, TopKPooling
# from torch_geometric.nn import global_max_pool as gmp
# from torch_geometric.nn import global_mean_pool as gap

from backpropagatable_ISM.compute_batch_rir_v2 import batch_simulate_rir_ism

from models.mesh2ir_models import MESH_NET, data_for_meshnet

class ShoeboxToRIR(nn.Module):
    def __init__(self,sample_rate : int = 16000, max_order : int = 15, rir_length : int = 3968, start_from_ir_onset : bool = False, normalized_distance=False):
        '''
        start_from_ir_onset will place the IR onset at t_samples = 41.
        Computationally, this avoids keeping a bunch of zeroes in memory, and having more space for ISM echoes to roam free.
        WARNING: THIS IS ONLY USEFUL IF YOU USE THE SYNCHRONIZE TOA OPTION ON THE LOSSES.
        IF YOU DONT USE THE SYNCHRONIZE TOA OPTION, THEN YOU WON'T BE ABLE TO COMPUTE THE LOSS CORRECTLY,
            OR YOU WILL HAVE TO MANUALLY RESYNCHRONIZE THE RIR, NON-BATCHWISE, SO THIS IS ONLY GOOD FOR INFERENCE. BUT EVEN THEN... WHY?
        '''
        super().__init__()
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order
        self.rir_length=rir_length
        self.window_length=81
        self.start_from_ir_onset = start_from_ir_onset
        self.normalized_distance= normalized_distance
        print("ShoeboxToRIR initialized.")

    def forward(self, input : torch.Tensor):
        '''
        This will generate a RIR from the RIR-in-a-Box latent space.

        Args:
            input (torch.Tensor) : shoebox parameters. shape B * 12. (Room_dimensions (3) [0.0,+inf], mic_position (3) [0.0,1.0], source_position (3) [0.0,1.0], absorption (3) [0.0,1.0]) # Walls, floor, ceiling

        Returns:
            shoebox_rir_batch (list of torch.Tensor): batch of rir. shape (batch_size, rir_length)
            shoebox_ir_onset_batch (tensor) : shape B
        '''
        room_dimensions, mic_position, source_position, absorption = ShoeboxToRIR.extract_shoebox_from_latent_representation(input)
        
        # Batch simulate rir
        shoebox_rir_batch=batch_simulate_rir_ism(room_dimensions, mic_position.unsqueeze(1), source_position, absorption,
                                                    self.max_order, self.sample_rate, output_length=self.rir_length,
                                                    window_length=self.window_length,
                                                    start_from_ir_onset=self.start_from_ir_onset,
                                                    normalized_distance=self.normalized_distance)

        # fix IR onset to be correct and comparable to baseline and GT
        if not self.start_from_ir_onset:
            # TODO this can cause problems if mic and src in the virtual room are too close. But I don't have time to think about that.
            shoebox_rir_batch = shoebox_rir_batch[...,self.window_length//2:]
            distances = norm(mic_position-source_position, dim=1)
            shoebox_ir_onset_batch = (self.sample_rate*distances/self.sound_speed)
        if self.start_from_ir_onset:
            # MAKE SURE TO ONLY ACTIVATE THIS IF YOU ARE DOING INFERENCE, HAVE BATCH WISE 1 AND YOU PLAN ON RESPATIALIZING THE RIR,
            # OR YOU ARE DOING BATCH IR GENERATION BUT ARE EXCLUSIVELY USING "SYNC IR ONSETS" / "SYNC TOA" OPTIONS IN LOSSES.
            shoebox_ir_onset_batch = torch.tensor([self.window_length//2], device=mic_position.device).repeat(mic_position.shape[0])

        return shoebox_rir_batch, shoebox_ir_onset_batch
    
    @staticmethod
    def extract_shoebox_from_latent_representation(input:Tensor):
        assert(input.shape[1] == 12)
        # Get shoebox parameters from RIRBox latent space.
        room_dimensions = input[:, 0:3]
        room_dimensions = room_dimensions + torch.ones_like(room_dimensions)
        mic_position = input[:, 3:6]*room_dimensions
        source_position = input[:, 6:9]*room_dimensions
        absorption = torch.cat((input[:, 9].unsqueeze(1).expand(-1,4),
                                input[:, 10].unsqueeze(1),
                                input[:, 11].unsqueeze(1)), dim=1)  # (batch_size, 6) # west, east, south, north, floor, ceiling
        absorption = (absorption*0.84) + 0.01 # Constrain to realistic absorption values
        absorption = absorption.unsqueeze(1)  # (batch_size, n_bands=1, n_walls=6)
        # n_bands=1 for now, because the backpropagatable ISM code does not support multiple bands yet.
        return[room_dimensions,mic_position,source_position,absorption]
    
    @staticmethod
    def respatialize_rirbox(rir : torch.Tensor, dp_onset_in_samples : int):
        '''
        Use this if you used the start_from_ir_onset option on ShoeboxToRIR and you want to have the RIR match the actual distance between mic and src.
        This is only implemented non-batch wise for now, because doing it batch_wise is a pain.
        dp onset in samples is distance between mic and src * sample rate / sound speed
        '''
        window_length=81
        if dp_onset_in_samples > (window_length//2):
            # pad rir by dp_onset_in_samples - (window_length//2)
            rir = torch.cat((torch.zeros(rir.shape[0], dp_onset_in_samples-(window_length//2),device=rir.device), rir), dim=1)
            # crop rir_ribox
            rir = rir[:,:3968]
        if dp_onset_in_samples < (window_length//2):
            # crop rir
            rir = rir[:,(window_length//2)-dp_onset_in_samples:]
        origin = torch.tensor([dp_onset_in_samples],device=rir.device)
        return rir, origin
        

class MeshToShoebox(nn.Module):
    '''
    MODEL 2: MESH_NET (GNN + 2-layer MLP)
     -> concatenate GT mic pos and GT src pos
     -> 3-layer MLP
     -> (3 room dim + 3 mic pos + 3 src pos + 3 absorption) embedding.
     -> LOSS: on RIR

    MODEL 3:
    -> concatenate GT mic pos and GT src pos
    -> 3-layer MLP
    -> (3 room dim + 3 absorption) embedding
    -> Concatenate GT mic pos and GT src pos again 
    -> 3-layer MLP 
    -> (3 room dim + 3 mic pos + 3 src pos + 3 absorption) embedding.  
    -> LOSS: on RIR
    '''
    def __init__(self, meshnet : MESH_NET = None, model : int = 2, MLP_Depth : int = 3, hidden_size : int = 64,
                       dropout_p : float = 0.5, random_noise : bool = False, distance_in_latent_vector : bool = False):
        super().__init__()
        assert model in [2,3, 4], "Model 2 or 3 or 4 only"
        # Model type
        self.model=model
        self.MLP_depth=MLP_Depth
        self.dropout = nn.Dropout(dropout_p)
        self.random_noise = random_noise
        self.hidden_size=hidden_size
        self.distance_in_latent_vector = distance_in_latent_vector
        if not self.distance_in_latent_vector :
            self.latent_vector_size = 14
            self.model3_intermediate_latent_vector_size = 12
        else :
            self.latent_vector_size = 15
            self.model3_intermediate_latent_vector_size = 13

        # Load (pretrained) mesh net
        if meshnet == None: self.meshnet = MESH_NET()
        else: self.meshnet = meshnet # for loading a pretrained meshnet

        # Linear layers
        if self.model == 2 :
            self.lin3 = torch.nn.Linear(self.latent_vector_size, hidden_size) ; nn.init.kaiming_normal_(self.lin3.weight, mode="fan_out")
            if self.MLP_depth >= 3: self.lin4 = torch.nn.Linear(hidden_size, hidden_size) ; nn.init.kaiming_normal_(self.lin4.weight, mode="fan_out")
            if self.MLP_depth >= 4: self.lin6 = torch.nn.Linear(hidden_size, hidden_size) ; nn.init.kaiming_normal_(self.lin6.weight, mode="fan_out")
            self.lin5 = torch.nn.Linear(hidden_size, 12) ; nn.init.xavier_normal_(self.lin5.weight)
        if self.model == 3 :
            self.lin3 = torch.nn.Linear(self.latent_vector_size, hidden_size) ; nn.init.kaiming_normal_(self.lin3.weight, mode="fan_out")
            if self.MLP_depth >= 3:self.lin4 = torch.nn.Linear(hidden_size, hidden_size) ; nn.init.kaiming_normal_(self.lin4.weight, mode="fan_out")
            if self.MLP_depth >= 4: self.lin9 = torch.nn.Linear(hidden_size, hidden_size) ; nn.init.kaiming_normal_(self.lin9.weight, mode="fan_out")
            self.lin5 = torch.nn.Linear(hidden_size, 6) ; nn.init.xavier_normal_(self.lin5.weight)
            self.lin6 = torch.nn.Linear(self.model3_intermediate_latent_vector_size, hidden_size) ; nn.init.kaiming_normal_(self.lin6.weight, mode="fan_out")
            if self.MLP_depth >= 3:self.lin7 = torch.nn.Linear(hidden_size, hidden_size) ; nn.init.kaiming_normal_(self.lin7.weight, mode="fan_out")
            if self.MLP_depth >= 4: self.lin10 = torch.nn.Linear(hidden_size, hidden_size) ; nn.init.kaiming_normal_(self.lin10.weight, mode="fan_out")
            self.lin8 = torch.nn.Linear(hidden_size, 6) ; nn.init.xavier_normal_(self.lin8.weight)

        # Activation
        self.softplus = torch.nn.Softplus()
        print("MeshToShoebox model ", self.model, " initialized")

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos : Tensor, batch_oracle_src_pos : Tensor):
        data = data_for_meshnet(x, edge_index, batch) # the pretrained mesh_net we use uses a data struct for input data.
        x = self.meshnet(data)

        # add tiny random noise to x to explore latent space more ???
        if self.random_noise : x = x + torch.rand_like(x, device=x.device)*1e-3

        if not self.distance_in_latent_vector : 
            x = torch.cat((x, batch_oracle_mic_pos, batch_oracle_src_pos), dim=1)
        else :
            batch_oracle_distances = norm(batch_oracle_mic_pos-batch_oracle_src_pos, dim=1).unsqueeze(1)
            x = torch.cat((x, batch_oracle_mic_pos, batch_oracle_src_pos, batch_oracle_distances), dim=1)

        if self.model == 2 :
            x = self.dropout(F.relu(self.lin3(x)))
            if self.MLP_depth >= 3: x = self.dropout(F.relu(self.lin4(x)))
            if self.MLP_depth >= 4: x = self.dropout(F.relu(self.lin6(x)))
            x = self.lin5(x)

            softplus_output = self.softplus(x[:,0:3])
            sigmoid_output = torch.sigmoid(x[:,3:12])

            x = torch.cat((softplus_output, sigmoid_output), dim=1)
            return(x)
        
        if self.model == 3 :
            x = self.dropout(F.relu(self.lin3(x)))
            if self.MLP_depth >=3: x = self.dropout(F.relu(self.lin4(x)))
            if self.MLP_depth >=4: x = self.dropout(F.relu(self.lin9(x)))
            x = self.lin5(x)

            room_dims = self.softplus(x[:,0:3])
            absorptions = torch.sigmoid(x[:,3:6])

            if not self.distance_in_latent_vector :
                x = torch.cat((room_dims, absorptions, batch_oracle_mic_pos, batch_oracle_src_pos), dim=1)
            else :
                x = torch.cat((room_dims, absorptions, batch_oracle_mic_pos, batch_oracle_src_pos, batch_oracle_distances), dim=1)

            x = self.dropout(F.relu(self.lin6(x)))
            if self.MLP_depth >= 3: x = self.dropout(F.relu(self.lin7(x)))
            if self.MLP_depth >= 4: x = self.dropout(F.relu(self.lin10(x)))
            x = self.lin8(x)

            mic_and_src_pos = torch.sigmoid(x)

            x = torch.cat((room_dims, mic_and_src_pos, absorptions), dim=1)

            return(x)

class RIRBox_FULL(nn.Module):
    '''
    combines both parts of the RIRBox model for simple inference.
    '''
    def __init__(self, mesh_to_sbox : MeshToShoebox, sbox_to_rir : ShoeboxToRIR, return_sbox : bool = True):
        super().__init__()
        self.mesh_to_sbox = mesh_to_sbox.eval()
        self.sbox_to_rir = sbox_to_rir.eval()
        self.return_sbox = return_sbox
        print("RIRBox_FULL initialized.")

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos : Tensor, batch_oracle_src_pos : Tensor):
        latent_shoebox_batch = self.mesh_to_sbox(x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos)
        shoebox_rir_batch, shoebox_origin_batch = self.sbox_to_rir(latent_shoebox_batch)
        if not self.return_sbox :
            return shoebox_rir_batch, shoebox_origin_batch
        else :
            return shoebox_rir_batch, shoebox_origin_batch, latent_shoebox_batch
