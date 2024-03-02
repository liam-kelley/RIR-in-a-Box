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

from models.mesh2ir_models import MESH_NET, data_for_meshnet, STAGE1_G

class ShoeboxToRIR(nn.Module):
    def __init__(self,sample_rate=16000, max_order=15, rir_length=3968, start_from_ir_onset=False):
        super().__init__()
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order
        self.rir_length=rir_length
        self.window_length=81
        self.start_from_ir_onset = start_from_ir_onset
        print("ShoeboxToRIR initialized.")

    def forward(self, input : torch.Tensor):
        '''
        This will generate a RIR from the RIR-in-a-Box latent space.

        Args:
            input (torch.Tensor) : shoebox parameters. shape B * 12. (Room_dimensions (3) [0.0,+inf], mic_position (3) [0.0,1.0], source_position (3) [0.0,1.0], absorption (3) [0.0,1.0]) # Walls, floor, ceiling

        Returns:
            shoebox_rir_batch (list of torch.Tensor): batch of rir. shape (batch_size, rir_length*)
            shoebox_ir_onset_batch (tensor) : shape B
        '''
        # Get shoebox parameters from RIRBox latent space.
        room_dimensions = input[:, 0:3]
        mic_position = input[:, 3:6]*room_dimensions
        source_position = input[:, 6:9]*room_dimensions
        absorption = torch.cat((input[:, 9].unsqueeze(1).expand(-1,4),
                                input[:, 10].unsqueeze(1),
                                input[:, 11].unsqueeze(1)), dim=1)  # (batch_size, 6) # west, east, south, north, floor, ceiling
        absorption = (absorption*0.84) + 0.01 # Constrain to realistic absorption values
        absorption = absorption.unsqueeze(1)  # (batch_size, n_bands=1, n_walls=6)
        # n_bands=1 for now, because the backpropagatable ISM code does not support multiple bands yet.
        
        # Batch simulate rir
        shoebox_rir_batch_2=batch_simulate_rir_ism(room_dimensions, mic_position.unsqueeze(1), source_position, absorption,
                                                    self.max_order, self.sample_rate, output_length=self.rir_length,
                                                    window_length=self.window_length,
                                                    start_from_ir_onset=self.start_from_ir_onset)

        # Get shoebox ir onset
        if self.start_from_ir_onset:
            shoebox_ir_onset_batch = torch.tensor([self.window_length//2], device=mic_position.device).repeat(mic_position.shape[0])
        else :
            distances = norm(mic_position-source_position, dim=1)
            shoebox_ir_onset_batch = self.window_length//2 + (self.sample_rate*distances/self.sound_speed)

        return shoebox_rir_batch_2, shoebox_ir_onset_batch
    
    @staticmethod
    def get_a_good_window_length(fs=16000, center_frequency=500):
        filter_transition_band = center_frequency *1 #* 0.2 # a good approximation.
        filter_order = fs / filter_transition_band # a good approximation.
        window_length = (filter_order // 2)*2  + 1
        return window_length

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
    def __init__(self, meshnet : MESH_NET = None, model : int = 2, MLP_Depth : int = 3):
        super().__init__()
        assert model in [2,3, 4], "Model 2 or 3 or 4 only"
        # Model type
        self.model=model
        self.MLP_depth=MLP_Depth

        # Load (pretrained) mesh net
        if meshnet == None: self.meshnet = MESH_NET()
        else: self.meshnet = meshnet # for loading a pretrained meshnet

        # Linear layers
        if self.model == 2 :
            self.lin3 = torch.nn.Linear(14, 48)
            if self.MLP_depth >= 3: self.lin4 = torch.nn.Linear(48, 48)
            if self.MLP_depth >= 4: self.lin6 = torch.nn.Linear(48, 48)
            self.lin5 = torch.nn.Linear(48, 12)
        if self.model == 3 :
            self.lin3 = torch.nn.Linear(14, 32)
            if self.MLP_depth >= 3:self.lin4 = torch.nn.Linear(32, 32)
            if self.MLP_depth >= 4: self.lin9 = torch.nn.Linear(32, 32)
            self.lin5 = torch.nn.Linear(32, 6)
            self.lin6 = torch.nn.Linear(12, 32)
            if self.MLP_depth >= 3:self.lin7 = torch.nn.Linear(32, 32)
            if self.MLP_depth >= 4: self.lin10 = torch.nn.Linear(32, 32)
            self.lin8 = torch.nn.Linear(32, 6)

        # Activation
        self.softplus = torch.nn.Softplus()
        print("MeshToShoebox model ", self.model, " initialized")

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos : Tensor, batch_oracle_src_pos : Tensor):
        data = data_for_meshnet(x, edge_index, batch) # the pretrained mesh_net we use uses a data struct for input data.
        x = self.meshnet(data)

        x = torch.cat((x, batch_oracle_mic_pos, batch_oracle_src_pos), dim=1)

        if self.model == 2 :
            x = F.relu(self.lin3(x))
            if self.MLP_depth >= 3: x = F.relu(self.lin4(x))
            if self.MLP_depth >= 4: x = F.relu(self.lin6(x))
            x = self.lin5(x)

            softplus_output = self.softplus(x[:,0:3])
            sigmoid_output = torch.sigmoid(x[:,3:12])

            x = torch.cat((softplus_output, sigmoid_output), dim=1)
            return(x)
        
        if self.model == 3 :
            x = F.relu(self.lin3(x))
            if self.MLP_depth >=3: x = F.relu(self.lin4(x))
            if self.MLP_depth >=4: x = F.relu(self.lin9(x))
            x = self.lin5(x)

            room_dims = self.softplus(x[:,0:3])
            absorptions = torch.sigmoid(x[:,3:6])

            x = torch.cat((room_dims, absorptions, batch_oracle_mic_pos, batch_oracle_src_pos), dim=1)

            x = F.relu(self.lin6(x))
            if self.MLP_depth >= 3: x = F.relu(self.lin7(x))
            if self.MLP_depth >= 4: x = F.relu(self.lin10(x))
            x = self.lin8(x)

            mic_and_src_pos = torch.sigmoid(x)

            x = torch.cat((room_dims, mic_and_src_pos, absorptions), dim=1)

            return(x)

class RIRBox_FULL(nn.Module):
    '''
    combines both parts of the RIRBox model for simple inference.
    '''
    def __init__(self, mesh_to_sbox : MeshToShoebox, sbox_to_rir : ShoeboxToRIR):
        super().__init__()
        self.mesh_to_sbox = mesh_to_sbox.eval()
        self.sbox_to_rir = sbox_to_rir.eval()
        print("RIRBox_FULL initialized.")

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos : Tensor, batch_oracle_src_pos : Tensor):
        latent_shoebox_batch = self.mesh_to_sbox(x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos)
        shoebox_rir_batch, shoebox_origin_batch = self.sbox_to_rir(latent_shoebox_batch)
        return shoebox_rir_batch, shoebox_origin_batch

class RIRBox_MESH2IR_Hybrid(nn.Module):
    '''
    combines both RIRBOX and MESH2IR at the mixing point, for potentially the highest quality RIRs. INFERENCE ONLY.
    '''
    def __init__(self, mesh_to_sbox : MeshToShoebox, sbox_to_rir : ShoeboxToRIR):
        super().__init__()
        self.mesh_to_sbox = mesh_to_sbox.eval()
        self.sbox_to_rir = sbox_to_rir.eval()
        print("RIRBox_MESH2IR_Hybrid initialized.")

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos : Tensor, batch_oracle_src_pos : Tensor,
                                     mesh2ir_estimated_rir_batch : Tensor, mesh2ir_estimated_origin_batch : Tensor):
        latent_shoebox_batch = self.mesh_to_sbox(x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos)
        shoebox_rir_batch, shoebox_origin_batch = self.sbox_to_rir(latent_shoebox_batch)

        # crop mesh2ir to 3968 (get rid of that weird std)
        mesh2ir_estimated_rir_batch = mesh2ir_estimated_rir_batch[:,0:3968]

        # get mixing point from rirbox latent shoebox volume and that formula
        batch_room_volume=torch.prod(latent_shoebox_batch[:,0:3], dim=1)
        batch_mixing_points = torch.floor(0.002 * torch.sqrt(batch_room_volume) * 16000).int()

        # This sucks to do batch wise, so i guess... Separate the batch, and iterate the procedure with a for loop.
        window_length=81
        shoebox_rirs = []
        for i in range(shoebox_rir_batch.shape[0]):
            temp_shoebox_rir = shoebox_rir_batch[i]
            temp_mesh2ir_rir = mesh2ir_estimated_rir_batch[i]
            # synchronize both origins/onsets to window length // 2 (rirbox should already be like that if you used the start_from_ir_onset option on ShoeboxToRIR.
            shoebox_syncronized_origin = int(max(shoebox_origin_batch[i].item() - (window_length // 2), 0))
            mesh2ir_syncronized_origin = int(max(mesh2ir_estimated_origin_batch[i].item() - (window_length // 2), 0))
            temp_shoebox_rir = temp_shoebox_rir[shoebox_syncronized_origin:]
            temp_mesh2ir_rir = temp_mesh2ir_rir[mesh2ir_syncronized_origin:]

            # combine mesh2ir rir from mixing point onwards
            until = min(len(temp_shoebox_rir), len(temp_mesh2ir_rir))
            mixing_point = batch_mixing_points[i].item()
            if until > mixing_point:
                # # additive mode
                # temp_shoebox_rir[mixing_point:until] = temp_shoebox_rir[mixing_point:until] + temp_mesh2ir_rir[mixing_point:until]
                # # replace mode
                # temp_shoebox_rir[mixing_point:until] = mesh2ir_estimated_rir_batch[i, mixing_point:until]
                # Ramp mode
                ramp_length = min(until-mixing_point, 200)
                temp_shoebox_rir[mixing_point:mixing_point+ramp_length] = temp_shoebox_rir[mixing_point:mixing_point+ramp_length] * torch.linspace(1,0,ramp_length, device=temp_shoebox_rir.device)
                temp_shoebox_rir[mixing_point:mixing_point+ramp_length] = temp_shoebox_rir[mixing_point:mixing_point+ramp_length] + temp_mesh2ir_rir[mixing_point:mixing_point+ramp_length] * torch.linspace(0,1,ramp_length, device=temp_shoebox_rir.device)
                temp_shoebox_rir[mixing_point+ramp_length:until] = temp_mesh2ir_rir[mixing_point+ramp_length:until]

            # append
            shoebox_rirs.append(temp_shoebox_rir)
        
        # Recombine and pad your newly fabricated rirs. Enjoy!
        shoebox_rir_batch = torch.nn.utils.rnn.pad_sequence(shoebox_rirs, batch_first=True, padding_value=0.0)
        shoebox_origin_batch = torch.tensor([window_length//2], device=shoebox_rir_batch.device).repeat(shoebox_rir_batch.shape[0])

        return shoebox_rir_batch, shoebox_origin_batch
    
    @staticmethod
    def get_batch_mixing_point(room_volume_batch : torch.Tensor):
        return torch.floor(0.002 * torch.sqrt(room_volume_batch) * 16000).int
