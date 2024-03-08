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

from models.rirbox_models import MeshToShoebox, ShoeboxToRIR

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
        # shoebox_rir_batch = torch.nn.utils.rnn.pad_sequence(shoebox_rirs, batch_first=True, padding_value=0.0)
        shoebox_origin_batch = torch.tensor([window_length//2], device=shoebox_rir_batch.device).repeat(shoebox_rir_batch.shape[0])
        
        return shoebox_rir_batch, shoebox_origin_batch
    
    @staticmethod
    def get_batch_mixing_point(room_volume_batch : torch.Tensor):
        constant = 0.002 # used in that paper diego sent
        constant = 0.004 # I like this one better TODO validation search needed
        return torch.floor(constant * torch.sqrt(room_volume_batch) * 16000).int