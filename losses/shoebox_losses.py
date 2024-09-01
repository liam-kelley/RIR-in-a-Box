'''
This script contains various Shoebox-based losses for pytorch.
'''

import torch
from torch.nn import MSELoss

class ShoeboxLoss(torch.nn.Module):
    def __init__(self):
        ''' Base class for shoebox losses. These losses work on toy data between a proposed shoebox and a target shoebox. '''
        super().__init__()
        self.mse=MSELoss()
        print(str(self) + " initialized.")

    def __str__(self) -> str:
        return super().__str__()
    
    def check_z_batches(self, proposed_z_batch : torch.Tensor, label_z_batch : torch.Tensor):
        assert(proposed_z_batch.shape==label_z_batch.shape)
        assert(proposed_z_batch.shape[1]==10)
        batch_size=proposed_z_batch.shape[0]
        device=proposed_z_batch.device
        return(batch_size,device)
    
    def forward(self):
        pass


class SBoxRoomDimensionsLoss(ShoeboxLoss):
    '''Loss module for Shoebox Room Dimensions. Useful for toy shoebox datasets.'''
    
    def __str__(self) -> str:
        return "Shoebox Room Dimensions Loss"
    
    def forward(self, proposed_z_batch, label_z_batch):
        '''
        args:
            proposed_z_batch: torch.Tensor. shape: (batch_size, 10)
            label_z_batch: torch.Tensor. shape: (batch_size, 10)
        returns:
            torch.Tensor. shape: (1)
        '''
        self.check_z_batches(proposed_z_batch, label_z_batch)
        proposed_room_dimensions=proposed_z_batch[:,:2]
        target_room_dimensions=label_z_batch[:,:2]
        target_room_dimensions_opposite=torch.cat((label_z_batch[:,1].unsqueeze(1),label_z_batch[:,0].unsqueeze(1)),dim=-1)
        room_dimensions_loss=torch.sqrt(self.mse(proposed_room_dimensions, target_room_dimensions)*self.mse(proposed_room_dimensions, target_room_dimensions_opposite))
        return room_dimensions_loss
    

class SBoxAbsorptionLoss(ShoeboxLoss):
    '''Loss module for Shoebox wall absorptions. Useful for toy shoebox datasets.'''
    
    def __str__(self) -> str:
        return "Shoebox Absorption Loss"
    
    def forward(self, proposed_z_batch, label_z_batch):
        '''
        args:
            proposed_z_batch: torch.Tensor. shape: (batch_size, 10)
            label_z_batch: torch.Tensor. shape: (batch_size, 10)
        returns:
            torch.Tensor. shape: (1)
        '''
        self.check_z_batches(proposed_z_batch, label_z_batch)
        proposed_absorption=proposed_z_batch[:,9]
        target_absorption=label_z_batch[:,9]
        absorption_loss=self.mse(proposed_absorption, target_absorption)
        return(absorption_loss)
    

class MicSrcConfigurationLoss(ShoeboxLoss):
    def __init__(self, return_separate_losses=True, lambdas={"mic":1,"src":1,"mic_src_vector":1,"src_mic_vector":1,"mic_src_distance":1}):
        '''
        Loss module for Shoebox Mic-Source Configuartions. Used on Toy shoebox dataset.
        Symmetries are accounted for and distances are accounted for.

        FORWARD:
        args:
            proposed_z_batch: torch.Tensor. shape: (batch_size, 10)
            label_z_batch: torch.Tensor. shape: (batch_size, 10)
        returns:
            mic_loss, source_loss, mic_source_vector_loss, source_mic_vector_loss, mic_source_distance_loss: all torch.Tensor. shape: (1)
            or
            total_loss: torch.Tensor. shape: (1)
        '''
        super().__init__()
        self.return_separate_losses=return_separate_losses
        self.lambdas=lambdas

    def __str__(self) -> str:
        return "Shoebox Mic-Src Configuration Loss"
    
    def forward(self, proposed_z_batch, label_z_batch, return_separate_losses=False):
        batch_size,device = self.check_z_batches(proposed_z_batch, label_z_batch)

        # proposed_room_dimensions=proposed_z_batch[:,:3]
        # target_room_dimensions=label_z_batch[:,:3]

        proposed_mic_pos=proposed_z_batch[:,3:6]
        proposed_src_pos=proposed_z_batch[:,6:9]

        # Get Mic, Src losses
        center_batch=torch.tensor([[0.5,0.5,0.5]],device=device).expand(batch_size,-1)
        target_mic_pos=label_z_batch[:,3:6]
        target_src_pos=label_z_batch[:,6:9]
        mic_loss = self.mse(torch.abs(target_mic_pos-center_batch), torch.abs(proposed_mic_pos-center_batch))
        src_loss = self.mse(torch.abs(target_src_pos-center_batch), torch.abs(proposed_src_pos-center_batch))

        # Get Mic-Src Vector, Src-Mic Vector losses
        mic_src_vector_loss = self.mse(torch.abs(target_src_pos-target_mic_pos), torch.abs(proposed_src_pos-target_mic_pos))
        src_mic_vector_loss = self.mse(torch.abs(target_mic_pos-target_src_pos), torch.abs(proposed_mic_pos-target_src_pos))
        
        # Get Mic-Src distance loss
        mic_src_distance=torch.linalg.norm(proposed_mic_pos-proposed_src_pos, dim=1)
        target_mic_src_distance=torch.linalg.norm(target_mic_pos-target_src_pos, dim=1)
        mic_src_distance_loss=self.mse(mic_src_distance, target_mic_src_distance)

        if self.return_separate_losses or return_separate_losses :
            return mic_loss, src_loss, mic_src_vector_loss, src_mic_vector_loss, mic_src_distance_loss
        else:
            total_loss= mic_loss * self.lambdas["mic"]+\
                        src_loss * self.lambdas["src"]+\
                        mic_src_vector_loss * self.lambdas["mic_src_vector"]+\
                        src_mic_vector_loss * self.lambdas["src_mic_vector"]+\
                        mic_src_distance_loss * self.lambdas["mic_src_distance"]
            return total_loss
