from typing import List, Union
import torch
from losses.rir_losses import BaseRIRLoss

class EnergyBins_Loss(BaseRIRLoss):
    def __init__(self, sample_rate=16000, synchronize_TOA=False, normalize_dp=False, frequency_wise=False, 
                pad_to_same_length=False, crop_to_same_length=True):
        super().__init__()

        self.sample_rate=sample_rate
        self.mse=torch.nn.MSELoss(reduction='mean')

        # Options
        self.synchronize_TOA=synchronize_TOA
        self.normalize_dp=normalize_dp
        self.frequency_wise=frequency_wise # if True, compute loss on 7 frequency bands (EDR)
        self.pad_to_same_length=pad_to_same_length
        self.crop_to_same_length=crop_to_same_length
        if self.pad_to_same_length == self.crop_to_same_length:
            print("pad_to_same_length and crop_to_same_length can't be both True or both False. Defaulting to crop.")
            self.pad_to_same_length, self.crop_to_same_length = False, True

        # Options print
        print("EnergyDecay_Loss Initialized", end='\n    ')
        if self.synchronize_TOA: print("> with TOA synchronization", end='\n    ')
        if self.normalize_dp: print("> with TOA normalization", end='\n    ')
        if self.frequency_wise: print("> with frequency-wise decay curves (EDR)", end='\n    ')
        if self.pad_to_same_length: print("> with RIR padding to same length", end='\n    ')
        if self.crop_to_same_length: print("> with RIR cropping to same length", end='\n    ')
        print("")
    
    def forward(self, shoebox_rir_batch : Union[List[torch.Tensor],torch.Tensor], shoebox_origin_batch : torch.Tensor,
                      label_rir_batch : Union[List[torch.Tensor],torch.Tensor], label_origin_batch : torch.Tensor):
        '''
        args:
            shoebox_rir_batch: list of torch.Tensor, each tensor is a shoebox rir
            shoebox_origin_batch: torch.Tensor, each element is the origin of the corresponding shoebox rir
            label_rir_batch: list of torch.Tensor, each tensor is a label rir
            label_origin_batch: torch.Tensor, each element is the origin of the corresponding label rir
        '''
        self.check_input_batches(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)

        # crop rirs to begin from Direct Path
        if self.synchronize_TOA:
            shoebox_rir_batch, label_rir_batch = self.crop_rirs_to_TOA_origin(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
        else:
            shoebox_rir_batch=torch.nn.utils.rnn.pad_sequence(shoebox_rir_batch, batch_first=True).to(shoebox_origin_batch.device)
            label_rir_batch=torch.nn.utils.rnn.pad_sequence(label_rir_batch, batch_first=True).to(shoebox_origin_batch.device)

        # normalize direct path amplitudes
        if self.normalize_dp:
            shoebox_rir_batch, label_rir_batch = self.normalize_rir_dp(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)

        # Convert to energy (or compute stft square magnitude on 7 bands (nfft//2 + 1))
        if self.frequency_wise:
            shoebox_rir_batch=torch.stft(shoebox_rir_batch, n_fft=13, return_complex=True)
            label_rir_batch=torch.stft(label_rir_batch, n_fft=13, return_complex=True)
            shoebox_rir_batch = (shoebox_rir_batch.real**2) + (shoebox_rir_batch.imag**2)
            label_rir_batch = (label_rir_batch.real**2) + (label_rir_batch.imag**2)
        else:
            shoebox_rir_batch=torch.pow(shoebox_rir_batch,2)
            label_rir_batch=torch.pow(label_rir_batch,2)

        # pad to same length
        if self.pad_to_same_length:
            shoebox_rir_batch, label_rir_batch = self.pad_rirs_to_same_length(shoebox_rir_batch, label_rir_batch)

        # crop to same length (saw it recommended in this paper : AV-RIR: Audio-Visual Room Impulse Response Estimation)
        if self.crop_to_same_length:
            shoebox_rir_batch, label_rir_batch = self.crop_rirs_to_same_length(shoebox_rir_batch, label_rir_batch)
            
        # do cumulative sum
        shoebox_rir_batch=torch.flip(torch.cumsum(torch.flip(shoebox_rir_batch, dims=(-1,)), dim=-1), dims=(-1,))
        label_rir_batch=torch.flip(torch.cumsum(torch.flip(label_rir_batch, dims=(-1,)), dim=-1), dims=(-1,))

        # Convert to energy bins
        strides=[i*self.sample_rate for i in [0.005, 0.001, 0.002,0.005,0.01]] # times in ms converted to sample rate
        shoebox_rir_batch=torch.cat([torch.diff(shoebox_rir_batch[...,::int(stride)]) for stride in strides], dim=-1)
        label_rir_batch=torch.cat([torch.diff(label_rir_batch[...,::int(stride)]) for stride in strides], dim=-1)

        # Compute loss
        loss=torch.sqrt(self.mse(shoebox_rir_batch, label_rir_batch))
        return loss