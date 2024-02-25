import torch
from typing import Union, List
import auraloss

class BaseRIRLoss(torch.nn.Module):
    def __init__(self):
        ''' Base class for RIR losses. These losses work in a supervised manner on RIRs.'''
        super().__init__()
        self.batch_size=None
        self.sample_rate=None
    
    def check_input_batches(self, shoebox_rir_batch : Union[List[torch.Tensor],torch.Tensor], shoebox_origin_batch : torch.Tensor,
                                  label_rir_batch : Union[List[torch.Tensor],torch.Tensor], label_origin_batch : torch.Tensor):
        # if list of different length tensors, check that they're tensor
        if isinstance(shoebox_rir_batch, list): assert type(shoebox_rir_batch[0])==torch.Tensor, "shoebox_rir_batch is a list but not of tensors"
        if isinstance(label_rir_batch, list): assert type(label_rir_batch[0])==torch.Tensor, "label_rir_batch is a list but not of tensors"
        
        # check that all batches have the same length and device
        assert len(shoebox_rir_batch)==len(label_rir_batch) == shoebox_origin_batch.shape[0] == label_origin_batch.shape[0], "Input batch sizes are not the same"
        assert shoebox_rir_batch[0].device == shoebox_origin_batch.device == label_rir_batch[0].device == label_origin_batch.device, "Input tensor devices are not the same"

        # Get batch size
        self.batch_size=shoebox_origin_batch.shape[0]
    
    def crop_rirs_to_TOA_origin(self, shoebox_rir_batch : Union[List[torch.Tensor],torch.Tensor], shoebox_origin_batch : torch.Tensor,
                                    label_rir_batch : Union[List[torch.Tensor],torch.Tensor], label_origin_batch : torch.Tensor):
        '''Uniformizes the length of the RIRs by cropping them to the Direct Path origin, combines them into a tensor.'''
        new_shoebox_rir_batch=[]
        new_label_rir_batch=[]
        for i in range(self.batch_size):
            origin_shoebox=int(shoebox_origin_batch[i].item())
            origin_label=int(label_origin_batch[i].item())
            new_shoebox_rir_batch.append(shoebox_rir_batch[i][max(0,origin_shoebox-40):]) # 40 is half the frac delay window size
            new_label_rir_batch.append(label_rir_batch[i][max(0,origin_label-40):])
            
        shoebox_rir_batch=torch.nn.utils.rnn.pad_sequence(new_shoebox_rir_batch, batch_first=True).to(shoebox_origin_batch.device)
        label_rir_batch=torch.nn.utils.rnn.pad_sequence(new_label_rir_batch, batch_first=True).to(shoebox_origin_batch.device)

        # Free memory
        for i in range(self.batch_size):
            new_shoebox_rir_batch[i] = None
            new_label_rir_batch[i] = None
        del new_shoebox_rir_batch
        del new_label_rir_batch

        return shoebox_rir_batch, label_rir_batch

    def normalize_rir_dp(self, shoebox_rir_batch : torch.Tensor, shoebox_origin_batch : torch.Tensor,
                                label_rir_batch : torch.Tensor, label_origin_batch : torch.Tensor):
        '''Normalizes the Direct Path amplitude of the RIRs.'''
        assert self.sample_rate!=None, "Sample rate must be set by child class to normalize_rir_dp"
        shoebox_dp_dist = (shoebox_origin_batch*343.0)/self.sample_rate
        label_dp_dist = (label_origin_batch*343.0)/self.sample_rate
        shoebox_rir_batch=shoebox_rir_batch*(shoebox_dp_dist.unsqueeze(1))
        label_rir_batch=label_rir_batch*(label_dp_dist.unsqueeze(1))
        return shoebox_rir_batch, label_rir_batch
    
    def pad_rirs_to_same_length(self, shoebox_rir_batch : torch.Tensor, label_rir_batch : torch.Tensor):
        '''Pad rirs to same length from the right.'''
        if shoebox_rir_batch.shape[-1] < label_rir_batch.shape[-1]:
            shoebox_rir_batch = torch.nn.functional.pad(shoebox_rir_batch, (0, label_rir_batch.shape[-1]-shoebox_rir_batch.shape[-1])) # padding from the beginning because we flipped
        elif shoebox_rir_batch.shape[-1] > label_rir_batch.shape[-1]:
            label_rir_batch = torch.nn.functional.pad(label_rir_batch, (0, shoebox_rir_batch.shape[-1]-label_rir_batch.shape[-1])) # padding from the beginning because we flipped
        return shoebox_rir_batch, label_rir_batch

    def crop_rirs_to_same_length(self, shoebox_rir_batch : torch.Tensor, label_rir_batch : torch.Tensor):
        '''Crop rirs to same length from the right'''
        if min(shoebox_rir_batch.shape[-1], label_rir_batch.shape[-1]) < 1024: # Had a glitch where the shoebox rir generated was so small that mrstft couldn't perform on it.
            shoebox_rir_batch, label_rir_batch = self.pad_rirs_to_same_length(shoebox_rir_batch,label_rir_batch[...,:1024])
        else:
            if shoebox_rir_batch.shape[-1] < label_rir_batch.shape[-1]:
                label_rir_batch=label_rir_batch[..., :shoebox_rir_batch.shape[-1]]
            elif shoebox_rir_batch.shape[-1] > label_rir_batch.shape[-1]:
                shoebox_rir_batch=shoebox_rir_batch[..., :label_rir_batch.shape[-1]]
        return shoebox_rir_batch, label_rir_batch

    def deemphasize_rir_early_reflections(self, shoebox_rir_batch : torch.Tensor, label_rir_batch : torch.Tensor, t0 : int = 2000):
        '''linear ramp up from 0 to 1 on the interval [0,t0] in samples'''
        device = shoebox_rir_batch.device
        ramp = torch.arange(0, t0, device=device) / t0
        # Determine the actual length to use (the minimum of t0 and the last dimension of the tensor)
        actual_length = min(t0, shoebox_rir_batch.shape[-1])
        # Adjust ramp size if necessary
        if actual_length < t0:
            ramp = ramp[:actual_length]
        # Applying the ramp to both shoebox_rir_batch and label_rir_batch up to the actual_length
        shoebox_rir_batch[..., :actual_length] = shoebox_rir_batch[..., :actual_length] * ramp.unsqueeze(0)
        label_rir_batch[..., :actual_length] = label_rir_batch[..., :actual_length] * ramp.unsqueeze(0)

        return shoebox_rir_batch, label_rir_batch

    def forward(self, shoebox_rir_batch : Union[List[torch.Tensor],torch.Tensor], shoebox_origin_batch : torch.Tensor,
                      label_rir_batch : Union[List[torch.Tensor],torch.Tensor], label_origin_batch : torch.Tensor):
        raise NotImplementedError


class EnergyDecay_Loss(BaseRIRLoss):
    def __init__(self, frequency_wise=False, synchronize_TOA=True, normalize_dp=False, normalize_decay_curve=True,
                 deemphasize_early_reflections=True, pad_to_same_length=False, crop_to_same_length=True):
        super().__init__()

        self.mse=torch.nn.MSELoss(reduction='mean')

        # Options
        self.frequency_wise=frequency_wise # if True, compute loss on 7 frequency bands (EDR)
        self.synchronize_TOA=synchronize_TOA
        self.normalize_dp=normalize_dp
        self.normalize_decay_curve=normalize_decay_curve
        self.deemphasize_early_reflections=deemphasize_early_reflections
        self.pad_to_same_length=pad_to_same_length
        self.crop_to_same_length=crop_to_same_length
        if self.pad_to_same_length == self.crop_to_same_length:
            print("pad_to_same_length and crop_to_same_length can't be both True or both False. Defaulting to crop.")
            self.pad_to_same_length, self.crop_to_same_length = False, True

        # Options print
        print("EnergyDecay_Loss Initialized", end='\n    ')
        if self.frequency_wise: print("> with frequency-wise decay curves (EDR)", end='\n    ')
        if self.synchronize_TOA: print("> with TOA synchronization", end='\n    ')
        if self.normalize_dp: print("> with TOA normalization", end='\n    ')
        if self.normalize_decay_curve: print("> with decay curve normalization", end='\n    ')
        if self.deemphasize_early_reflections: print("> with deemphasized early reflections", end='\n    ')
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

        # do cumulative sum
        shoebox_rir_batch=torch.cumsum(torch.flip(shoebox_rir_batch, dims=[-1]), dim=-1) # Cumulative sum starting from the end
        label_rir_batch=torch.cumsum(torch.flip(label_rir_batch,dims=[-1]), dim=-1) # Cumulative sum starting from the end
        
        # flip
        shoebox_rir_batch, label_rir_batch = torch.flip(shoebox_rir_batch,dims=[-1]), torch.flip(label_rir_batch,dims=[-1])

        # normalize energy decay curves
        if self.normalize_decay_curve:
            if self.frequency_wise : sb_normalizer=shoebox_rir_batch[...,0,None] ; label_normalizer=label_rir_batch[...,0,None]
            else: sb_normalizer=shoebox_rir_batch[:,0].unsqueeze(1) ; label_normalizer=label_rir_batch[:,0].unsqueeze(1)
            shoebox_rir_batch=shoebox_rir_batch/(sb_normalizer+1e-8)
            label_rir_batch=label_rir_batch/(label_normalizer+1e-8)
            del sb_normalizer, label_normalizer

        # pad to same length
        if self.pad_to_same_length:
            shoebox_rir_batch, label_rir_batch = self.pad_rirs_to_same_length(shoebox_rir_batch, label_rir_batch)

        # crop to same length (saw it recommended in this paper : AV-RIR: Audio-Visual Room Impulse Response Estimation)
        if self.crop_to_same_length:
            shoebox_rir_batch, label_rir_batch = self.crop_rirs_to_same_length(shoebox_rir_batch, label_rir_batch)
            
        # Deemphasize early reflections linearly
        if self.deemphasize_early_reflections :
            shoebox_rir_batch, label_rir_batch = self.deemphasize_rir_early_reflections(shoebox_rir_batch, label_rir_batch, t0=2000)

        # Compute loss
        loss=self.mse(shoebox_rir_batch, label_rir_batch)
        return loss
    

class EnergyBins_Loss(BaseRIRLoss):
    def __init__(self, sample_rate=16000, synchronize_TOA=True, normalize_dp=True, frequency_wise=False, 
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


class MRSTFT_Loss(BaseRIRLoss):
    def __init__(self, sample_rate=16000, device='cpu',
                 synchronize_TOA=True, deemphasize_early_reflections=True, normalize_dp=True,
                pad_to_same_length=False, crop_to_same_length=True):
        super().__init__()

        if sample_rate == 16000:
            self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes = [256, 512, 1024], # 16000 sample rate
                    hop_sizes = [64, 128, 256],
                    win_lengths = [256, 512, 1024],
                    scale="mel",
                    n_bins=36,
                    sample_rate=sample_rate,
                    perceptual_weighting=True,
                    device=device,
                )
        elif sample_rate == 48000:
            self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[1024, 2048, 8192], # 48000 sample rate
                    hop_sizes=[256, 512, 2048],
                    win_lengths=[1024, 2048, 8192],
                    scale="mel",
                    n_bins=128,
                    sample_rate=sample_rate,
                    perceptual_weighting=True,
                    device=device,
                )
        else:
            raise NotImplementedError("MRSTFT Loss only implemented for 16000 and 48000 sample rates.")
        
        self.sample_rate=sample_rate

        # Options
        self.synchronize_TOA=synchronize_TOA
        self.deemphasize_early_reflections=deemphasize_early_reflections
        self.normalize_dp=normalize_dp
        self.pad_to_same_length=pad_to_same_length
        self.crop_to_same_length=crop_to_same_length
        if self.pad_to_same_length == self.crop_to_same_length:
            print("pad_to_same_length and crop_to_same_length can't be both True or both False. Defaulting to crop.")
            self.pad_to_same_length, self.crop_to_same_length = False, True

        # Options print
        print("MRSTFT_Loss Initialized", end='\n    ')
        if self.synchronize_TOA: print("> with TOA synchronization", end='\n    ')
        if self.deemphasize_early_reflections: print("> with deemphasized early reflections", end='\n    ')
        if self.normalize_dp: print("> with normalization", end='\n    ')
        if self.pad_to_same_length: print("> with RIR padding to same length", end='\n    ')
        if self.crop_to_same_length: print("> with RIR cropping to same length", end='\n    ')
        print("")
    
    def forward(self, shoebox_rir_batch : Union[List[torch.Tensor],torch.Tensor], shoebox_origin_batch : torch.Tensor,
                      label_rir_batch : Union[List[torch.Tensor],torch.Tensor], label_origin_batch : torch.Tensor):
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

        # pad to same length
        if self.pad_to_same_length:
            shoebox_rir_batch, label_rir_batch = self.pad_rirs_to_same_length(shoebox_rir_batch, label_rir_batch)
            
        # crop to same length (saw it recommended in this paper : AV-RIR: Audio-Visual Room Impulse Response Estimation)
        if self.crop_to_same_length:
            shoebox_rir_batch, label_rir_batch = self.crop_rirs_to_same_length(shoebox_rir_batch, label_rir_batch)

        # Deemphasize early reflections linearly
        if self.deemphasize_early_reflections :
            shoebox_rir_batch, label_rir_batch = self.deemphasize_rir_early_reflections(shoebox_rir_batch, label_rir_batch)

        # There was an error here
        batch_mrstft_loss = self.mrstft( shoebox_rir_batch[:,None,:], label_rir_batch[:,None,:] ) # Calculate batch_mrstft # Add a dimension for channels
        return batch_mrstft_loss   
        

class AcousticianMetrics_Loss(BaseRIRLoss):
    def __init__(self, sample_rate=16000, synchronize_TOA=True, crop_to_same_length=True, normalize_dp=False, frequency_wise=False,
                 normalize_total_energy=False, pad_to_same_length=False, MeanAroundMedian_pruning=True):
        super().__init__()

        self.mse=torch.nn.MSELoss(reduction='mean')
        self.sample_rate=sample_rate

        # Options
        self.synchronize_TOA=synchronize_TOA
        self.crop_to_same_length=crop_to_same_length
        self.normalize_dp=normalize_dp
        self.frequency_wise=frequency_wise # if True, compute loss on 7 frequency bands (EDR)
        self.normalize_total_energy=normalize_total_energy
        self.pad_to_same_length=pad_to_same_length
        if self.pad_to_same_length == self.crop_to_same_length:
            print("pad_to_same_length and crop_to_same_length can't be both True or both False. Defaulting to crop.")
            self.pad_to_same_length, self.crop_to_same_length = False, True
        self.MeanAroundMedian_pruning=MeanAroundMedian_pruning

        # Options print
        print("AcousticianMetrics_Loss Initialized", end='\n    ')
        if self.synchronize_TOA: print("> with TOA synchronization", end='\n    ')
        if self.crop_to_same_length: print("> with RIR cropping to same length", end='\n    ')
        if self.normalize_dp: print("> with TOA normalization", end='\n    ')
        if self.frequency_wise: print("> with frequency-wise decay curves (EDR)", end='\n    ')
        if self.normalize_total_energy: print("> with total energy normalization", end='\n    ')
        if self.pad_to_same_length: print("> with RIR padding to same length", end='\n    ')
        if self.MeanAroundMedian_pruning: print("> with MeanAroundMedian pruning", end='\n    ')
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

        # crop to same length (saw it recommended in this paper : AV-RIR: Audio-Visual Room Impulse Response Estimation)
        if self.crop_to_same_length:
            shoebox_rir_batch, label_rir_batch = self.crop_rirs_to_same_length(shoebox_rir_batch, label_rir_batch)

        # normalize direct path amplitudes
        if self.normalize_dp:
            shoebox_rir_batch, label_rir_batch = self.normalize_rir_dp(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)

        # Get energy
        if self.frequency_wise:
            shoebox_rir_batch=torch.stft(shoebox_rir_batch, n_fft=13, return_complex=True)
            label_rir_batch=torch.stft(label_rir_batch, n_fft=13, return_complex=True)
            batch_input_rir2 = (shoebox_rir_batch.real**2) + (shoebox_rir_batch.imag**2)
            batch_label_rir2 = (label_rir_batch.real**2) + (label_rir_batch.imag**2)
        else:
            batch_input_rir2=torch.pow(shoebox_rir_batch,2)
            batch_label_rir2=torch.pow(label_rir_batch,2)

        del shoebox_rir_batch, label_rir_batch

        # Precalculate sum(rir^2)
        batch_input_rir2_sum=torch.sum(batch_input_rir2, axis=-1)
        batch_label_rir2_sum=torch.sum(batch_label_rir2, axis=-1)

        # Normalize energy
        if self.normalize_total_energy:
            batch_input_rir2=batch_input_rir2/(batch_input_rir2_sum.unsqueeze(-1).expand_as(batch_input_rir2))
            batch_label_rir2=batch_label_rir2/(batch_label_rir2_sum.unsqueeze(-1).expand_as(batch_label_rir2))

        # C80
        batch_eighty_ms=(0.080*self.sample_rate)*torch.ones(self.batch_size, device=batch_input_rir2.device) # in samples
        
        batch_before_80ms, batch_after_80ms = batch_cut_before_and_after_index(batch_input_rir2, batch_eighty_ms, cut_severity=0.05)
        lower_integral=torch.sum(batch_before_80ms, dim=-1)
        top_integral=torch.sum(batch_after_80ms, dim=-1)
        batch_input_c80 = 10*torch.log10((top_integral/(lower_integral + 1e-10))+torch.ones_like(lower_integral))
        
        batch_before_80ms, batch_after_80ms = batch_cut_before_and_after_index(batch_label_rir2, batch_eighty_ms, cut_severity=0.05)
        lower_integral=torch.sum(batch_before_80ms, dim=-1)
        top_integral=torch.sum(batch_after_80ms, dim=-1)
        batch_label_c80 = 10*torch.log10((top_integral/(lower_integral + 1e-10))+torch.ones_like(lower_integral))

        del batch_before_80ms, batch_after_80ms, lower_integral, top_integral

        # D
        batch_fifty_ms=(0.050**self.sample_rate)*torch.ones(self.batch_size, device=batch_input_rir2.device) # in samples
    
        batch_before_50ms = batch_cut_before_and_after_index(batch_input_rir2, batch_fifty_ms, cut_severity=0.05, return_after=False)
        partial_integral=torch.sum(batch_before_50ms, dim=-1)
        batch_input_D = partial_integral/batch_input_rir2_sum

        batch_before_50ms = batch_cut_before_and_after_index(batch_label_rir2, batch_fifty_ms, cut_severity=0.05, return_after=False)
        partial_integral=torch.sum(batch_before_50ms, dim=-1)
        batch_label_D = partial_integral/batch_label_rir2_sum

        del batch_before_50ms, partial_integral

        # RT60
        batch_input_edc=torch.flip(torch.cumsum(torch.flip(batch_input_rir2, [-1]), dim=-1), [-1]) # Cumulative sum from the end
        batch_label_edc=torch.flip(torch.cumsum(torch.flip(batch_label_rir2, [-1]), dim=-1), [-1]) # Cumulative sum from the end

        # epsilon1, epsilon2 = 0.005, 0.5 # Only observe the decay curve between these two thresholds for RT60 estimation (this is bad)
        # batch_input_thresholds=(batch_input_edc.detach()[:,0]).unsqueeze(-1).expand_as(batch_input_edc)
        # batch_label_thresholds=(batch_label_edc.detach()[:,0]).unsqueeze(-1).expand_as(batch_label_edc)

        # batch_input_regression_mask = torch.logical_and(
        #     batch_input_thresholds * epsilon1 < batch_input_edc, 
        #     batch_input_edc < batch_input_thresholds * epsilon2
        # )
        # batch_label_regression_mask = torch.logical_and(
        #     batch_label_thresholds * epsilon1 < batch_label_edc, 
        #     batch_label_edc < batch_label_thresholds * epsilon2
        # )

        # del batch_input_thresholds, batch_label_thresholds

        if not self.normalize_total_energy: # RT60 needs to have its edc normalized
            batch_input_edc=batch_input_edc/batch_input_rir2_sum.unsqueeze(1).expand_as(batch_input_edc)
            batch_label_edc=batch_label_edc/batch_label_rir2_sum.unsqueeze(1).expand_as(batch_input_edc)

        # pad to same length (if we aren't cropping to same length earlier, then we'll need to this here to make the input and label comparable)
        if self.pad_to_same_length:
            batch_input_edc, batch_label_edc = self.pad_rirs_to_same_length(batch_input_edc, batch_label_edc)
            # batch_input_regression_mask, batch_label_regression_mask = self.pad_rirs_to_same_length(batch_input_regression_mask, batch_label_regression_mask)

        batch_input_edc = -torch.log(torch.clamp(batch_input_edc, min=1e-10))
        batch_label_edc = -torch.log(torch.clamp(batch_label_edc, min=1e-10))

        batch_input_times_seconds=(torch.arange(batch_input_edc.shape[-1], device=batch_input_edc.device) / self.sample_rate).unsqueeze(0).expand_as(batch_input_edc)
        batch_label_times_seconds=(torch.arange(batch_label_edc.shape[-1], device=batch_label_edc.device) / self.sample_rate).unsqueeze(0).expand_as(batch_label_edc)

        input_betas = (batch_input_edc[:,1:]/batch_input_times_seconds[:,1:])
        label_betas = (batch_label_edc[:,1:]/batch_label_times_seconds[:,1:])

        del batch_input_edc, batch_label_edc, batch_input_times_seconds, batch_label_times_seconds

        # Used to use median instead of mean, but median is not differentiable. Now I'm using a "Mean Around Median".
        if self.MeanAroundMedian_pruning:
            input_variance = torch.var(input_betas.detach(), dim=-1)
            label_variance = torch.var(label_betas.detach(), dim=-1)
            input_std = torch.sqrt(input_variance).unsqueeze(-1).expand_as(input_betas)
            label_std = torch.sqrt(label_variance).unsqueeze(-1).expand_as(label_betas)
            del input_variance, label_variance
            input_median = torch.median(input_betas.detach(), dim=-1)[0].unsqueeze(-1).expand_as(input_betas)
            label_median = torch.median(label_betas.detach(), dim=-1)[0].unsqueeze(-1).expand_as(label_betas)
            input_betas=input_betas[torch.abs(input_betas.detach()-input_median) < 2*input_std]
            label_betas=label_betas[torch.abs(label_betas.detach()-label_median) < 2*label_std]
            del input_std, label_std, input_median, label_median
        input_regressed_beta = torch.mean(input_betas, dim=-1)
        label_regressed_beta = torch.mean(label_betas, dim=-1)

        input_rt60=6/(input_regressed_beta+1e-10)
        label_rt60=6/(label_regressed_beta+1e-10)

        del input_regressed_beta, label_regressed_beta

        # Compute losses
        loss_c80=self.mse(batch_input_c80, batch_label_c80)
        loss_D=self.mse(batch_input_D, batch_label_D)
        loss_rt60=self.mse(input_rt60, label_rt60)

        input_betas,label_betas = self.crop_rirs_to_same_length(input_betas,label_betas)      
        loss_betas = self.mse(input_betas,label_betas)
        
        return loss_c80, loss_D, loss_rt60, loss_betas


####### C80 and D Utility #######

def batch_cut_before_and_after_index(batch_tensor : torch.Tensor, batch_indexes : torch.Tensor, cut_severity=1.0, return_after=True):
    '''
    Cuts a batch_tensor before and after specific batch_indexes.
    '''
    assert len(batch_tensor.shape)==2
    assert batch_tensor.shape[0]==batch_indexes.shape[0]

    arange=torch.arange(batch_tensor.shape[1],device=batch_tensor.device)
    batch_arange=arange.unsqueeze(0).expand(batch_tensor.shape[0], -1)

    batch_indexes=batch_indexes.unsqueeze(1).expand(-1,batch_tensor.shape[1])  # repeat(batch_tensor.shape[1], 1).transpose(0,1)

    batch_cut_after_index=torch.sigmoid((batch_arange-batch_indexes)*cut_severity)
    batch_cut_before_index=1-batch_cut_after_index
    
    if return_after:
        return batch_tensor*batch_cut_before_index, batch_tensor*batch_cut_after_index
    else:
        return batch_tensor*batch_cut_before_index