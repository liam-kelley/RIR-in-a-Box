####### ms_env Utility #######

def batch_get_envelopes_no_filtering(batch_rir_fft):
    '''
    From frequential domain to normalized enveloppes
    '''
    batch_rir_fft=torch.fft.irfft(batch_rir_fft)
    batch_rir_fft=torch.abs(batch_rir_fft)
    batch_rir_fft=batch_rir_fft/torch.max(batch_rir_fft, dim=1).values
    
    return batch_rir_fft

def truncate_to_origin_and_pad(batch_rir1_enveloppe, batch_rir2_enveloppe, batch_origin1, batch_origin2, truncate_less=1):
    '''
    Truncates the batch_rir_enveloppe tensors to the origin points and pads them with zeros at the end so they keep matching size.
    '''
    assert(batch_origin1!=None)
    assert(batch_origin2!=None)
    assert(batch_origin1.shape[0]==batch_origin2.shape[0])

    origin_truncated_list1=[]
    origin_truncated_list2=[]
    batch_origin1.to('cpu')
    batch_origin2.to('cpu')
    for i in range(batch_rir1_enveloppe.shape[0]):
        item1=batch_origin1[i].item()
        item2=batch_origin2[i].item()
        if item1 is float('NaN'): index1=0
        else: index1=int(item1)
        if item2 is float('NaN'): index2=0
        else: index2=int(item2)
        origin_truncated_list1.append(nn.functional.pad(batch_rir1_enveloppe[i][max(index1-truncate_less,0):], (0,index1)))
        origin_truncated_list2.append(nn.functional.pad(batch_rir2_enveloppe[i][max(index2-truncate_less,0):], (0,index2)))
    batch_rir1_enveloppe=torch.stack(origin_truncated_list1)
    batch_rir2_enveloppe=torch.stack(origin_truncated_list2)
    return batch_rir1_enveloppe, batch_rir2_enveloppe

def pad_to_match_size(tensor1, tensor2):
    '''Pads either tensor so they match size.
    Works with batches or not batches'''
    if tensor1.shape[-1] > tensor2.shape[-1]:
        pad_length=tensor1.shape[-1]-tensor2.shape[-1]
        # Apply zero-padding to tensor2
        tensor2_padded = torch.nn.functional.pad(tensor2, (0, pad_length))
        return tensor1, tensor2_padded
    elif tensor1.shape[-1] < tensor2.shape[-1]:
        pad_length=tensor2.shape[-1]-tensor1.shape[-1]
        # Apply zero-padding to tensor1
        tensor1_padded = torch.nn.functional.pad(tensor1, (0, pad_length))
        return tensor1_padded, tensor2
    else:
        return tensor1, tensor2

def create_filter(tensor, beginning_sample, sample_width):
    filter=torch.arange(0,tensor.shape[-1]).float()
    def func(x):
        if x < beginning_sample:
            return 1.0
        elif x > beginning_sample + sample_width:
            return 0.0
        else:
            return(0.5 + (math.cos((math.pi)*(x-(beginning_sample))/sample_width)/2))
    filter.apply_(func)
    return filter

def get_filtered_envelopes(batch_rir_fft, filters):
    '''
    Returns a batch of envelopes filtered by a batch of filters.
    Shape of batch_rir_fft : (batch_size, rir_length)
    Shape of filters : (batch_size, rir_length, number_of_filters)
    Shape of batch_normalized_filtered_rir_envelope : (batch_size, rir_length, number_of_filters)
    '''
    batch_rir_fft=batch_rir_fft.repeat(filters.shape[2],1,1).transpose(0,1)#.transpose(1,2)
    batch_filtered_rir=torch.fft.irfft(batch_rir_fft*(filters.transpose(1,2))).transpose(1,2)
    batch_filtered_rir_envelope=torch.abs(batch_filtered_rir)
    normalizer_max=(torch.max(batch_filtered_rir_envelope, dim=1).values).repeat(batch_filtered_rir_envelope.shape[1],1,1).transpose(0,1)
    batch_normalized_filtered_rir_envelope=batch_filtered_rir_envelope/normalizer_max
    return batch_normalized_filtered_rir_envelope

####### Batch Metrics #######

def batch_C80(batch_rir2, sample_rate, batch_origin, cut_severity=0.05):
    '''
    Backpropagatable C80 calculation.
    Soft (fractional) cut using sigmoid at 80 ms.
    '''
    #Get the vector of origin points.
    batch_eighty_ms=(0.080*sample_rate)*torch.ones_like(batch_origin, device=batch_origin.device) + batch_origin # in samples
    
    batch_before_80ms, batch_after_80ms = batch_cut_before_and_after_index(batch_rir2, batch_eighty_ms, cut_severity=cut_severity)

    lower_integral=torch.sum(batch_before_80ms, dim=1)
    top_integral=torch.sum(batch_after_80ms, dim=1)
    
    return 10*torch.log10((top_integral/(lower_integral + 1e-10))+torch.ones_like(lower_integral))

def batch_D(batch_rir2, sample_rate, batch_rir2_sum, batch_origin, cut_severity=0.05):
    '''
    Backpropagatable D calculation.
    Soft (fractional) cut using sigmoid at 50 ms.
    '''
    batch_fifty_ms=(0.080*sample_rate)*torch.ones_like(batch_origin, device=batch_origin.device) + batch_origin # in samples
    
    batch_before_50ms = batch_cut_before_and_after_index(batch_rir2, batch_fifty_ms, cut_severity=cut_severity, return_after=False)

    partial_integral=torch.sum(batch_before_50ms)
    full_integral = batch_rir2_sum

    return partial_integral/full_integral

def batch_center_time(batch_rir2, sample_rate, batch_rir2_sum, batch_origin, beta=10):
    '''
    Backpropagatable center_time calculation.
    Softplus for the origin stuff
    '''
    lower_integral=batch_rir2_sum

    softplus=torch.nn.Softplus(beta=beta) # Make it a bit less soft

    batch_abscisse=torch.arange(batch_rir2.shape[1], dtype=torch.float, device=batch_rir2.device).repeat(batch_origin.shape[0],1)
    batch_origin=batch_origin.repeat(batch_abscisse.shape[1], 1).transpose(0,1)

    batch_translated_abscisse=batch_abscisse-batch_origin
    batch_t=softplus(batch_translated_abscisse) # Using a translated softplus since origins are floats
    
    top_integral=torch.sum(batch_t*batch_rir2, dim=1) / sample_rate
    return top_integral/lower_integral

def batch_RT60(batch_rir2, sample_rate, batch_origin, epsilon=0.0005, plot=False, give_betas_for_plotting=False):
    '''
    Backpropagatable (?) RT60 calculation.
    Slow decay curve things?
    Does it even work??
    '''
    batch_size=batch_origin.shape[0]
    batch_decay_curve=torch.flip(torch.cumsum(torch.flip(batch_rir2, [1]), dim=1), [1]) # Cumulative sum from the end (using two flips)
    batch_thresholds=epsilon*batch_decay_curve[0]

    # Find when the curve has properly stopped decreasing : regressing further than that is pointless. These aren't torch operations.
    batch_tail_cut_index=[]
    for batch_idx in range(batch_size):
        tail_cut_index=-1
        while batch_decay_curve[batch_idx, tail_cut_index] < batch_thresholds[batch_idx]:
            tail_cut_index-=1
        tail_cut_index %= batch_decay_curve.shape[1]
        batch_tail_cut_index.append(tail_cut_index)

    batch_rt60=[]
    if give_betas_for_plotting: batch_betas=[]
    for idx in range(batch_size):
        int_origin=int(batch_origin[idx].item())
        tail_cut=batch_tail_cut_index[idx]
        try:
            shortened_decay_curve=batch_decay_curve[idx, int_origin:tail_cut] # cut off silent intro and cut off tails in the batch
            shortened_decay_curve=shortened_decay_curve/shortened_decay_curve[0] # Normalize to 1.0 HOPEFULLY
        except IndexError:
            print("WARNING! INDEX ERROR IN RT60 CALCULATION. Woops")
            print(int_origin,tail_cut)
            shortened_decay_curve=batch_decay_curve[idx,:] # Just don't cut intro and tails in batch
            shortened_decay_curve=shortened_decay_curve/torch.max(shortened_decay_curve) # Normalize to 1.0 brutally
        rt60max=torch.max(shortened_decay_curve.detach()).item()
        if rt60max != 1.0:
            print("WARNING! RT60 MAX IS NOT AT ORIGIN !! Woops")
            print("max : ",rt60max)
        log_decay_curve = -torch.log(torch.clamp(shortened_decay_curve, min=1e-10))

        # Get all the betas and average them
        times_samples=torch.arange(tail_cut-int_origin, device=batch_rir2.device)
        times_seconds = times_samples / sample_rate
        betas=log_decay_curve[1:]/times_seconds[1:]
        regressed_beta=torch.median(betas) # median is more robust to outliers!!
    
        rt60=6/(regressed_beta+1e-10)

        if plot:
            print("regressed beta",regressed_beta)

            plt.figure()
            plt.title("RT60 beta regression")
            plt.xlabel("betas")
            plt.ylabel("beta value")
            plt.ylim([-10,120])
            plt.scatter(np.arange(len(betas)),betas.cpu().numpy())
            plt.axhline(y=regressed_beta.item(), ls="dashdot", c="green", label="regressed beta")
            plt.legend()

            plt.figure()
            plt.title("RT60 Decay curve Approximation")
            plt.xlabel("Time in seconds")
            plt.plot(times_seconds.detach().cpu().numpy(),shortened_decay_curve.detach().cpu().numpy())
            plt.plot(times_seconds.detach().cpu().numpy(),torch.exp(-regressed_beta.item()*times_seconds.detach()).cpu().numpy(), label="regressed beta")
            plt.axvline(x=rt60.item(), ls="dashdot", c="black", label="rt60")
            plt.legend()

            plt.figure()
            plt.title("RT60 Decay curve Approximation (log scale)")
            plt.xlabel("Time in seconds")
            plt.plot(times_seconds.detach().cpu().numpy(),-log_decay_curve.detach().cpu().numpy())
            plt.plot(times_seconds.detach().cpu().numpy(),times_seconds.detach().cpu().numpy()*(-regressed_beta).item(), label="regressed beta")
            plt.axvline(x=rt60.item(), ls="dashdot", c="black", label="rt60")
            plt.legend()
            plt.show()

        batch_rt60.append(rt60)
        if give_betas_for_plotting: batch_betas.append(betas)

    batch_rt60=torch.stack(batch_rt60)
    
    if give_betas_for_plotting:
        torch.stack(batch_betas)
        return batch_rt60, batch_betas
    
    return batch_rt60

def batch_ms_env_diff(batch_rir1, batch_rir2,
                      filtering=False,
                      care_about_origin=True, batch_origin1=None, batch_origin2=None):
    '''
    Backpropagatable ms_env_diff calculation.
    Filtering is an option.
    Caring about the origin is an option.
    WARNING! CREATING FILTERS EACH TIME IS SLOW(?)
    '''
    batch_size=batch_rir1.shape[0]

    # Go to frequential domain
    batch_rir1_fft=torch.fft.rfft(batch_rir1)
    batch_rir2_fft=torch.fft.rfft(batch_rir2)
    assert(batch_rir1_fft.shape==batch_rir2_fft.shape)

    if filtering:
        # Create filters
        filters=[create_filter(batch_rir1_fft, 1, bw) for bw in [100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000]]
        filters=torch.stack(filters, dim=1).repeat(batch_size,1,1).to(batch_rir1_fft.device)
        assert(filters.shape[0]==batch_size)
        assert(filters.shape[1]==batch_rir1_fft.shape[1])
        assert(filters.shape[2]==len([100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000]))
        batch_rir1_filtered_envelopes=get_filtered_envelopes(batch_rir1_fft,filters)
        batch_rir2_filtered_envelopes=get_filtered_envelopes(batch_rir2_fft,filters)

        # Sum all filtered rirs, then normalize
        batch_rir1_sumfiltered_envelopes=torch.sum(batch_rir1_filtered_envelopes, dim=2)
        batch_rir2_sumfiltered_envelopes=torch.sum(batch_rir2_filtered_envelopes, dim=2)
        batch_rir1_sumfiltered_envelope=batch_rir1_sumfiltered_envelopes/(torch.max(batch_rir1_sumfiltered_envelopes, dim=1).values).repeat(batch_rir1_sumfiltered_envelopes.shape[1],1).transpose(0,1)
        batch_rir2_sumfiltered_envelope=batch_rir2_sumfiltered_envelopes/(torch.max(batch_rir2_sumfiltered_envelopes, dim=1).values).repeat(batch_rir2_sumfiltered_envelopes.shape[1],1).transpose(0,1)

        if care_about_origin:
            batch_rir1_sumfiltered_envelope, batch_rir2_sumfiltered_envelope = truncate_to_origin_and_pad(batch_rir1_sumfiltered_envelope, batch_rir2_sumfiltered_envelope, batch_origin1, batch_origin2)
        batch_ms_env_loss=torch.sum(torch.abs(batch_rir1_sumfiltered_envelope-batch_rir2_sumfiltered_envelope), dim=1)

        return batch_ms_env_loss
    
    else:
        batch_rir1_enveloppe = batch_get_envelopes_no_filtering(batch_rir1_fft)
        batch_rir2_enveloppe = batch_get_envelopes_no_filtering(batch_rir2_fft)
        if care_about_origin:
            batch_rir1_enveloppe, batch_rir2_enveloppe = truncate_to_origin_and_pad(batch_rir1_enveloppe, batch_rir2_enveloppe, batch_origin1, batch_origin2)
        batch_ms_env_loss=torch.sum(torch.abs(batch_rir1_enveloppe-batch_rir2_enveloppe), dim=1)
        return batch_ms_env_loss

####### Class #######

class RIRMetricsLoss(nn.Module):
    def __init__(self, sample_rate=48000, lambda_param={'mrstft': 1, 'd': 1, 'c80': 1, 'rt60':1, 'center_time': 1, 'ms_env': 1}, return_separate_losses=False,
                 mrstft_care_about_origin=True, ms_env_filtering=True, ms_env_care_about_origin=True, print_info=False):
        super().__init__()
        # Simulation parameters
        self.sample_rate=sample_rate

        self.lambda_param=lambda_param
        # an initial lambda multiplication that kind of does a pre-normalization of the variances between the different losses
        self.pre_lambdas={'mrstft': 0.6, 'd': 4, 'c80': 0.015, 'rt60':5, 'center_time': 100, 'ms_env': 0.04}

        # Options for mrstft
        self.mrstft=None
        self.mrstft_care_about_origin=mrstft_care_about_origin

        # Options for ms_env
        self.filters=None
        self.ms_env_filtering=ms_env_filtering
        self.ms_env_care_about_origin=ms_env_care_about_origin

        # Initialize losses
        self.which_losses=self.init_which_losses([])
        self.loss_dict={}
        for loss in self.which_losses:
            self.loss_dict[loss]=[]

        self.print_info=print_info
        self.return_separate_losses=return_separate_losses
    
    def init_which_losses(self, which_losses=[]):
        '''
        Initializes which losses will be used.
        If none are specified, losses will be chosen based on which lambda parameters are non-zero.
        '''
        if which_losses==[]:
            for loss in self.lambda_param.keys():
                if self.lambda_param[loss] != 0:
                    which_losses.append(loss)
        print("RIRMetrisLoss Initialized. Using losses : ", which_losses, end=' ')
        if ('mrstft' in which_losses):
            if self.mrstft_care_about_origin: print("with mrstft caring about origin", end=' ')
            else: print("", end=' ')
        if ('ms_env' in which_losses):
            if self.ms_env_filtering: print("", end='')
            else: print("without (Multi Filtered Envelope Sum) Filtering activated", end=' ')
        if ('ms_env' in which_losses):
            if self.ms_env_care_about_origin: print("", end=' ')
            else: print("without (Multi Filtered Envelope Sum) caring about origin", end=' ')
        print("")
        return(which_losses)

    def forward(self, shoebox_rir_batch : Union[List[torch.Tensor],torch.Tensor], shoebox_origin_batch : torch.Tensor,
                      label_rir_batch : Union[List[torch.Tensor],torch.Tensor], label_origin_batch : torch.Tensor):
        device=shoebox_origin_batch.device
        label_origin_batch=label_origin_batch.to(device)

        if isinstance(shoebox_rir_batch, list):
            assert(isinstance(shoebox_rir_batch[0], torch.Tensor))
            shoebox_rir_batch=torch.nn.utils.rnn.pad_sequence(shoebox_rir_batch, batch_first=True).to(device)

        if isinstance(label_rir_batch, list):
            assert(isinstance(label_rir_batch[0], torch.Tensor))
            label_rir_batch=torch.nn.utils.rnn.pad_sequence(label_rir_batch, batch_first=True).to(device)

        assert(len(shoebox_rir_batch)==len(label_rir_batch) == shoebox_origin_batch.shape[0] == label_origin_batch.shape[0])

        # Re-init loss_dict
        for loss in self.which_losses:
            self.loss_dict[loss]=[]

        # Calculate mrstft loss
        if 'mrstft' in self.which_losses:

            #init mrstft if not already done
            if self.mrstft==None:
                self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
                    # fft_sizes=[1024, 2048, 8192], # 48000 sample rate
                    # hop_sizes=[256, 512, 2048],
                    # win_lengths=[1024, 2048, 8192],
                    fft_sizes = [256, 512, 1024], # 16000 sample rate
                    hop_sizes = [64, 128, 256],
                    win_lengths = [256, 512, 1024],
                    scale="mel",
                    # n_bins=128,
                    n_bins=36,
                    sample_rate=self.sample_rate,
                    perceptual_weighting=True,
                    device= shoebox_rir_batch.device,
                )
            
            shoebox_rir_batch,label_rir_batch = pad_to_match_size(shoebox_rir_batch,label_rir_batch)
            if self.mrstft_care_about_origin: x, y = truncate_to_origin_and_pad(shoebox_rir_batch, label_rir_batch, shoebox_origin_batch, label_origin_batch)
            else: x, y = shoebox_rir_batch, label_rir_batch
            batch_mrstft_loss = self.mrstft( x[:,None,:], y[:,None,:] ) # Calculate batch_mrstft # Add a dimension for channels
            batch_mrstft_loss = self.pre_lambdas['mrstft'] * batch_mrstft_loss # pre lambda
            self.loss_dict['mrstft']=batch_mrstft_loss # Store
   
        # Precalculations
        if 'ms_env' in self.which_losses or 'rt60' in self.which_losses or 'd' in self.which_losses or \
            'c80' in self.which_losses or 'center_time' in self.which_losses:

            # Precalculate rir^2
            batch_input_rir2=torch.pow(shoebox_rir_batch,2)
            batch_label_rir2=torch.pow(label_rir_batch,2)
        
            if 'rt60' in self.which_losses or 'd' in self.which_losses or \
                'c80' in self.which_losses or 'center_time' in self.which_losses:

                # Precalculate sum(rir^2)
                batch_input_rir2_sum=torch.sum(batch_input_rir2, axis=1)
                batch_label_rir2_sum=torch.sum(batch_label_rir2, axis=1)

        # Calculate losses
        if 'c80' in self.which_losses:
            batch_input_c80 = batch_C80(batch_input_rir2, self.sample_rate, shoebox_origin_batch) # Calculate batch_C80
            batch_label_c80 = batch_C80(batch_label_rir2, self.sample_rate, label_origin_batch) # Calculate batch_C80

            batch_c80_loss = torch.abs(batch_input_c80-batch_label_c80) # Difference
            batch_c80_loss = self.pre_lambdas['c80'] * batch_c80_loss # pre lambda
            self.loss_dict['c80']=batch_c80_loss # Store

        if 'd' in self.which_losses:
            batch_input_d = batch_D(batch_input_rir2, self.sample_rate, batch_input_rir2_sum, shoebox_origin_batch) # Calculate batch_D
            batch_label_d = batch_D(batch_label_rir2, self.sample_rate, batch_label_rir2_sum, label_origin_batch) # Calculate batch_D

            batch_d_loss = torch.abs(batch_input_d-batch_label_d)  # Difference
            batch_d_loss = self.pre_lambdas['d']*batch_d_loss # pre lambda
            self.loss_dict['d']=batch_d_loss # Store

        if 'center_time' in self.which_losses:
            batch_input_c_t = batch_center_time(batch_input_rir2, self.sample_rate, batch_input_rir2_sum, shoebox_origin_batch) # Calculate batch_c_t
            batch_label_c_t = batch_center_time(batch_label_rir2, self.sample_rate, batch_label_rir2_sum, label_origin_batch) # Calculate batch_c_t

            batch_center_time_loss= torch.abs(batch_input_c_t-batch_label_c_t)  # Difference
            batch_center_time_loss=self.pre_lambdas['center_time']*batch_center_time_loss # pre lambda
            self.loss_dict['center_time']=batch_center_time_loss # Store

        if 'rt60' in self.which_losses:
            batch_input_rt60 = batch_RT60(batch_input_rir2, self.sample_rate, shoebox_origin_batch) # Calculate batch_RT60
            batch_label_rt60 = batch_RT60(batch_label_rir2, self.sample_rate, label_origin_batch) # Calculate batch_RT60
            
            batch_rt60_loss = torch.abs(batch_input_rt60- batch_label_rt60) # Difference
            batch_rt60_loss = self.pre_lambdas['rt60']*batch_rt60_loss # pre lambda
            self.loss_dict['rt60']=batch_rt60_loss # Store
            
        if 'ms_env' in self.which_losses:
            if self.ms_env_care_about_origin:
                batch_ms_env_loss=batch_ms_env_diff(batch_input_rir2, batch_label_rir2,
                                                    filtering=self.ms_env_filtering,
                                                    care_about_origin=True, batch_origin1=shoebox_origin_batch, batch_origin2=label_origin_batch)
            else:
                batch_ms_env_loss=batch_ms_env_diff(batch_input_rir2, batch_label_rir2,
                                                    filtering=self.ms_env_filtering,
                                                    care_about_origin=False)
            batch_ms_env_loss=self.pre_lambdas['ms_env']*batch_ms_env_loss # pre lambda
            self.loss_dict['ms_env']=batch_ms_env_loss # store
        
        # Average batch losses
        if self.print_info : print("Average batch losses : ", end="")
        for loss in self.which_losses:
            if loss == 'mrstft':
                    if self.print_info : print(loss, self.loss_dict[loss].item(), end=" ")
                    continue
            self.loss_dict[loss]=torch.mean(self.loss_dict[loss])
            if self.print_info : print(loss, self.loss_dict[loss].item(), end=" ")

        # Cleanup
        del shoebox_rir_batch, label_rir_batch, shoebox_origin_batch, label_origin_batch
        del batch_input_rir2, batch_label_rir2, batch_input_rir2_sum, batch_label_rir2_sum

        # Returns
        if self.return_separate_losses:
            return self.loss_dict
        else:    
            # Get total loss
            total_loss=torch.zeros(1, device=shoebox_rir_batch.device)
            for loss in self.which_losses:
                total_loss=total_loss + self.loss_dict[loss]*self.lambda_param[loss]

            if self.print_info : print("\ntotal_loss",total_loss.item())
            return total_loss
    
### PLOT FUNCTIONS ###
from typing import List
import numpy as np
from matplotlib.patches import Rectangle

def get_indexes(i,width):
    iy=i%width
    ix=i//width
    return ix, iy

def plot_rir_metrics(batch_rir1, batch_rir2, batch_origin1, batch_origin2, name_1="encoder", name_2="label"): # , addendum="", show=True
    '''
    Visualizes metrics for a single rir pair or the first element of a batch of rir pairs.
    Graph 1: Envelope Visualisation
    Graph 2: RT60 Visualisation
    Graph 3: C80, D, CENTER_TIME Visualisation
    Graph 4: Filtered Envelope Sum (ms_env) Visualisation
    Graph 5: Multi-Resolution STFT Visualisation
    Graph 6: Bar graphs between compared metrics.
    '''
    sample_rate=48000
    # Prepare inputs, manage batch or not batch
    batch_rir1=batch_rir1.detach()
    batch_rir2=batch_rir2.detach()
    if len(batch_rir1.shape)==2 and len(batch_rir2.shape)==2:
        rir1=batch_rir1[0,:]
        rir2=batch_rir2[0,:]
        batch_rir1=rir1[None,:]
        batch_rir2=rir2[None,:]
    elif len(batch_rir1.shape)==1 and len(batch_rir2.shape)==1:
        rir1=batch_rir1
        rir2=batch_rir2
        batch_rir1=batch_rir1[None,:]
        batch_rir2=batch_rir2[None,:]
    else:
        raise ValueError("batch_rir1 and batch_rir2 must be either have a batch dimension or not.")
    assert(isinstance(batch_origin1,torch.Tensor))
    assert(isinstance(batch_origin2,torch.Tensor))
    batch_origin1=batch_origin1.detach()
    batch_origin2=batch_origin2.detach()
    if batch_origin1.shape[0]>1 and batch_origin2.shape[0]>1:
        origin1=batch_origin1[0]
        origin2=batch_origin2[0]
        batch_origin1=torch.unsqueeze(origin1, dim=0)
        batch_origin2=torch.unsqueeze(origin2, dim=0)
    elif batch_origin1.shape[0]==1 and batch_origin2.shape[0]==1:
        origin1=batch_origin1
        origin2=batch_origin2
        batch_origin1=torch.unsqueeze(batch_origin1, dim=0)
        batch_origin2=torch.unsqueeze(batch_origin2, dim=0)
    else:
        raise("batch_origin1 and batch_origin2 must be either have a batch dimension or not.")

    # Precalculate rir^2 for metrics
    batch_rir1_2=torch.pow(batch_rir1,2)
    batch_rir2_2=torch.pow(batch_rir2,2)

    # Precalculate sum(rir^2) for metrics
    batch_rir1_2_sum=torch.sum(batch_rir1_2)
    batch_rir2_2_sum=torch.sum(batch_rir2_2)

    # Calculate metrics
    batch_rir1_c80 = batch_C80(batch_rir1_2, sample_rate, batch_origin1) # Calculate batch_C80
    batch_rir2_c80 = batch_C80(batch_rir2_2, sample_rate, batch_origin2) # Calculate batch_C80
    batch_rir1_d = batch_D(batch_rir1_2, sample_rate, batch_rir1_2_sum, batch_origin1) # Calculate batch_D
    batch_rir2_d = batch_D(batch_rir2_2, sample_rate, batch_rir2_2_sum, batch_origin2) # Calculate batch_D
    batch_rir1_c_t = batch_center_time(batch_rir1_2, sample_rate, batch_rir1_2_sum, batch_origin1)*sample_rate # Calculate batch_c_t
    batch_rir2_c_t = batch_center_time(batch_rir2_2, sample_rate, batch_rir2_2_sum, batch_origin2)*sample_rate # Calculate batch_c_t
    batch_rir1_rt60, batch_betas1 = batch_RT60(batch_rir1_2, sample_rate, batch_origin1, plot=False, give_betas_for_plotting=True) # Calculate batch_RT60
    batch_rir2_rt60, batch_betas2 = batch_RT60(batch_rir2_2, sample_rate, batch_origin2, plot=False, give_betas_for_plotting=True) # Calculate batch_RT60
    rir1_c80=batch_rir1_c80.item()
    rir2_c80=batch_rir2_c80.item()
    rir1_d=batch_rir1_d.item()
    rir2_d=batch_rir2_d.item()
    rir1_c_t=batch_rir1_c_t.item()/sample_rate
    rir2_c_t=batch_rir2_c_t.item()/sample_rate
    rir1_rt60=batch_rir1_rt60.item()
    rir2_rt60=batch_rir2_rt60.item()
    rir1_regressed_beta=6/rir1_rt60
    rir2_regressed_beta=6/rir2_rt60
    batch_betas1=batch_betas1[0]
    batch_betas2=batch_betas2[0]
    
    #############################################################################################################################################################################################
    ############## TEMPORAL #####################################################################################################################################################################
    #############################################################################################################################################################################################

    # Initialize plotting
    width=1
    height=2
    fig1, axs1 = plt.subplots(height,width, figsize=(13,9.3))
    fig1.suptitle("Envelope Visualisation")
    
    ################# Temporal synced origins
    i=0
    synced_rir1, synced_rir2 = truncate_to_origin_and_pad(batch_rir1, batch_rir2, batch_origin1, batch_origin2, truncate_less=10)
    synced_rir1, synced_rir2 = synced_rir1[0,:], synced_rir2[0,:]
    x_axis = np.arange(0, synced_rir1.shape[0], 1)/sample_rate

    axs1[i].set_title('RIR Envelopes')
    axs1[i].plot(x_axis,torch.abs(torch.real(synced_rir1)).cpu().numpy(), c="blue", alpha=0.7, label=name_1 + " rir")
    axs1[i].plot(x_axis,torch.abs(torch.real(synced_rir2)).cpu().numpy(), c="orange", alpha=0.5, label=name_2 + " rir")
    axs1[i].set_xlim(-250/sample_rate, 14000/sample_rate)
    axs1[i].set_ylim(0, 1)
    axs1[i].set_xlabel("time in s")
    axs1[i].set_ylabel("envelope (normalized)")

    axs1[i].axvline(x=rir1_c_t, ls="dashdot", c="blue", label=name_1 + " c_t")
    axs1[i].axvline(x=rir2_c_t, ls="dashdot", c="red", label=name_2 + " c_t")
    axs1[i].axvline(x=rir1_rt60, ls="dotted", c="blue", label=name_1 + " rt60")
    axs1[i].axvline(x=rir2_rt60, ls="dotted", c="red", label=name_2 + " rt60")
    axs1[i].axvline(x=0.05, ls="dashed", alpha=0.4, c="grey", label="50ms (D)")
    axs1[i].axvline(x=0.08, ls="dashed", alpha=0.4, c="grey", label="80ms (C80)")

    ################# Temporal synced origins difference
    i+=1
    axs1[i].set_title('RIR Envelopes Difference')
    axs1[i].plot(x_axis,torch.abs(torch.abs(torch.real(synced_rir1)) - torch.abs(torch.real(synced_rir2))).cpu().numpy(), alpha=0.7, color='green', label="Difference")
    axs1[i].set_xlim(-250/sample_rate, 14000/sample_rate)
    axs1[i].set_ylim(0, 1)
    axs1[i].set_xlabel("time in s")
    axs1[i].set_ylabel("envelope (normalized)")

        # Finalize plotting
    for _ in range(width*height):
        # ix, iy = get_indexes(_,width)
        axs1[_].legend()
        axs1[_].grid(True, ls=':', alpha=0.5)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.show()

    #############################################################################################################################################################################################
    ############## RT60 #########################################################################################################################################################################
    #############################################################################################################################################################################################

    # Initialize plotting
    width=2
    height=2
    fig4, axs4 = plt.subplots(height,width, figsize=(13,9.3))
    fig4.suptitle("RT60 Visualisation")
    
    ################# Temporal synced origins

    synced_rir1, synced_rir2 = truncate_to_origin_and_pad(batch_rir1, batch_rir2, batch_origin1, batch_origin2, truncate_less=10)
    synced_rir1, synced_rir2 = synced_rir1[0,:], synced_rir2[0,:]
    x_axis = np.arange(0, synced_rir1.shape[0], 1)/sample_rate

    synced_rir1_2, synced_rir2_2 = truncate_to_origin_and_pad(batch_rir1_2, batch_rir2_2, batch_origin1, batch_origin2, truncate_less=10)

    decay_curve1=torch.flip(torch.cumsum(torch.flip(synced_rir1_2, [1]), dim=1), [1])[0]
    decay_curve1=decay_curve1/decay_curve1[0] # normalize
    decay_curve2=torch.flip(torch.cumsum(torch.flip(synced_rir2_2, [1]), dim=1), [1])[0]
    decay_curve2=decay_curve2/decay_curve2[0] # normalize

    i=0
    ix, iy = get_indexes(i,width)
    axs4[ix,iy].set_title(name_1+" Approximated decay curve and RT60")
    
    axs4[ix,iy].plot(x_axis,torch.abs(torch.real(synced_rir1)).cpu().numpy(), c="darkblue", alpha=0.2, label=name_1 + " rir")
    axs4[ix,iy].plot(x_axis,decay_curve1.cpu().numpy(), c="blue", alpha=0.5, label=name_1 + " rir decay curve")
    axs4[ix,iy].plot(x_axis,np.exp(-rir1_regressed_beta*x_axis), ls="solid", c="blue", label=name_1 + " approx decay curve")
    axs4[ix,iy].axvline(x=rir1_rt60, ls="dotted", c="blue", label=name_1 + " rt60")

    axs4[ix,iy].set_xlim(-250/sample_rate, 12000/sample_rate)
    axs4[ix,iy].set_ylim(0, 1)
    axs4[ix,iy].set_xlabel("time in s")
    axs4[ix,iy].set_ylabel("envelope (normalized)")
    axs4[ix,iy].legend()
    axs4[ix,iy].grid(True, ls=':', alpha=0.5)

    i+=1
    ix, iy = get_indexes(i,width)
    axs4[ix,iy].set_title(name_1+" RT60 beta regression")
    axs4[ix,iy].set_xlabel("betas")
    axs4[ix,iy].set_ylabel("beta value")
    axs4[ix,iy].set_ylim([-10,120])
    axs4[ix,iy].scatter(np.arange(len(batch_betas1)),batch_betas1.cpu().numpy(), c='green',s=1, label="proposed betas")
    axs4[ix,iy].axhline(y=rir1_regressed_beta, ls="dashdot", c="green", label="median beta")
    axs4[ix,iy].legend()

    i+=1
    ix, iy = get_indexes(i,width)
    axs4[ix,iy].set_title(name_2+" Approximated decay curve and RT60")
    axs4[ix,iy].plot(x_axis,torch.abs(torch.real(synced_rir2)).cpu().numpy(), c="orange", alpha=0.2, label=name_2 + " rir")
    axs4[ix,iy].plot(x_axis,decay_curve2.cpu().numpy(), c="orange", alpha=0.5, label=name_2 + " rir decay curve")
    axs4[ix,iy].plot(x_axis,np.exp(-rir2_regressed_beta*x_axis), ls="solid", c="red", label=name_2 + " approx decay curve")
    axs4[ix,iy].axvline(x=rir2_rt60, ls="dotted", c="red", label=name_2 + " rt60")

    axs4[ix,iy].set_xlim(-250/sample_rate, 12000/sample_rate)
    axs4[ix,iy].set_ylim(0, 1)
    axs4[ix,iy].set_xlabel("time in s")
    axs4[ix,iy].set_ylabel("envelope (normalized)")
    axs4[ix,iy].legend()
    axs4[ix,iy].grid(True, ls=':', alpha=0.5)

    i+=1
    ix, iy = get_indexes(i,width)
    axs4[ix,iy].set_title(name_2+" RT60 beta regression")
    axs4[ix,iy].set_xlabel("betas")
    axs4[ix,iy].set_ylabel("beta value")
    axs4[ix,iy].set_ylim([-10,120])
    axs4[ix,iy].scatter(np.arange(len(batch_betas2)),batch_betas2.cpu().numpy(), c='green',s=1, label="proposed betas")
    axs4[ix,iy].axhline(y=rir2_regressed_beta, ls="dashdot", c="green", label="median beta")
    axs4[ix,iy].legend()
    
    
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig4.show()

    ###############################################################################################################################################################################################
    ############## C80, D, CENTER_TIME #####################################################################################################################################################################
    ###############################################################################################################################################################################################

    # Initialize plotting
    width=3
    height=2
    fig6, axs6 = plt.subplots(height,width, figsize=(17,9.3))
    fig6.suptitle("C80, D, CENTER_TIME Visualisation")

    index_50ms=int(0.05*sample_rate)
    index_80ms=int(0.08*sample_rate)
    plot_beginning=24/sample_rate

    i=0
    ix, iy = get_indexes(i,width)
    axs6[ix,iy].set_title(name_1 + " D visualisation")
    axs6[ix,iy].plot(x_axis[:index_50ms],
                    decay_curve1[:index_50ms].cpu().numpy()-np.ones(index_50ms)*decay_curve1[index_50ms].item(),
                    c="darkblue", alpha=1, label=name_1 + " rir decay curve before 50 ms")
    axs6[ix,iy].plot(x_axis[index_50ms:],
                    (decay_curve1[index_50ms:]).cpu().numpy(),
                    c="blue", alpha=1, label=name_1 + " rir decay curve after 50 ms")
    axs6[ix,iy].axvline(x=0.05, ls="dashed", alpha=0.4, c="black")
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 0), 1+plot_beginning, 1-decay_curve1[index_50ms].item(),
                                    alpha=0.2, facecolor = 'darkblue', fill=True))
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 1-decay_curve1[index_50ms].item()), 1+plot_beginning, decay_curve1[index_50ms].item(),
                                    alpha=0.1, facecolor = 'blue', fill=True))
    axs6[ix,iy].set_xlim(-plot_beginning, 12000/sample_rate)
    axs6[ix,iy].set_ylim(0, 1)
    axs6[ix,iy].set_xlabel("time in s")
    axs6[ix,iy].set_ylabel("decay curve (normalized)")
    axs6[ix,iy].axhline(y=1-decay_curve1[index_50ms].item(), ls="solid", alpha=1, c="black")
    axs6[ix,iy].text(0.07, max(min( (1-decay_curve1[index_50ms].item()), 0.9) , decay_curve1[index_50ms].item()), 'D = '+str(1-decay_curve1[index_50ms].item()),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
    # axs6[ix,iy].legend()
    axs6[ix,iy].grid(True, ls=':', alpha=0.5)
    current_ticks = list(axs6[ix,iy].get_yticks())
    current_ticks.append(1-decay_curve1[index_50ms].item())
    # current_ticks.append(decay_curve1[index_50ms].item())
    axs6[ix,iy].set_yticks(current_ticks)
    

    i=3
    ix, iy = get_indexes(i,width)
    axs6[ix,iy].set_title(name_2 + " D visualisation")
    axs6[ix,iy].plot(x_axis[:index_50ms],
                    decay_curve2[:index_50ms].cpu().numpy()-np.ones(index_50ms)*decay_curve2[index_50ms].item(),
                    c="darkorange", alpha=1, label=name_2 + " rir decay curve before 50 ms")
    axs6[ix,iy].plot(x_axis[index_50ms:],
                    (decay_curve2[index_50ms:]).cpu().numpy(),
                    c="orange", alpha=1, label=name_2 + " rir decay curve after 50 ms")
    axs6[ix,iy].axvline(x=0.05, ls="dashed", alpha=0.4, c="black")
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 0), 1+plot_beginning, 1-decay_curve2[index_50ms].item(),
                                    alpha=0.3, facecolor = 'orange', fill=True))
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 1-decay_curve2[index_50ms].item()), 1+plot_beginning, decay_curve2[index_50ms].item(),
                                    alpha=0.1, facecolor = 'darkorange', fill=True))
    axs6[ix,iy].set_xlim(-plot_beginning, 12000/sample_rate)
    axs6[ix,iy].set_ylim(0, 1)
    axs6[ix,iy].set_xlabel("time in s")
    axs6[ix,iy].set_ylabel("decay curve (normalized)")
    axs6[ix,iy].axhline(y=1-decay_curve2[index_50ms].item(), ls="solid", alpha=1, c="black")
    axs6[ix,iy].text(0.07, max(min( (1-decay_curve2[index_50ms].item()), 0.9) , decay_curve2[index_50ms].item()), 'D = '+str(1-decay_curve2[index_50ms].item()),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
    # axs6[ix,iy].legend()
    axs6[ix,iy].grid(True, ls=':', alpha=0.5)
    current_ticks = list(axs6[ix,iy].get_yticks())
    current_ticks.append(1-decay_curve2[index_50ms].item())
    # current_ticks.append(decay_curve2[index_50ms].item())
    axs6[ix,iy].set_yticks(current_ticks)


    i=1
    ix, iy = get_indexes(i,width)
    axs6[ix,iy].set_title(name_1 + " C80 visualisation")
    axs6[ix,iy].plot(x_axis[:index_80ms],
                    decay_curve1[:index_80ms].cpu().numpy()-np.ones(index_80ms)*decay_curve1[index_80ms].item(),
                    c="darkblue", alpha=1, label=name_1 + " rir decay curve before 80 ms")
    axs6[ix,iy].plot(x_axis[index_80ms:],
                    (decay_curve1[index_80ms:]).cpu().numpy(),
                    c="blue", alpha=1, label=name_1 + " rir decay curve after 80 ms")
    axs6[ix,iy].axvline(x=0.08, ls="dashed", alpha=0.4, c="black")
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 0), 0.08+plot_beginning, 1-decay_curve1[index_80ms].item(),
                                    alpha=0.2, facecolor = 'darkblue', fill=True))
    axs6[ix,iy].add_patch(Rectangle((0.08, 0), 12000/sample_rate, decay_curve1[index_80ms].item(),
                                    alpha=0.1, facecolor = 'blue', fill=True))
    axs6[ix,iy].add_patch(Rectangle((0.08, decay_curve1[index_80ms].item()), (12000/sample_rate)-0.08, 1-decay_curve1[index_80ms].item()*2,
                                    alpha=0.3, facecolor = 'green', fill=True))
    axs6[ix,iy].add_patch(Rectangle((0.08, decay_curve1[index_80ms].item()), (12000/sample_rate)-0.08, 1-decay_curve1[index_80ms].item()*2,
                                    alpha=1, fill=False, lw=1.5, ls='solid', edgecolor='black'))
    axs6[ix,iy].set_ylim(0, 1)
    axs6[ix,iy].set_xlabel("time in s")
    axs6[ix,iy].set_ylabel("decay curve (normalized)")
    # axs6[ix,iy].legend()
    axs6[ix,iy].grid(True, ls=':', alpha=0.5)
    c80=(1-decay_curve1[index_80ms].item())/decay_curve1[index_80ms].item()
    axs6[ix,iy].text(0.1, decay_curve1[index_80ms].item() + (1-decay_curve1[index_80ms].item())/2 , 'C80 = '+str(c80),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
    current_ticks = list(axs6[ix,iy].get_yticks())
    current_ticks.append(1-decay_curve1[index_80ms].item())
    current_ticks.append(decay_curve1[index_80ms].item())
    axs6[ix,iy].set_yticks(current_ticks)
    # current_ticks = list(axs6[ix,iy].get_xticks())
    # current_ticks.append(0.08)
    # axs6[ix,iy].set_xticks(current_ticks)
    axs6[ix,iy].set_xlim(-plot_beginning, 12000/sample_rate)

    i=4
    ix, iy = get_indexes(i,width)
    axs6[ix,iy].set_title(name_2 + " C80 visualisation")
    axs6[ix,iy].plot(x_axis[:index_80ms],
                    decay_curve2[:index_80ms].cpu().numpy()-np.ones(index_80ms)*decay_curve2[index_80ms].item(),
                    c="darkorange", alpha=1, label=name_2 + " rir decay curve before 80 ms")
    axs6[ix,iy].plot(x_axis[index_80ms:],
                    (decay_curve2[index_80ms:]).cpu().numpy(),
                    c="orange", alpha=1, label=name_2 + " rir decay curve after 80 ms")
    axs6[ix,iy].axvline(x=0.08, ls="dashed", alpha=0.4, c="black")
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 0), 0.08+plot_beginning, 1-decay_curve2[index_80ms].item(),
                                    alpha=0.3, facecolor = 'orange', fill=True))
    axs6[ix,iy].add_patch(Rectangle((0.08, 0), 12000/sample_rate, decay_curve2[index_80ms].item(),
                                    alpha=0.1, facecolor = 'darkorange', fill=True))
    axs6[ix,iy].add_patch(Rectangle((0.08, decay_curve2[index_80ms].item()), (12000/sample_rate)-0.08, 1-decay_curve2[index_80ms].item()*2,
                                    alpha=0.3, facecolor = 'green', fill=True))
    axs6[ix,iy].add_patch(Rectangle((0.08, decay_curve2[index_80ms].item()), (12000/sample_rate)-0.08, 1-decay_curve2[index_80ms].item()*2,
                                    alpha=1, fill=False, lw=1.5, ls='solid', edgecolor='black'))
    axs6[ix,iy].set_ylim(0, 1)
    axs6[ix,iy].set_xlabel("time in s")
    axs6[ix,iy].set_ylabel("decay curve (normalized)")
    # axs6[ix,iy].legend()
    axs6[ix,iy].grid(True, ls=':', alpha=0.5)
    c80=(1-decay_curve2[index_80ms].item())/decay_curve2[index_80ms].item()
    axs6[ix,iy].text(0.1, decay_curve2[index_80ms].item() + (1-decay_curve2[index_80ms].item())/2 , 'C80 = '+str(c80),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
    current_ticks = list(axs6[ix,iy].get_yticks())
    current_ticks.append(1-decay_curve2[index_80ms].item())
    current_ticks.append(decay_curve2[index_80ms].item())
    axs6[ix,iy].set_yticks(current_ticks)
    # current_ticks = list(axs6[ix,iy].get_xticks())
    # current_ticks.append(0.08)
    # axs6[ix,iy].set_xticks(current_ticks)
    axs6[ix,iy].set_xlim(-plot_beginning, 12000/sample_rate)


    i=2
    ix, iy = get_indexes(i,width)
    axs6[ix,iy].set_title(name_1 + " center_time visualisation")
    index_center=int(rir1_c_t*sample_rate)
    axs6[ix,iy].plot(x_axis[:index_center+1],
                     torch.abs(torch.real(synced_rir1[:index_center+1])).cpu().numpy(),
                     c="darkblue", alpha=0.2, label=name_1 + " rir before center_time")
    axs6[ix,iy].plot(x_axis[index_center:],
                     torch.abs(torch.real(synced_rir1[index_center:])).cpu().numpy(),
                     c="blue", alpha=0.2, label=name_1 + " rir after center_time")
    axs6[ix,iy].plot(x_axis[:index_center+1],
                    decay_curve1[:index_center+1].cpu().numpy(),
                    c="darkblue", alpha=1, label=name_1 + " rir decay curve before center_time")
    axs6[ix,iy].plot(x_axis[index_center:],
                    decay_curve1[index_center:].cpu().numpy(),
                    c="blue", alpha=1, label=name_1 + " rir decay curve after center_time")
    axs6[ix,iy].set_xlim(-plot_beginning, 12000/sample_rate)
    axs6[ix,iy].set_ylim(0, 1)
    axs6[ix,iy].set_xlabel("time in s")
    axs6[ix,iy].set_ylabel("decay curve (normalized)")
    axs6[ix,iy].grid(True, ls=':', alpha=0.5)
    axs6[ix,iy].axvline(x=rir1_c_t, ls="solid", c="black", label=name_1 + " TRUE c_t")
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 0), plot_beginning+rir1_c_t, 1,
                                    alpha=0.2, facecolor = 'darkblue', fill=True))
    axs6[ix,iy].add_patch(Rectangle((rir1_c_t, 0), 1-rir1_c_t, 1,
                                    alpha=0.1, facecolor = 'blue', fill=True))
    axs6[ix,iy].text(max(min(rir1_c_t-0.006,0.1),0.002), 0.7, 'c_t = '+str(rir1_c_t),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
    current_ticks = list(axs6[ix,iy].get_yticks())
    current_ticks.append(0.5)
    axs6[ix,iy].set_yticks(current_ticks)
    # current_ticks = list(axs6[ix,iy].get_xticks())
    # current_ticks.append(rir1_c_t)
    # axs6[ix,iy].set_xticks(current_ticks)

    i=5
    ix, iy = get_indexes(i,width)
    axs6[ix,iy].set_title(name_2 + " center_time visualisation")
    index_center=int(rir2_c_t*sample_rate)
    axs6[ix,iy].plot(x_axis[:index_center+1],
                     torch.abs(torch.real(synced_rir2[:index_center+1])).cpu().numpy(),
                     c="darkorange", alpha=0.2, label=name_2 + " rir before center_time")
    axs6[ix,iy].plot(x_axis[index_center:],
                     torch.abs(torch.real(synced_rir2[index_center:])).cpu().numpy(),
                     c="orange", alpha=0.2, label=name_2 + " rir after center_time")
    axs6[ix,iy].plot(x_axis[:index_center+1],
                    decay_curve2[:index_center+1].cpu().numpy(),
                    c="darkorange", alpha=1, label=name_2 + " rir decay curve before center_time")
    axs6[ix,iy].plot(x_axis[index_center:],
                    decay_curve2[index_center:].cpu().numpy(),
                    c="orange", alpha=1, label=name_2 + " rir decay curve before center_time")
    axs6[ix,iy].set_xlim(-plot_beginning, 12000/sample_rate)
    axs6[ix,iy].set_ylim(0, 1)
    axs6[ix,iy].set_xlabel("time in s")
    axs6[ix,iy].set_ylabel("decay curve (normalized)")
    axs6[ix,iy].grid(True, ls=':', alpha=0.5)
    axs6[ix,iy].axvline(x=rir2_c_t, ls="solid", c="black", label=name_2 + " TRUE c_t")
    axs6[ix,iy].add_patch(Rectangle((-plot_beginning, 0), plot_beginning+rir2_c_t, 1,
                                    alpha=0.3, facecolor = 'orange', fill=True))
    axs6[ix,iy].add_patch(Rectangle((rir2_c_t, 0), 1-rir2_c_t, 1,
                                    alpha=0.1, facecolor = 'darkorange', fill=True))
    axs6[ix,iy].text(max(min(rir2_c_t-0.006,0.1),0.002), 0.7, 'c_t = '+str(rir2_c_t),
                     bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
    current_ticks = list(axs6[ix,iy].get_yticks())
    current_ticks.append(0.5)
    axs6[ix,iy].set_yticks(current_ticks)
    # current_ticks = list(axs6[ix,iy].get_xticks())
    # current_ticks.append(rir2_c_t)
    # axs6[ix,iy].set_xticks(current_ticks)

    ###############################################################################################################################################################################################
    ############## MS_ENV #####################################################################################################################################################################
    ###############################################################################################################################################################################################
    
    # Initialize plotting
    width=3
    height=3
    fig3, axs3 = plt.subplots(height,width, figsize=(19,9.8)) #figsize=(19,9.8))
    fig3.suptitle("Filtered Envelope Sum (ms_env) Visualisation")

    batch_rir1_fft=torch.fft.rfft(batch_rir1)
    batch_rir2_fft=torch.fft.rfft(batch_rir2)
    assert(batch_rir1_fft.shape==batch_rir2_fft.shape)

    # og_filters=[100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000]
    og_filters=[100, 300, 500, 1000, 2000, 4000, 10000]
    filters=[create_filter(batch_rir1_fft[0], 1, bw) for bw in og_filters]
    filters=torch.stack(filters, dim=1).repeat(1,1,1) #add a batch_size dimension

    assert(filters.shape[0]==1)
    assert(filters.shape[1]==batch_rir1_fft.shape[-1])
    assert(filters.shape[2]==len(og_filters))
    rir1_filtered_envelopes=get_filtered_envelopes(batch_rir1_fft.cpu(),filters.cpu())
    rir2_filtered_envelopes=get_filtered_envelopes(batch_rir2_fft.cpu(),filters.cpu())

    rir1_filtered_envelopes, rir2_filtered_envelopes = rir1_filtered_envelopes[0,:,:], rir2_filtered_envelopes[0,:,:] # take out batch_size dimension

    # Each ms_env filter
    i= -1
    for filter_idx in range(filters.shape[2]):
        i+=1
        ix, iy = get_indexes(i,width)
        rir1_env, rir2_env = truncate_to_origin_and_pad(rir1_filtered_envelopes[None,:,filter_idx], rir2_filtered_envelopes[None,:,filter_idx], batch_origin1, batch_origin2)
        rir1_env, rir2_env = rir1_env[0,:], rir2_env[0,:] # take out batch_size dimension
        x_axis = np.arange(0, rir1_env.shape[0], 1)/sample_rate

        axs3[ix,iy].set_title('ms_env Individual Filter With f0 :'+ str(og_filters[filter_idx]))
        axs3[ix,iy].plot(x_axis ,rir1_env.cpu().numpy(), c="blue", alpha=0.7, label=name_1 + " filtered env")
        axs3[ix,iy].plot(x_axis, rir2_env.cpu().numpy(), c="orange", alpha=0.5, label=name_2 + " filtered env")
        axs3[ix,iy].legend()
        axs3[ix,iy].grid(True, ls=':', alpha=0.5)
        axs3[ix,iy].set_xlabel("time in s")
        axs3[ix,iy].set_ylabel("envelope (normalized)")
        axs3[ix,iy].set_ylim(0, 1)
       
        # axs3[ix,iy].set_title('ms_env filter cutoff :'+ str(og_filters[filter_idx]) + " difference")
        # axs3[ix,iy].plot(torch.abs(rir1_env-rir2_env).cpu().numpy(), label="difference")
        # axs3[ix, iy].legend()
        # axs3[ix, iy].grid(True, ls=':', alpha=0.5)
        # i+=1
        # ix, iy = get_indexes(i,width)

    # Sum of all filtered rirs
    rir1_sumfiltered_envelope=torch.sum(rir1_filtered_envelopes, dim=1)
    rir2_sumfiltered_envelope=torch.sum(rir2_filtered_envelopes, dim=1)
    rir1_sumfiltered_envelope=rir1_sumfiltered_envelope/torch.max(rir1_sumfiltered_envelope)
    rir2_sumfiltered_envelope=rir2_sumfiltered_envelope/torch.max(rir2_sumfiltered_envelope)

    rir1_sumenv, rir2_sumenv = truncate_to_origin_and_pad(rir1_sumfiltered_envelope[None,:], rir2_sumfiltered_envelope[None,:], batch_origin1, batch_origin2)
    rir1_sumenv, rir2_sumenv = rir1_sumenv[0,:], rir2_sumenv[0,:] # take out batch_size dimension
    x_axis = np.arange(0, rir1_sumenv.shape[0], 1)/sample_rate
    
    i+=1
    ix, iy = get_indexes(i,width)
    axs3[ix,iy].set_title('ms_env Sum Of All Filters')
    axs3[ix,iy].plot(x_axis,rir1_sumenv.cpu().numpy(),  c="blue", alpha=0.7, label=name_1 + " sum of filtered env")
    axs3[ix,iy].plot(x_axis,rir2_sumenv.cpu().numpy(), c="orange", alpha=0.5, label=name_2 + " sum of filtered env")
    axs3[ix,iy].legend()
    axs3[ix,iy].grid(True, ls=':', alpha=0.5)
    axs3[ix,iy].set_xlabel("time in s")
    axs3[ix,iy].set_ylabel("envelope (normalized)")
    axs3[ix,iy].set_ylim(0, 1)

    i+=1
    ix, iy = get_indexes(i,width)
    axs3[ix,iy].set_title('ms_env Sum Of All Filters Difference')
    axs3[ix,iy].plot(x_axis, torch.abs(rir1_sumenv-rir2_sumenv).cpu().numpy(), c="green", alpha=0.7, label="difference")
    axs3[ix,iy].legend()
    axs3[ix,iy].grid(True, ls=':', alpha=0.5)
    axs3[ix,iy].set_xlabel("time in s")
    axs3[ix,iy].set_ylabel("envelope (normalized)")
    axs3[ix,iy].set_ylim(0, 1)

    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.show()

    ###############################################################################################################################################################################################
    ############## MRSTFT ########################################################################################################################################################################
    ###############################################################################################################################################################################################

    # stft difference plot between two rirs
    import scipy.signal as signal
    width=2
    height=2
    fig5, axs5 = plt.subplots(height,width, figsize=(17,9.8)) #figsize=(19,9.8))
    fig5.suptitle("Multi-Resolution STFT Visualisation")

    synced_rir1, synced_rir2 = truncate_to_origin_and_pad(batch_rir1, batch_rir2, batch_origin1, batch_origin2, truncate_less=10)
    synced_rir1, synced_rir2 = synced_rir1[0,:], synced_rir2[0,:]
    x_axis = np.arange(0, synced_rir1.shape[0], 1)/sample_rate

    i=0
    ix, iy = get_indexes(i,width)
    axs5[ix,iy].set_title('RIR Envelopes')
    axs5[ix,iy].plot(x_axis,torch.abs(torch.real(synced_rir1)).cpu().numpy(), c="blue", alpha=0.7, label=name_1 + " rir")
    axs5[ix,iy].plot(x_axis,torch.abs(torch.real(synced_rir2)).cpu().numpy(), c="orange", alpha=0.5, label=name_2 + " rir")
    axs5[ix,iy].set_xlim(-250/sample_rate, 14000/sample_rate)
    axs5[ix,iy].set_ylim(0, 1)
    axs5[ix,iy].set_xlabel("time in s")
    axs5[ix,iy].set_ylabel("envelope (normalized)")
    axs5[ix,iy].legend()
    axs5[ix,iy].grid(True, ls=':', alpha=0.5)

    fft_sizes: List[int] = [1024, 2048, 512]
    hop_sizes: List[int] = [120, 240, 50]
    win_lengths: List[int] = [600, 1200, 240]

    for stft_param_index in range(len(fft_sizes)):
        stfts = [torch.stft(rir, fft_sizes[stft_param_index], hop_sizes[stft_param_index], win_lengths[stft_param_index], torch.hann_window(win_lengths[stft_param_index]).to(rir.device), return_complex=True)
                 for rir in [synced_rir1,synced_rir2]]
        
        magnitudes = [torch.sqrt(torch.clamp((stfts[rir].real**2) + (stfts[rir].imag**2), min=1e-8))
                      for rir in range(len([rir1,rir2]))]
        
        i+=1
        ix, iy = get_indexes(i,width)
        axs5[ix,iy].set_title('STFT Difference (fft_size ' + str(fft_sizes[stft_param_index]) + ' hop_size ' + str(hop_sizes[stft_param_index]) + ' win_length ' + str(win_lengths[stft_param_index]) + ')')
        quadmesh = axs5[ix,iy].pcolormesh((torch.abs(magnitudes[0]-magnitudes[1])).cpu().numpy(), cmap='inferno')
        # Add the colorbar for this specific subplot
        cbar=fig5.colorbar(quadmesh, ax=axs5[ix,iy])
        cbar.set_label("Magnitude Difference")
        axs5[ix,iy].set_xlabel("time in s")
        axs5[ix,iy].set_ylabel("frequency in Hz")
        # set the ticks to be in ms and Hz instead of just data indices
        def formatter_function_generator(product_param):
            def formatter_function(x, pos):
                return(product_param*x)
            return formatter_function
        axs5[ix,iy].xaxis.set_major_locator(plt.MultipleLocator(int(magnitudes[0].shape[1]/10)))
        axs5[ix,iy].xaxis.set_major_formatter(plt.FuncFormatter(formatter_function_generator(hop_sizes[stft_param_index]/sample_rate)))
        axs5[ix,iy].set_yticklabels(np.round(np.linspace(0, sample_rate/2, 6)))
        axs5[ix,iy].set_xlim(0, 14000/hop_sizes[stft_param_index])

    fig5.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig5.show()

    ###############################################################################################################################################################################################
    ############## BAR GRAPHS #####################################################################################################################################################################
    ###############################################################################################################################################################################################

    # C80 and D and RT60 and center_time bar graphs
    fig2, axs2 = plt.subplots(1,4,figsize=(9,4))
    fig2.suptitle("Comparative metrics")
    barWidth = 0.2
    x1 = 0.25
    x2 = 0.75
    y1 = [rir1_c80, rir1_d, rir1_rt60, rir1_c_t]
    y2 = [rir2_c80, rir2_d, rir2_rt60, rir2_c_t]
    metric_list=['C80', 'D', 'RT60', 'center_time']
    for metric in range(len(metric_list)):
        axs2[metric].set_title(metric_list[metric])
        axs2[metric].bar(x1, y1[metric], color ='blue', width = barWidth,
            edgecolor ='grey')
        axs2[metric].bar(x2, y2[metric], color ='orange', width = barWidth,
            edgecolor ='grey')
        
        if metric==0: axs2[metric].set_ylabel('ratio of energy before and after 80ms')
        if metric==1:
            axs2[metric].set_ylim(0, 1.05)
            axs2[metric].set_yticks([0,0.2,0.4,0.6,0.8,1])
            axs2[metric].set_yticklabels(["0%","20%","40%","60%","80%","100%"])
            axs2[metric].set_ylabel('percent of energy wthin 50ms of peak')
            axs2[metric].axhline(y=1, ls="dotted", c="black")
        if metric==3 or metric==2: axs2[metric].set_ylabel('time in s')

        axs2[metric].set_xlim(0, 1)
        axs2[metric].set_xticks([x1, x2])
        axs2[metric].set_xticklabels([name_1, name_2])
        axs2[metric].grid(True, ls=':', alpha=0.5)
    
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.show()

    plt.show()
    