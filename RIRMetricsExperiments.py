import torch
# from torchaudio.prototype.functional import simulate_rir_ism
from compute_rir import simulate_rir_ism
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from scipy.optimize import curve_fit
import math
import auraloss
from scipy.special import logit
from scipy.signal import find_peaks
from time import time
import pandas as pd
from LKLogger import LKLogger
import torch.nn as nn

# room generating

def generate_random_shoebox_input(max_room_dimensions=np.array([20,20,20]), min_percentile=0.4):
    '''Room generation function'''
    assert len(max_room_dimensions)==3
    room_dimensions=np.multiply(np.random.rand(3)*(1-min_percentile)+np.ones(3)*min_percentile,max_room_dimensions)
    mic_position=np.multiply(np.random.rand(3),room_dimensions)
    source_position=np.multiply(np.random.rand(3),room_dimensions)
    return room_dimensions, mic_position, source_position

# number manipulation

def fl_p(number, precision=4):
    '''
    returns a float with only a specific number of numbers after .
    '''
    exponent=10**precision
    return(int(number*exponent)/exponent)

def unlog_unlogit(log_room_dimensions, inv_sig_mic_position, inv_sig_source_position):
    room_dimensions = torch.exp(log_room_dimensions)
    mic_position = torch.sigmoid(inv_sig_mic_position)*room_dimensions
    source_position = torch.sigmoid(inv_sig_source_position)*room_dimensions
    return(room_dimensions, mic_position, source_position)

def cut_before_and_after_index(tensor, index=0, cut_severity=1.0):
    '''
    Cuts a tensor before and after a specific index.
    '''
    boom=torch.arange(len(tensor),device=tensor.device)
    cut_after_index=torch.sigmoid((boom-index)*cut_severity)
    
    cut_before_index=1-cut_after_index
    return tensor*cut_before_index, tensor*cut_after_index

# get raw rirs

def get_pytorch_rir(room_dimensions,mic_position,source_position,sample_rate, max_order=10):
    '''Forward pass through my pytorch ism implementation'''
    torch_absorption=[[0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]]
    torch_rir = simulate_rir_ism(
        room_dimensions,
        mic_position,
        source_position[None,:],
        max_order=max_order,
        absorption=torch_absorption,
        sample_rate=float(sample_rate),
    )
    return torch_rir

def get_pyroom_rir(room_dimensions,mic_position,source_position,sample_rate, max_order=10):
    '''Forward pass through the pyroom ism implementation'''
    room_dimensions_pyroom = room_dimensions
    mic_array_pyroom = np.array(mic_position)[None,:].T.astype(float) # np.array([[5, 4, 1.5]]).T
    source_position_pyroom = source_position
    wall_material = pra.Material(energy_absorption=0.3, scattering=0.0)
    room = pra.ShoeBox(room_dimensions_pyroom, materials=wall_material, fs=float(sample_rate), max_order=max_order)
    room.add_source(source_position_pyroom)
    room.add_microphone_array(mic_array_pyroom)
    room.compute_rir()
    pyroom_rir = room.rir[0][0]
    return pyroom_rir

# rir manipulation

def add_zero_buffer(tensor, num_zeroes):
    if len(tensor.shape) == 1:
        padding = torch.zeros(num_zeroes, dtype=tensor.dtype, device=tensor.device)
    elif len(tensor.shape) == 2:
        padding = torch.zeros(tensor.size()[0], num_zeroes, dtype=tensor.dtype, device=tensor.device)
    else:
        print("tensor has too many dimensions")
        print("tensor shape", tensor.shape)
        print("tensor dim", len(tensor.shape))
        raise ValueError
    padded_tensor = torch.cat((padding, tensor), dim=-1)
    return padded_tensor

def pad_to_match_size(tensor1, tensor2):
    
    if tensor1.shape[-1] > tensor2.shape[-1]:
        pad_length=tensor1.shape[-1]-tensor2.shape[-1]
        # Apply zero-padding to tensor2
        tensor2_padded = torch.nn.functional.pad(tensor2, (0, pad_length))
        return tensor1, tensor2_padded
    else:
        pad_length=tensor2.shape[-1]-tensor1.shape[-1]
        # Apply zero-padding to tensor1
        tensor1_padded = torch.nn.functional.pad(tensor1, (0, pad_length))
        return tensor1_padded, tensor2

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
    return filter.to(tensor.device)

def make_both_rir_comparable(torch_rir, pyroom_rir): #, torch_distance, sample_rate, sound_speed):
    '''
    process the two rir so they can be compared properly.
    Order matters here because torch_rir needs an initial buffer, which pyroom already has
    Both RIRs are:
    - zero padded to the same length
    - filtered in the frequency domain
    - normalized to 1.0

    Works with batches of rirs as well
    '''

    torch_rir = add_zero_buffer(torch_rir, 40)
    # if pyroom_rir is not already a torch tensor, convert it
    if type(pyroom_rir) != torch.Tensor:
        pyroom_rir=torch.tensor(pyroom_rir,dtype=torch.float32, device=torch_rir.device)
    torch_rir, pyroom_rir=pad_to_match_size(torch_rir,pyroom_rir)

    # Filter out extremely high frequencies with a filter in the frequency domain.
    torch_rir_fft=torch.fft.rfft(torch_rir)
    pyroom_rir_fft=torch.fft.rfft(pyroom_rir)
    filter=create_filter(torch_rir_fft, 11000,500)
    filtered_torch_rir_fft=torch_rir_fft*filter
    filtered_pyroom_rir_fft=pyroom_rir_fft*filter
    torch_rir=torch.fft.irfft(filtered_torch_rir_fft)
    pyroom_rir=torch.fft.irfft(filtered_pyroom_rir_fft)

    #  Normalize rir to 1.0
    torch_rir=torch_rir/(torch.max(torch.abs(torch_rir),dim=1).values.repeat(torch_rir.shape[1],1).T)
    pyroom_rir=pyroom_rir/(torch.max(torch.abs(pyroom_rir),dim=1).values.repeat(torch_rir.shape[1],1).T)

    return torch_rir, pyroom_rir

def plot_rir(torch_rir, pyroom_rir, addendum="", show=True, torch_origin=0, pyroom_origin=0):
    torch_rir_np=torch_rir.detach().cpu().numpy()
    pyroom_rir_np=pyroom_rir.detach().cpu().numpy()

    torch_fft=torch.fft.rfft(torch_rir)
    pyroom_fft=torch.fft.rfft(pyroom_rir)
    filters=[create_filter(torch_fft, 1, bw) for bw in [100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000]]

    fig1, axs = plt.subplots(2,2)
    fig1.suptitle("Pytorch VS Pyroom "+ addendum)
    
    axs[0,0].set_title('Temporal')
    axs[0,0].plot(np.real(torch_rir_np), alpha=0.5, label='pytorch')
    axs[0,0].plot(np.real(pyroom_rir_np), alpha=0.5, label='pyroom')
    
    if torch_origin != 0 : axs[0,0].axvline(x=torch_origin.detach().cpu().numpy(), c="mediumblue",label='torch_origin')
    if pyroom_origin !=0 : axs[0,0].axvline(x=pyroom_origin, c="darkorange",label='pyroom_origin')
    
    axs[0,0].plot(get_envelope_no_filtering(torch_fft.detach()).cpu().numpy(), alpha=0.5, label='TORCH ENVELOPE')
    axs[0,0].plot(get_envelope_no_filtering(pyroom_fft.detach()).cpu().numpy(), alpha=0.5, label='ROOM ENVELOPE')
    # axs[0,0].plot(sum_and_normalize_envelopes(get_filtered_envelopes(torch_fft.detach(),filters=filters)).numpy(), alpha=0.5, label='TORCH ENVELOPE FILTERED')
    # axs[0,0].plot(sum_and_normalize_envelopes(get_filtered_envelopes(pyroom_fft.detach(),filters=filters)).numpy(), alpha=0.5, label='ROOM ENVELOPE FILTERED')
    axs[0,0].legend()
    
    axs[1,0].set_title('Temporal Difference')
    axs[1,0].plot(np.abs(pyroom_rir_np-torch_rir_np), alpha=0.5, label='difference')
    axs[1,0].plot(np.abs(get_envelope_no_filtering(torch_fft.detach()).cpu().numpy()-\
                         get_envelope_no_filtering(pyroom_fft.detach()).cpu().numpy()), alpha=0.5, label='ENVELOPE difference')
    axs[1,0].legend()

    axs[0,1].set_title('FFT')
    axs[0,1].plot(np.abs(np.fft.rfft(torch_rir_np)), label='pytorch')
    axs[0,1].plot(np.abs(np.fft.rfft(pyroom_rir_np)), alpha=0.5, label='pyroom')
    axs[0,1].legend()
    
    axs[1,1].set_title('FFT Difference')
    axs[1,1].plot(np.fft.rfft(torch_rir_np) - np.fft.rfft(pyroom_rir_np))

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show: plt.show()

def fix_tensors(room_dimensions, mic_position, source_position):
    # Ensure room_dimensions is positive
    torch.clamp(room_dimensions, 1, float('inf'))

    # Clamp mic_position and source_position within the room dimensions
    torch.clamp(mic_position, torch.Tensor([0,0,0]), room_dimensions)
    torch.clamp(source_position, torch.Tensor([0,0,0]), room_dimensions)
    
    return room_dimensions, mic_position, source_position

def get_envelope_no_filtering(rir_fft):
    rir_fft=torch.fft.irfft(rir_fft)
    rir_fft=rir_fft/torch.max(torch.abs(rir_fft))
    rir_fft=torch.abs(rir_fft)
    
    return rir_fft

def get_filtered_envelopes(rir_fft, filters):
    filtered_rir_envelopes=[]
    for i in range(0,len(filters)):
        # filtered_rir_ffts.append(rir_fft*filters[i])
        filtered_rir=torch.fft.irfft(rir_fft*filters[i])
        filtered_rir_envelope=torch.abs(filtered_rir)
        normalized_filtered_rir_envelope=filtered_rir_envelope/torch.max(filtered_rir_envelope)
        filtered_rir_envelopes.append(normalized_filtered_rir_envelope)
    
    return filtered_rir_envelopes

def sum_and_normalize_envelopes(envelopes):
    sum_envelope=torch.zeros(envelopes[0].shape)
    for envelope in envelopes:
        sum_envelope=sum_envelope+envelope
    return sum_envelope/torch.max(sum_envelope)

# metrics

def RT60(rir2, sample_rate, rir2_sum=None, return_rir2_sum=False, origin=0, epsilon=0.001, plot=False):
    '''
    RT60 is backpropagatable if the origin of the decay curve is given!!
    We first draw the decay curve starting from the origin (the first echo/peak)
    We convert that curve to the log domain, draw lines between the origin (0,0) and all the points to get the proposed angles (betas)
    Get the average beta.
    Get the RT60.
    You can optionally return the rir2 sum.
    '''

    if rir2_sum == None:
        full_integral=torch.sum(rir2)
    else:
        full_integral=rir2_sum
    
    origin_int=int(origin.item()) # ORIGIN MUST BE BACKPROPAGATABLE GRR

    # calculate the delay curve starting from the end to avoid redundant sums
    decay_curve=[]
    decay_curve.append(rir2[-1].reshape(1))
    for t in range(1,len(rir2)-origin_int):
        decay_curve.append(rir2[-1-t].reshape(1) + decay_curve[-1])
    decay_curve.reverse()

    # Non arbitrary tail cut index : find when the curve has properly stopped decreasing
    tail_cut_index=0
    threshold=epsilon*decay_curve[0]

    while decay_curve[tail_cut_index] > threshold:
        if tail_cut_index == len(decay_curve)-1:
            print("RT60 : tail cut index is the end of the curve")
            break
        tail_cut_index+=1

    # Cat all the tensors together and apply the log (only before the tail)
    # decay_curve_tens=torch.Tensor(tail_cut_index)
    decay_curve_tens=torch.cat(decay_curve[:tail_cut_index]) #, out=decay_curve_tens)
    decay_curve_tens=decay_curve_tens/decay_curve_tens[0] # Normalize to 1.0
    log_decay = -torch.log(decay_curve_tens)
    
    # Get all the betas and average them
    times_samples=torch.arange(tail_cut_index, device=rir2.device)
    times_seconds = times_samples / sample_rate
    betas=log_decay[1:]/times_seconds[1:]
    regressed_beta=torch.mean(betas)
    
    if plot:
        print("regressed beta",regressed_beta)

        plt.figure()
        plt.title("decay curve")
        plt.plot(times_seconds.detach().numpy(),decay_curve_tens.detach().numpy())

        plt.figure()
        plt.title("rt60 beta regression on log decay_curve")
        plt.plot(times_seconds.detach().numpy(),log_decay.detach().numpy())
        plt.plot(times_seconds.detach().numpy(),times_seconds.detach().numpy()*regressed_beta.item(), label="regressed beta")
        plt.legend()
        plt.show()

    rt60=6*math.log(10)/regressed_beta

    if return_rir2_sum: return rt60, full_integral
    else: return rt60

def D(rir2, sample_rate, rir2_sum=None, return_rir2_sum=False, origin=0):
    '''
    backpropagatable
    since INT isn't backpropagatable, and floats can't be used as indexes... So let's use a trick.
    We multiply by a sigmoid which is transposed by the origin/fifty ms mark.
    '''
    fifty_ms=torch.Tensor([0.050*sample_rate]).to(rir2.device) + origin
    before_50ms, after_50ms = cut_before_and_after_index(rir2, index=fifty_ms, cut_severity=0.05)

    partial_integral=torch.sum(before_50ms) #torch.sum(rir2[:fifty_ms])
    if rir2_sum == None : 
        full_integral=partial_integral + torch.sum(after_50ms) #torch.sum(rir2[fifty_ms:])
    else:
        full_integral=rir2_sum
    
    if return_rir2_sum :
        return partial_integral/full_integral, rir2_sum
    else:
        return partial_integral/full_integral

def C80(rir2, sample_rate, return_rir2_sum=False, origin=0):
    '''
    backpropagatable
    since INT isn't backpropagatable, and floats can't be used as indexes... So let's use a trick.
    We multiply by a sigmoid which is transposed by the origin/fifty ms mark.
    '''
    eighty_ms=torch.Tensor([0.080*sample_rate]).to(rir2.device) + origin
    before_80ms, after_80ms = cut_before_and_after_index(rir2, index=eighty_ms, cut_severity=0.05)

    lower_integral=torch.sum(before_80ms)
    top_integral=torch.sum(after_80ms)
    
    if return_rir2_sum :
        return 10*torch.log10(top_integral/lower_integral), top_integral+lower_integral
    else:
        return 10*torch.log10(top_integral/lower_integral)

def center_time(rir2, sample_rate, rir2_sum=None, origin=0):
    '''backpropagatable'''
    if rir2_sum==None:
        lower_integral=torch.sum(rir2)
    else:
        lower_integral=rir2_sum
    top_integral=0

    softplus=torch.nn.Softplus()
    t=softplus(torch.arange(len(rir2), dtype=torch.float, device=rir2.device)-origin)#t is composed of zeroes up to origin, then starts incrementing. Using a translated softplus
    top_integral=torch.sum(torch.multiply(t,rir2))/sample_rate
    return top_integral/lower_integral

def ms_enveloppe_diff(rir1,rir2,filtering=False):
        '''
        backpropagatable
        Computes the difference directly,which the other metrics do not do.
        Manages if filters aren't initialized
        '''
        rir1_fft=torch.fft.rfft(rir1)
        rir2_fft=torch.fft.rfft(rir2)
        assert(rir1_fft.shape==rir2_fft.shape)
        if filtering:
            filters=[create_filter(rir1_fft, 1, bw) for bw in [100, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000]] # [50, 100, 200, 300, 400, 500, 1000, 2000, 4000]
            rir1_envelopes=get_filtered_envelopes(rir1_fft,filters)
            rir2_envelopes=get_filtered_envelopes(rir2_fft,filters)
            loss=0.0
            for i in range(0,len(filters)):
                loss=loss+torch.sum(torch.abs(rir1_envelopes[i]-rir2_envelopes[i]))
            return loss
        else:
            rir1_enveloppe=get_envelope_no_filtering(rir1_fft)
            rir2_enveloppe=get_envelope_no_filtering(rir2_fft)
            return torch.sum(torch.abs(rir1_enveloppe-rir2_enveloppe))
    
class RIRmetricsExperiments():
    '''
    
    '''
    def __init__(self, sample_rate=48000, lr=1e-3, lambda_param={'mrstft': 1, 'd': 1, 'c81': 1, 'center_time': 1, 'ms_env': 1},
                 target_room_dimensions=[6, 5, 4], target_mic_position=[3,2,1], target_source_position=[4,3,2],
                 device=None,
                 max_order=10):

        # Simulation parameters
        self.sample_rate = sample_rate
        self.sound_speed = 343.0
        self.max_order=max_order
        if device==None: self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device=torch.device(device)

        # Pyroom target
        self.set_target(target_room_dimensions=target_room_dimensions, target_mic_position=target_mic_position, target_source_position=target_source_position)

        # Inits for rirs and losses
        self.lambda_param = lambda_param
        self.which_losses=self.init_which_losses([]) # self.init_which_losses(['ms_env'])
        self.precalculated_metrics=None
        self.torch_rir=None
        self.pyroom_rir=None
        self.init_loss_logs()
        self.mrstft=None

    def set_target(self, target_room_dimensions=None, target_mic_position=None, target_source_position=None):
        '''Sets pyroom target. Used in initialization and can be useful if you want to change pyroom targets without reinitializing the class.
        Can be used with all None to reinitialize the target RIR if you changed the pyroom target parameters.'''
        if target_room_dimensions!=None: self.target_room_dimensions = target_room_dimensions 
        if target_mic_position!=None: self.target_mic_position = target_mic_position
        if target_source_position!=None: self.target_source_position = target_source_position
        self.pyroom_rir_og=get_pyroom_rir(self.target_room_dimensions, self.target_mic_position, self.target_source_position, self.sample_rate, self.max_order)
        peak_indexes, _ =find_peaks(self.pyroom_rir_og/np.max(self.pyroom_rir_og),height=0.3)
        self.pyroom_origin = peak_indexes[0]

    def init_loss_logs(self):
        '''Initializes the loss logs / Resets the loss logs after an experiment.'''
        self.loss_log = {}
        for loss in self.which_losses:
            self.loss_log[loss]=[]
        self.loss_log["total_loss"]=[]
        self.loss_dict={}

    def init_which_losses(self, which_losses=[]):
        '''
        Initializes which losses will be used.
        If none are specified, losses will be chosen based on which lambda parameters on non-zero.
        '''
        if which_losses==[]:
            for loss in self.lambda_param.keys():
                if self.lambda_param[loss] != 0:
                    which_losses.append(loss)
        return(which_losses)

    def overfit_training(self, num_iterations=10, lr=1e-6, plot=True, timing=False, plot_losses=True, print_stuff=True,
            initial_room_dimensions=[6, 5, 4], initial_mic_position=[3,2,1], initial_source_position=[4.2,3.8,2.5]):
        '''
        EXPERIMENT
        This method gradually iterates on self.log_room_dimensions, self.inv_sig_mic_position and self.inv_sig_source_position using gradient descent.
        It tries to minimize the metrics when compared with a specific target (the pyroom target).
        '''
        # Initial torch state
        self.log_room_dimensions = torch.tensor(np.log(np.array(initial_room_dimensions)), dtype=torch.float32, requires_grad=True, device=self.device)
        self.inv_sig_mic_position = torch.tensor(logit(np.array(initial_mic_position)/np.array(initial_room_dimensions)), dtype=torch.float32, requires_grad=True, device=self.device)
        self.inv_sig_source_position = torch.tensor(logit(np.array(initial_source_position)/np.array(initial_room_dimensions)), dtype=torch.float32, requires_grad=True, device=self.device)

        # inits for overfit training
        torch.autograd.set_detect_anomaly(False)
        optimizer = torch.optim.SGD([self.log_room_dimensions,self.inv_sig_mic_position, self.inv_sig_source_position], lr=lr, momentum=0.9)
        best_loss=float('inf')
        best_room_dimensions, best_mic_position, best_source_position = None, None, None
        best_iteration=0

        if print_stuff:
            self.print_iteration_info(text="Initial", print_loss=False, print_target=True)
            print("##################### TRAINING START #####################\n\n")

        if timing:
            time_dict={}
            time_log_dict=None
        for iteration in range(num_iterations):
            if timing: time_dict['initial_time']=time()
            
            #Un-log and un-logit the tensors
            room_dimensions, mic_position, source_position = unlog_unlogit(self.log_room_dimensions, self.inv_sig_mic_position, self.inv_sig_source_position)
            if timing: time_dict['unlog_time']=time()
            
            # Forward pass through backpropagatable pytorch shoebox RIR calculation
            self.torch_rir=get_pytorch_rir(room_dimensions,mic_position,source_position,self.sample_rate, max_order=self.max_order)
            if timing: time_dict['torch_rir_time']=time()
            
            # Make calculated torch rir comparable with the pyroom rir
            self.torch_rir, self.pyroom_rir = make_both_rir_comparable(self.torch_rir, self.pyroom_rir_og)
            if timing: time_dict['make_comparable_time']=time()
            
            # Get torch origin
            torch_distance = torch.linalg.norm(mic_position-source_position)
            torch_origin = 40 + (self.sample_rate*torch_distance/self.sound_speed)
            if timing: time_dict['origin_time']=time()
            
            # plot initial rir
            if plot and iteration == 0 : plot_rir(self.torch_rir,self.pyroom_rir,addendum="During Optimization, Iteration "+ str(iteration), show=False, torch_origin=torch_origin.detach(), pyroom_origin=self.pyroom_origin)
            if timing: time_dict['plot_time']=time()
            
            # possibly precalculate pyroom losses
            if self.precalculated_metrics == None: self.precalculate_pyroom_metrics(pyroom_origin=torch.Tensor([self.pyroom_origin]).to(self.device))
            if timing: time_dict['precalculate_metrics_time']=time()
            
            # Calculate losses
            self.update_losses(ms_env_filtering=True, torch_origin=torch_origin)
            if timing: time_dict['update_losses_time']=time()
            total_loss=0
            for loss in self.loss_dict.keys():
                total_loss=total_loss + self.lambda_param[loss]*self.loss_dict[loss]

            # Check if this is the best iteration and save it if yes
            if total_loss < best_loss:
                best_loss=total_loss
                best_room_dimensions, best_mic_position, best_source_position = room_dimensions, mic_position, source_position
                best_iteration=iteration

            # Log the loss values
            for loss in self.loss_dict.keys():
                self.loss_log[loss].append(self.loss_dict[loss].item()*self.lambda_param[loss])
            self.loss_log["total_loss"].append(total_loss.item())
            
            if print_stuff : self.print_iteration_info(iteration=iteration, total_loss=total_loss, text="Intermediate", print_loss=True)

            if timing: time_dict['log_losses_time']=time()

            # Backpropagation
            # total_loss=-total_loss
            total_loss.backward()
            if timing: time_dict['backward_time']=time()

            # self.print_gradients()
            # self.log_room_dimensions.grad=-self.log_room_dimensions.grad
            # self.inv_sig_mic_position.grad=-self.inv_sig_mic_position.grad
            # self.inv_sig_source_position.grad=-self.inv_sig_source_position.grad

            # Perform optimizer step
            optimizer.step() # Do gradient descent
            if timing: time_dict['opt_step_time']=time()

            # Clear gradients
            optimizer.zero_grad()

            if timing:
                time_keys=['initial_time', 'unlog_time', 'torch_rir_time',
                           'make_comparable_time', 'origin_time', 'plot_time',
                           'precalculate_metrics_time', 'update_losses_time', 'log_losses_time',
                           'backward_time', 'opt_step_time']
                if time_log_dict==None: time_log_dict={key:[] for key in time_keys}; time_log_dict['total_time']=[]
                for i in range(1,len(time_keys)):
                    # print(time_keys[i], time_dict[time_keys[i]]-time_dict[time_keys[i-1]])
                    time_log_dict[time_keys[i]].append(time_dict[time_keys[i]]-time_dict[time_keys[i-1]])
                time_log_dict['total_time'].append(time_dict['opt_step_time']-time_dict['initial_time'])

        if print_stuff : self.print_iteration_info(text="Final", print_loss=False)

        if plot:
            # Plot the final rir
            plot_rir(self.torch_rir, self.pyroom_rir, addendum="After Optimization", show=False)
            # Plot the best rir
            best_torch_rir=get_pytorch_rir(best_room_dimensions,best_mic_position,best_source_position,self.sample_rate)
            best_torch_rir, best_pyroom_rir = make_both_rir_comparable(best_torch_rir, self.pyroom_rir_og)
            plot_rir(best_torch_rir, best_pyroom_rir, addendum="Best RIR (iteration "+str(best_iteration)+")", show=False)
            if not plot_losses and not timing: plt.show()
      
        if timing:#plot logged times
            plt.figure()
            plt.title("Time log for device "+str(self.device))
            plt.xlabel('Iteration')
            plt.ylabel('Time')
            for key in time_log_dict.keys():
                plt.plot(time_log_dict[key], label=key)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout()
            if not plot_losses: plt.show()

        if plot_losses:
            plt.figure()
            plt.title('Loss over iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Loss value')
            for loss in self.loss_log.keys():
                plt.plot(self.loss_log[loss], label=loss)
            plt.legend()
            plt.show()
            plt.savefig('loss_plot2.png')

        torch.cuda.empty_cache()

    def grid_losses(self, grid_step=2, plot=True):
        '''
        EXPERIMENT
        this method is used to try and show how convex the different metrics are and wether they can be used to do regression or not.
        We fix both the mic and room dimensions to be identical to the target, and then try all different source positions in a grid-like fashion.
        We will then have a different loss value for every source position in space.
        We can then plot these values in a 2D space with colors for the values and the source position as the position with height being fixed.

        There is no need to do regression, so no need to use the optimizer, or self.log_rooom.. or self.in_sig_src...
        '''
        self.loss_log["source_position"]=[]

        room_dimensions=torch.tensor(self.target_room_dimensions, dtype=torch.float32, requires_grad=False, device=self.device)
        mic_position=torch.tensor(self.target_mic_position, dtype=torch.float32, requires_grad=False, device=self.device)

        print("room dimensions ", self.target_room_dimensions)
        print("mic position ", self.target_mic_position)
        print("source grid step ", grid_step)

        print("\n##################### GRID LOSSES START #####################\n\n")

        def get_grid_of_source_positions(room_dimensions, source_height, step):
            ''''''
            grid_of_source_positions=[]
            for x in np.arange(0,room_dimensions[0],step):
                for y in np.arange(0,room_dimensions[1],step):
                    grid_of_source_positions.append([x,y,source_height])
            return grid_of_source_positions

        grid_of_source_positions=get_grid_of_source_positions(self.target_room_dimensions, self.target_source_position[2],grid_step)

        # calculate grid of losses
        for source_position_as_list in grid_of_source_positions:
            source_position=torch.tensor(source_position_as_list, dtype=torch.float32, requires_grad=False, device=self.device)

            self.torch_rir=get_pytorch_rir(room_dimensions,mic_position,source_position,self.sample_rate, max_order=self.max_order)

            self.torch_rir, self.pyroom_rir = make_both_rir_comparable(self.torch_rir, self.pyroom_rir_og)
            
            torch_distance = torch.linalg.norm(mic_position-source_position)
            torch_origin = 40 + (self.sample_rate*torch_distance/self.sound_speed)
            
            if self.precalculated_metrics == None: self.precalculate_pyroom_metrics(pyroom_origin=torch.Tensor([self.pyroom_origin]).to(self.device))
            
            self.update_losses(ms_env_filtering=True, torch_origin=torch_origin)
            total_loss=0
            for loss in self.loss_dict.keys():
                total_loss=total_loss + self.lambda_param[loss]*self.loss_dict[loss]

            for loss in self.loss_dict.keys():
                self.loss_log[loss].append(self.loss_dict[loss].item()*self.lambda_param[loss])
            self.loss_log["total_loss"].append(total_loss.item())
            self.loss_log["source_position"].append(source_position_as_list)
            
            #print losses and current source position
            print("Source position : ", source_position_as_list)
            print("Losses :", end=" ")
            for loss in self.loss_dict.keys():
                print(loss, fl_p(self.loss_dict[loss].item()*self.lambda_param[loss]), end=" , ")
            print("total_loss", fl_p(total_loss.item()))

        # plot grid of losses
        if plot:
            losses_to_plot=list(self.loss_dict.keys()) #+ ["total_loss"]
            number_of_plots=len(losses_to_plot)
            number_of_plots_y=math.ceil(math.sqrt(number_of_plots))
            number_of_plots_x=math.ceil(number_of_plots/number_of_plots_y)
            fig_grid, axs_grid = plt.subplots(number_of_plots_x,number_of_plots_y, figsize=(17, 10))
            # fig_grid.suptitle("Grid losses")
            i=0
            for loss in losses_to_plot:
                y=i%number_of_plots_y
                x=i//number_of_plots_y
                axs_grid[x,y].set_title(loss +' loss based on source position')
                axs_grid[x,y].set_xlabel('Source position x')
                axs_grid[x,y].set_ylabel('Source position y')
                axs_grid[x,y].grid(True, ls=':', alpha=0.5)
                axs_grid[x,y].add_patch(Rectangle((0,0),self.target_room_dimensions[0],self.target_room_dimensions[1],
                        edgecolor='black',
                        facecolor='none',
                        lw=3))
                # print(self.target_source_position[0], self.target_source_position[1])
                axs_grid[x,y].scatter([self.target_source_position[0]], [self.target_source_position[1]], c="red", marker='x', label="Target source")
                axs_grid[x,y].scatter([self.target_mic_position[0]], [self.target_mic_position[1]], c="blue", marker='D', label="Microphone")
                # print("self.loss_log[source_position]", [thing[0] for thing in self.loss_log["corresponding source_position"]])
                # print("self.loss_log[corresponding source_position]", self.loss_log["corresponding source_position"][:][0])
                scatter=axs_grid[x,y].scatter([thing[0] for thing in self.loss_log["corresponding source_position"]],
                            [thing[1] for thing in self.loss_log["corresponding source_position"]],
                            c=self.loss_log[loss],
                            alpha=0.5)
                axs_grid[x,y].legend()
                # axs_grid[x,y].colorbar()
                plt.colorbar(scatter,ax=axs_grid[x,y])
                i+=1
            plt.tight_layout()
            plt.show()

        torch.cuda.empty_cache()

    def rooms_losses(self, grid_step=3, min_room_dimensions=None, max_room_dimensions=None, plot=True, print_stuff=True, logger=None):
        '''
        EXPERIMENT
        this method is used to try and show how convex the different metrics are and wether they can be used to do regression or not.
        mic/source positions = target mic/source positions.
        room height = Target room height.
        If you modify the targets, please run self.set_target() before running this method.
        Else, you might be comparing different types of rooms to different types of rooms (you might want this behaviour, but it's not the default)

        We try many different room_dimensions in a grid-like fashion and compute their loss against the target room.
        We will then have a different loss value for every room size.
        We then plot these values in a 2D space with colored points for the values of the losses giving us sort of a heat map.

        There is no regression, we're just testing the convexity of the losses.
        '''
        # Inits
        self.loss_log["room_dimension"]=[]
        if logger==None:
            logger=LKLogger(filename='room_task_log.csv', columns_for_a_new_log_file= [ "max_order",
                                                                                        "room_dimension_x","room_dimension_y","room_dimension_z",
                                                                                        "target_room_dimensions_x","target_room_dimensions_y","target_room_dimensions_z",
                                                                                        "source_position_x","source_position_y","source_position_z",
                                                                                        "mic_position_x","mic_position_y","mic_position_z",
                                                                                        ] + self.which_losses)

        # Manage default max and min room dimensions
        max_room_dimensions=max_room_dimensions if max_room_dimensions!=None else [i *1.8 for i in self.target_room_dimensions]
        # print(self.target_mic_position[0],self.target_source_position[0])
        min_room_dimensions=min_room_dimensions if min_room_dimensions!=None else [max(self.target_mic_position[i],self.target_source_position[i]) + 0.2 for i in range(3)]
        
        #init mic and src positions
        mic_position=torch.tensor(self.target_mic_position, dtype=torch.float32, requires_grad=False, device=self.device)
        source_position=torch.tensor(self.target_source_position, dtype=torch.float32, requires_grad=False, device=self.device)

        # Print relevant info
        if print_stuff:
            print("target room dimensions ", self.target_room_dimensions)
            print("mic position ", self.target_mic_position)
            print("source_position ", self.target_source_position)
            print("room grid step ", grid_step)
            print('min_room_dimensions', min_room_dimensions)
            print('max_room_dimensions', max_room_dimensions)

            print("\n##################### ROOM LOSSES START #####################\n\n")

        # Get grid of room dimensions
        def get_grid_of_room_dimensions(room_h, min_size, max_size, step):
            '''
            Get many different room_dimensions in a grid-like fashion. the room height is fixed
            '''
            grid_of_room_dimensions=[]
            for x in np.arange(min_size[0],max_size[0],step):
                for y in np.arange(min_size[1],max_size[1],step):
                    grid_of_room_dimensions.append([x,y,room_h])
            return grid_of_room_dimensions
        grid_of_room_dimensions=get_grid_of_room_dimensions(self.target_room_dimensions[2], min_room_dimensions,max_room_dimensions,grid_step)

        # calculate grid of losses
        for room_dimension_as_list in grid_of_room_dimensions:
            task_parameters={"max_order":self.max_order,
                "room_dimension_x": room_dimension_as_list[0],"room_dimension_y": room_dimension_as_list[1],"room_dimension_z": room_dimension_as_list[2],
                "target_room_dimensions_x": self.target_room_dimensions[0],"target_room_dimensions_y": self.target_room_dimensions[1], "target_room_dimensions_z": self.target_room_dimensions[2],
                "source_position_x": self.target_source_position[0],"source_position_y": self.target_source_position[1], "source_position_z": self.target_source_position[2],
                "mic_position_x": self.target_mic_position[0],"mic_position_y": self.target_mic_position[1], "mic_position_z": self.target_mic_position[2]
                }
            if not logger.check_if_line_in_log(task_parameters):
                room_dimensions=torch.tensor(room_dimension_as_list, dtype=torch.float32, requires_grad=False, device=self.device)
                self.torch_rir=get_pytorch_rir(room_dimensions,mic_position,source_position,self.sample_rate, max_order=self.max_order)
                self.torch_rir, self.pyroom_rir = make_both_rir_comparable(self.torch_rir, self.pyroom_rir_og)
                torch_distance = torch.linalg.norm(mic_position-source_position)
                torch_origin = 40 + (self.sample_rate*torch_distance/self.sound_speed)
                if self.precalculated_metrics == None: self.precalculate_pyroom_metrics(pyroom_origin=torch.Tensor([self.pyroom_origin]).to(self.device))

                self.update_losses(ms_env_filtering=True, torch_origin=torch_origin)
                total_loss=0
                for loss in self.loss_dict.keys():
                    total_loss=total_loss + self.lambda_param[loss]*self.loss_dict[loss]

                for loss in self.loss_dict.keys():
                    self.loss_log[loss].append(self.loss_dict[loss].item()*self.lambda_param[loss])
                self.loss_log["total_loss"].append(total_loss.item())
                self.loss_log["room_dimension"].append(room_dimension_as_list)
                
                #print losses and current Room dimension
                if print_stuff:
                    print("Room dimension : ", room_dimension_as_list)
                    print("Losses :", end=" ")
                    for loss in self.loss_dict.keys():
                        print(loss, fl_p(self.loss_dict[loss].item()*self.lambda_param[loss]), end=" , ")
                    print("total_loss", fl_p(total_loss.item()))

                # log task parameters
                logger.add_line_to_log(task_parameters)
            else:
                print("Already logged this task:", task_parameters, "skipping")
        
        # plot grid of losses
        if plot:
            losses_to_plot=list(self.loss_dict.keys()) #+ ["total_loss"]
            # print("losses_to_plot", losses_to_plot)
            number_of_plots_x=2
            number_of_plots_y=3
            fig_rooms, axs_rooms = plt.subplots(number_of_plots_x,number_of_plots_y, figsize=(17, 10))
            # fig_rooms.suptitle("Grid losses")
            i=0
            for loss in losses_to_plot:
                x=i//(number_of_plots_y)
                y=i%(number_of_plots_y)
                
                axs_rooms[x,y].set_title(loss +' loss based on room size')
                axs_rooms[x,y].set_xlabel('x (m)')
                axs_rooms[x,y].set_ylabel('y (m)')
                axs_rooms[x,y].grid(True, ls=':', alpha=0.5)

                # plot rectangles
                X=grid_of_room_dimensions
                Y=self.loss_log[loss]
                sorted_grid_of_room_dimensions=[x for _, x in sorted(zip(Y, X), key=lambda pair: pair[0])]
                # sorted_loss=sorted(self.loss_log[loss])
                # cmap=mpl.cm.get_cmap('YlGnBu')
                # cmap=mpl.cm.get_cmap('viridis')
                # max_loss=max(self.loss_log[loss])
                # min_loss=min(self.loss_log[loss])
                # for iterator in range(0,len(grid_of_room_dimensions)): # plot rectangles
                #     value=(sorted_loss[iterator]-min_loss)/(max_loss-min_loss)
                #     color=cmap(value)
                #     axs_rooms[x,y].add_patch(Rectangle((0,0),sorted_grid_of_room_dimensions[iterator][0],sorted_grid_of_room_dimensions[iterator][1],
                #         edgecolor=color,
                #         facecolor='none',
                #         lw=5,
                #         alpha=value))
                scatter=axs_rooms[x,y].scatter([thing[0] for thing in self.loss_log["room_dimension"]],
                            [thing[1] for thing in self.loss_log["room_dimension"]],
                            c=self.loss_log[loss],
                            alpha=1)
                # lowest loss rectangle
                axs_rooms[x,y].add_patch(Rectangle((0,0),sorted_grid_of_room_dimensions[0][0],sorted_grid_of_room_dimensions[0][1],
                        edgecolor='blue',
                        facecolor='none',
                        lw=5,
                        alpha=0.5,
                        label="Lowest loss room"))
                # target room rectangle
                axs_rooms[x,y].add_patch(Rectangle((0,0),self.target_room_dimensions[0],self.target_room_dimensions[1],
                        edgecolor='red',
                        facecolor='none',
                        lw=5,
                        alpha=0.5,
                        label="Target room"))
                plt.colorbar(scatter,ax=axs_rooms[x,y])
                # norm = mpl.colors.Normalize(vmin=min_loss, vmax=max_loss)
                # cb = mpl.colorbar.ColorbarBase(ax=axs_rooms[x,y], cmap=cmap,
                #                                 norm=norm)
                axs_rooms[x,y].scatter([self.target_source_position[0]], [self.target_source_position[1]], c="red", marker='x', label="Source")
                axs_rooms[x,y].scatter([self.target_mic_position[0]], [self.target_mic_position[1]], c="red", marker='D', label="Microphone")
                # scatter=axs_rooms[x,y].scatter([thing[0] for thing in self.loss_log["source_position"]],
                #             [thing[1] for thing in self.loss_log["source_position"]],
                #             c=self.loss_log[loss],
                #             alpha=0.5)
                axs_rooms[x,y].legend()
                i+=1
            plt.tight_layout()
            plt.show()
        
        torch.cuda.empty_cache()

    def precalculate_pyroom_metrics(self, pyroom_origin=0):
        '''
        Calculates d, c80 and center_time metrics.
        Stores them in self.precalculated_metrics
        '''
        # precalculate the rir^2
        pyroom_rir2=torch.pow(self.pyroom_rir,2)
        # extract the rir2 sum
        c80, pyroom_rir2_sum = C80(pyroom_rir2, self.sample_rate, return_rir2_sum=True, origin=pyroom_origin)
        rt60 = RT60(pyroom_rir2, self.sample_rate, rir2_sum=pyroom_rir2_sum, origin=pyroom_origin)
        d = D(pyroom_rir2, self.sample_rate, rir2_sum=pyroom_rir2_sum, origin=pyroom_origin)
        c_t= center_time(pyroom_rir2, self.sample_rate, rir2_sum=pyroom_rir2_sum, origin=pyroom_origin)

        self.precalculated_metrics={"rt60": rt60,"d": d, "c80": c80, "center_time": c_t}

    def update_losses(self, ms_env_filtering=False, torch_origin=0):
        '''
        Updates self.loss_dict
        '''
        # manage if metrics weren't precalculated
        if self.precalculated_metrics==None:
            self.precalculated_metrics = self.precalculate_metrics()

        # precalculate torch_rir2 if needed.
        if 'rt60' in self.which_losses or 'd' in self.which_losses or 'c80' in self.which_losses or 'center_time' in self.which_losses:
            torch_rir2=torch.pow(self.torch_rir,2) # precalculate rir^2, don't need to store it.

        # prepare precalculate sum(rir^2)
        torch_rir2_sum=None 

        # Update mrstft loss
        if 'mrstft' in self.which_losses:
            if self.mrstft==None:
                self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[1024, 2048, 8192],
                    hop_sizes=[256, 512, 2048],
                    win_lengths=[1024, 2048, 8192],
                    scale="mel",
                    n_bins=128,
                    sample_rate=self.sample_rate,
                    perceptual_weighting=True,
                    device=self.device,
                )
            mrstft_loss = self.mrstft(self.torch_rir[None, None, :], self.pyroom_rir[None, None, :])
            self.loss_dict['mrstft']=mrstft_loss

        # Update c80 loss
        if 'c80' in self.which_losses:
            c80, torch_rir2_sum = C80(torch_rir2, self.sample_rate, return_rir2_sum=True, origin=torch_origin)# Calculate torch C80, use rir2_sum if possible, get rir2_sum if possible
            c80_loss = torch.abs(c80 - self.precalculated_metrics['c80'] ) # Difference
            self.loss_dict['c80']=c80_loss # Store

        # Update rt60 loss
        if 'rt60' in self.which_losses:
            rt60, torch_rir2_sum = RT60(torch_rir2, self.sample_rate, rir2_sum=torch_rir2_sum, return_rir2_sum=True, origin=torch_origin) # Calculate torch RT60, use rir2_sum if possible, get rir2_sum if possible
            rt60_loss = torch.abs(rt60 - self.precalculated_metrics['rt60'] ) # Difference
            self.loss_dict['rt60']=rt60_loss # Store

        # Update d loss
        if 'd' in self.which_losses:
            d, torch_rir2_sum = D(torch_rir2, self.sample_rate, rir2_sum=torch_rir2_sum, return_rir2_sum=True, origin=torch_origin) # Calculate torch D, use rir2_sum if possible, get rir2_sum if possible
            d_loss = torch.abs(d - self.precalculated_metrics['d'])  # Difference
            self.loss_dict['d']=d_loss # Store

        # Update center_time loss
        if 'center_time' in self.which_losses:
            c_t = center_time(torch_rir2, self.sample_rate, rir2_sum=torch_rir2_sum, origin=torch_origin) # Calculate torch c_t, use rir2_sum if possible
            center_time_loss=torch.abs(c_t - self.precalculated_metrics['center_time'])  # Difference
            self.loss_dict['center_time']=center_time_loss # Store
        
        # Update ms_env loss
        if 'ms_env' in self.which_losses:
            ms_env_loss=ms_enveloppe_diff(self.torch_rir, self.pyroom_rir, filtering= ms_env_filtering)
            self.loss_dict['ms_env']=ms_env_loss

    def print_iteration_info(self, iteration=None, total_loss=torch.Tensor([0]), text="Intermediate", print_loss=True, print_target=False):
        if print_loss:
            if iteration!=None: print("Iteration:", iteration, "Loss:", (fl_p(total_loss.item())))
            #print each individual loss form the loss dictionary
            for loss in self.loss_dict.keys():
                print(loss, fl_p(self.loss_dict[loss].item()*self.lambda_param[loss]), end=" ")
        
        room_dimensions, mic_position, source_position = unlog_unlogit(self.log_room_dimensions, self.inv_sig_mic_position, self.inv_sig_source_position)

        print("\n"+ text + " room_dimensions", [fl_p(v) for v in room_dimensions.tolist()], end="")
        if print_target: print(" target_room_dimensions", self.target_room_dimensions)
        else: print("")
        print(text + " mic_position", [fl_p(v) for v in mic_position.tolist()], end="")
        if print_target: print(" target_mic_position",self.target_mic_position)
        else: print("")
        print(text + " source_position", [fl_p(v) for v in  source_position.tolist()], end="")
        if print_target: print(" target_source_position",self.target_source_position)
        else: print("")
        print("")

    def print_gradients(self):
        print("Gradients :")
        print("log room_dimensions",self.log_room_dimensions.grad)
        print("inv sig mic_position",self.inv_sig_mic_position.grad)
        print("inv sig source_position",self.inv_sig_source_position.grad)
        print("")

def measure_time(FWRM, task='rooms', devices=['cpu', 'cuda'], max_orders=[5, 7, 10, 12, 14],
                 lr=1e-6, num_iterations=10,
                 grid_step=3,
                 min_room_dimensions=4,max_room_dimensions=11,
                 logger=None):

    if logger==None:
        logger=LKLogger(filename='measure_time_log.csv', columns_for_a_new_log_file=['task', 'device', 'max_order', 'time',
                                                                                     'grid_step', 'min_room_dimensions', 'max_room_dimensions',
                                                                                     'lr', 'num_iterations'])

    for device in devices:
        for max_order in max_orders:
            FWRM.device=device
            FWRM.max_order=max_order

            if not logger.check_if_line_in_log({'task':task,'device':device,'max_order':max_order}):
                print("\n calculating time for task ", task, " on device ", device, " and max_order ", max_order)

                # if not, calculate time
                time1=time()
                if task=='overfit': FWRM.overfit_training(lr=lr, num_iterations=num_iterations, timing=False, plot=False, plot_losses=False, print_stuff=True)
                if task=='sources': FWRM.grid_losses(grid_step=grid_step, plot=False)
                if task=='rooms': FWRM.rooms_losses(grid_step=grid_step, min_room_dimensions=min_room_dimensions,max_room_dimensions=max_room_dimensions, plot=False, print_stuff=False)
                calc_time=time()-time1
                print("time : ", calc_time, "\n")

                logger.add_line_to_log({'task':task, 'device':device, 'max_order':max_order, 'time':calc_time,
                                        'grid_step':grid_step, 'min_room_dimensions':min_room_dimensions, 'max_room_dimensions':max_room_dimensions,
                                        'lr':lr, 'num_iterations':num_iterations})
            else:
                print("\n task ", task, " on device ", device, " and max_order ", max_order, " already calculated, skipping \n")

def plot_measure_time_log():
    df=pd.read_csv('./measure_time_log.csv')
    for task in df['task'].unique():
        df_task=df[df['task']==task]
        plt.figure()
        for device in df_task['device'].unique():
            df_device=df_task[df_task['device']==device]
            #sort df_device by max_order
            df_device=df_device.sort_values(by=['max_order'])
            # print(df_device['max_order'])
            # print(df_device['time'])
            plt.plot(df_device['max_order'].to_numpy(), df_device['time'].to_numpy(), label=device)
        if task=='overfit':
            plt.title("'overfit' task : calculate 10 iterations (forward + backward pass)")
        if task=='rooms':
            plt.title("'rooms' task : calculate 9 iterations (only a forward pass is done)")
        plt.ylim(0, df_task['time'].max()+10)
        plt.xlabel('max_order')
        plt.ylabel('time (s)')
        plt.grid(True,ls=':', alpha=0.5)
        plt.legend()
    plt.show()

def main():
    RIRME = RIRmetricsExperiments(target_room_dimensions=[6, 5, 4], target_mic_position=[1,0.5,2], target_source_position=[2,3,2],
                                    lambda_param={'mrstft': 1.5, 'rt60':7,'d': 3.5, 'c80': 0.25, 'center_time': 100, 'ms_env': 2e-4}, #lambda_param={'mrstft': 1.5, 'rt60':7,'d': 3.5, 'c80': 0.25, 'center_time': 100, 'ms_env': 2e-4}
                                    )
    RIRME.device='cuda'
    RIRME.max_order=12
    # RIRME.device='cpu'
    # RIRME.overfit_training(timing=True, plot=False,plot_losses=False)
    
    # Running experiments
    for room_dimensions in [[3.5,2.5], [5,4],[10,7]]:
        for room_height in [1, 2, 3 , 5 , 100]:
            
            RIRME.target_room_dimensions=[room_dimensions[0], room_dimensions[1], room_height]
            RIRME.target_mic_position=[1,1.5,RIRME.target_room_dimensions[2]/2]  # RIRME.target_mic_position=[0.5,1,RIRME.target_room_dimensions[2]/2]
            RIRME.target_source_position=[2,1,RIRME.target_room_dimensions[2]/2] # RIRME.target_source_position=[1,0.25,RIRME.target_room_dimensions[2]/2]
            RIRME.rooms_losses(grid_step=0.3, plot=True, print_stuff=True)
            RIRME.init_loss_logs() # reset loss logs

    # measure_time(RIRME=RIRME, task='rooms', devices=['cpu','cuda'], max_orders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    # measure_time(RIRME=RIRME, task='overfit', devices=['cpu','cuda'], max_orders=[1,2,3,4, 5, 6, 7, 8, 9, 10, 11,12])
    # plot_measure_time_log()
    
if __name__ == '__main__':
    main()