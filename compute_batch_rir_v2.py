'''
This is a reimplementation of torchaudio.prototype.functional.simulate_rir_ism
It is modified to work with pytorch tensors with autograd.
The original function is not differentiable because of C code, and therefore cannot be used in a neural network.
'''

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import pad

# Untouched function from torch implementation
def _hann(x: torch.Tensor, T: int):
    """Compute the Hann window where the values are truncated based on window length.
    torch.hann_window can only sample window function at integer points, the method is to sample
    continuous window function at non-integer points.

    Args:
        x (torch.Tensor): The fractional component of time delay Tensor.
        T (torch.Tensor): The window length of sinc function.

    Returns:
        (torch.Tensor): The hann window Tensor where values outside
            the sinc window (`T`) is set to zero.
    """
    y = torch.where(
        torch.abs(x) <= T / 2,
        0.5 * (1 + torch.cos(2 * math.pi * x / T)),
        x.new_zeros(1),
    )
    return y

# Untouched function from torch implementation
def _frac_delay(batch_delay: torch.Tensor, batch_delay_i: torch.Tensor, delay_filter_length: int):
    """Compute fractional delay of impulse response signal.

    Args:
        batch_delay (torch.Tensor): The time delay Tensor in samples. (batch_size, n_image_source, n_mics=1)
        batch_delay_i (torch.Tensor): The integer part of delay.  (batch_size, n_image_source, n_mics=1)
        delay_filter_length (int): The window length for sinc function.

    Returns:
        (torch.Tensor): The impulse response Tensor for all image sources. (n_image_sources, delay_filter_length)
    """
    assert  delay_filter_length % 2 == 1, "The filter length must be odd"

    pad = delay_filter_length // 2
    n = torch.arange(-pad, pad + 1, device=batch_delay.device) + batch_delay_i[..., None] # (batch_size, n_image_source, n_mics=1, delay_filter_length)
    batch_delay = batch_delay[..., None]

    return torch.special.sinc(n - batch_delay) * _hann(n - batch_delay, 2 * pad)

def _batch_validate_inputs(room: torch.Tensor, mic_array: torch.Tensor, source: torch.Tensor, absorption: torch.Tensor):
    '''Only supports mono band absorption, 3D dimensions, and 1 mic for now.'''
    
    assert(room.device==source.device==mic_array.device==absorption.device) , "All inputs room, source, mic array and absorption must be on the same device."

    batch_size = room.shape[0]
    assert room.shape == (batch_size, 3), f"room batch must be a 2D Tensor (batch_size, 3). Found {room.shape}."
    assert source.shape == (batch_size, 3), f"source batch must be a 2D Tensor (batch_size, 3). Found {source.shape}."
    assert mic_array.shape == (batch_size, 1, 3), f"mic_array batch must be a 3D Tensor (batch_size, n_mics=1, 3). Found {mic_array.shape}."
    
    NUM_WALL = 6 # Shoebox room
    assert absorption.shape == (batch_size, 1, NUM_WALL), f"Absorption must be a 3D Tensor of shape (batch_size, 1, n_walls=6). Found {absorption.shape}."

def _batch_compute_image_sources(
    room: torch.Tensor,
    source: torch.Tensor,
    max_order: int,
    absorption,
    scatter: Optional[torch.Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Compute image sources in a shoebox-like room. Only one frequency band for attentuation is supported for now.

    Args:
        room (torch.Tensor): The batch of 1D Tensors to determine the room size. (batch_size, 3)
        source (torch.Tensor): The batch of coordinates of the sound source. (batch_size, 3)
        max_order (int): The maximum number of reflections of the source.
        absorption (torch.Tensor): The absorption coefficients of wall materials. (batch_size, 1, 6)
        scatter (torch.Tensor): The scattering coefficients of wall materials. (batch_size, 1, 6). If ``None``, it is not
            used in image source computation. (Default: ``None``)

    Returns:
        (torch.Tensor): The coordinates of all image sources within ``max_order`` number of reflections.
            Tensor with dimensions `(num_image_source, D)`.
        (torch.Tensor): The attenuation of corresponding image sources. Tensor with dimensions
            `(num_band, num_image_source)`.
    """
    batch_size = room.shape[0]
    
    if scatter is None: tr = torch.sqrt(1 - absorption)
    else: tr = torch.sqrt(1 - absorption) * torch.sqrt(1 - scatter)

    # Constructing the grid
    ind = torch.arange(-max_order, max_order + 1, device=source.device)
    xyz = torch.meshgrid(ind, ind, ind, indexing="ij") # Make a grid as big as the max_order
    xyz = torch.stack([c.reshape((-1,)) for c in xyz], dim=-1)
    xyz = xyz[xyz.abs().sum(dim=-1) <= max_order] # Remove the points that are too far away
    
    n_image_sources = xyz.shape[0]
    xyz=xyz.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, n_image_source (4991 for max order 15), 3) as a view!

    # compute locations of image sources
    d = room.view(batch_size,1,3)
    s = source.view(batch_size,1,3)
    batch_img_loc = torch.where(xyz % 2 == 1, d * (xyz + 1) - s, d * xyz + s) # (batch_size, n_image_source, 3)

    # attenuation
    exp_lo = abs(torch.floor((xyz / 2))).unsqueeze(1) # (batch_size, num_band=1, n_image_source, 3)
    exp_hi = abs(torch.floor((xyz + 1) / 2)).unsqueeze(1) # (batch_size, num_band=1, n_image_source, 3)
    t_lo = tr[:,:, ::2].unsqueeze(2).expand(-1,-1, n_image_sources, -1)  # (batch_size, num_band=1, n_image_source, left walls = 6/2)
    t_hi = tr[:,:, 1::2].unsqueeze(2).expand(-1,-1, n_image_sources, -1)  # (batch_size, num_band=1, n_image_source, right walls = 6/2)

    batch_att = torch.prod((t_lo**exp_lo) * (t_hi**exp_hi), dim=-1)  # (batch_size, num_band, num_image_source)

    assert batch_img_loc.shape == (batch_size, n_image_sources, 3), f"Expecting ({batch_size}, {n_image_sources}, 3). Found {batch_img_loc.shape}."
    assert batch_att.shape == (batch_size, 1, n_image_sources), f"Expecting ({batch_size}, n_mics=1, {n_image_sources}). Found {batch_att.shape}."

    return batch_img_loc, batch_att

# New function
def _batch_add_all_mini_irs_together(batch_mini_irs, batch_delay_i, rir_length, delay_filter_length):
        '''
        does what torch.ops.torchaudio._simulate_rir does, but is a bit backpropagatable.
        Should definetely not be implemented wit for loops.
        I am just assuming n_mics=1 for now because thats what I need
        
        Args:
            batch_mini_irs (torch.Tensor): (batch_size, n_band=1, n_image_source, n_mics=1, delay_filter_length)
            batch_delay_i (torch.Tensor): (batch_size, n_image_source, n_mics=1)
            rir_length (int): full length of the rir signal
            delay_filter_length (int): The length of the delay filter.

        returns an rir of shape (rir_length)
        '''
        batch_size, _, n_image_source, _, _ = batch_mini_irs.shape
        
        og_device=batch_mini_irs.device

        batch_mini_irs=batch_mini_irs.to("cpu")
        rir = torch.zeros(batch_size, rir_length, device="cpu")

        pad_left = batch_delay_i
        pad_right = rir_length - delay_filter_length - batch_delay_i
        pad_left=pad_left.to('cpu')
        pad_right=pad_right.to('cpu')

        for b in range(batch_size):
            for i in range(n_image_source): # for each image source
                rir[b,:] = rir[b,:] + pad(batch_mini_irs[b,0,i,0,:], (pad_left[b,i],pad_right[b,i]))

        rir=rir.to(og_device)

        return rir

def batch_simulate_rir_ism(batch_room_dimensions: torch.Tensor,
                           batch_mic_position : torch.Tensor,
                           batch_source_position : torch.Tensor,
                           batch_absorption : torch.Tensor,
                           max_order : int, sample_rate : float = 16000.0, sound_speed: float = 343.0,
                           output_length: Optional[int] = None, delay_filter_length: int = 81,
                           #center_frequency: Optional[torch.Tensor] = None, # Not managing multi-band processing for now
) -> Tensor:
    """
    Simulate room impulse responses (RIRs) using image source method (ISM).
    I'm just not doing multi-band processing for now
    look up torchaudio.prototype.functional.simulate_rir_ism if you're interested
    
    Args:
        batch_room_dimensions must be a 2D Tensor (batch_size, 3).
        batch_mic_position must be a 3D Tensor (batch_size, n_mics=1, 3).
        batch_source_position must be a 2D Tensor (batch_size, 3).
        batch_absorption must be a 3D Tensor of shape (batch_size, 1, n_walls=6).
        max_order (int): The maximum number of reflections of the source.
        sample_rate (float): The sample rate of the RIRs. (Default: ``16000.0``)
        sound_speed (float): The speed of sound. (Default: ``343.0``)
        output_length (int, optional): The length of the output RIRs. (Default: ``None``)
        delay_filter_length (int): The length of the delay filter. (Default: ``81``)

    Returns:
        (torch.Tensor): batch of rir (batch_size, max_rir_length)
    """
    _batch_validate_inputs(batch_room_dimensions, batch_mic_position, batch_source_position, batch_absorption)
    
    batch_img_location, batch_att = _batch_compute_image_sources(batch_room_dimensions, batch_source_position, max_order, batch_absorption) # (n_image_source, 3[x,y,z]) , (n_band=1, n_image_source)

    # compute distances between image sources and microphones
    vec = batch_img_location[:,:, None, :] - batch_mic_position[:, None, :, :]
    batch_dist = torch.linalg.norm(vec, dim=-1)  # (batch_size, n_image_source, n_mics=1)

    #attenuate image sources
    batch_img_src_att = batch_att[..., None] / batch_dist[None, ...]  # (batch_size, n_band=1, n_image_source, n_mics=1)

    # separate delays in integer / frac part 
    batch_delay = batch_dist * sample_rate / sound_speed  # time to batch_delay in samples (fractionnal delay) (batch_size, n_image_source, n_mics=1)
    batch_delay_i = torch.ceil(batch_delay.detach()).int()  # integer part (batch_size, n_image_source, n_mics=1)

    # compute the shorts IRs corresponding to each image source
    batch_mini_irs = batch_img_src_att[..., None] * _frac_delay(batch_delay, batch_delay_i, delay_filter_length)[:,None, ...] # (batch_size,n_band=1, n_image_source, n_mics=1, delay_filter_length)
    rir_length = int(batch_delay_i.max() + batch_mini_irs.shape[-1]) # full length of the rir signal

    # sum up rir signals of all image sources into one waveform.
    rir = _batch_add_all_mini_irs_together(batch_mini_irs, batch_delay_i, rir_length,delay_filter_length) # (rir_length) # I am just assuming n_mics=1 for now

    if output_length is not None:
        if output_length > rir.shape[-1]:
            rir = torch.nn.functional.pad(rir, (0, output_length - rir.shape[-1]), "constant", 0.0)
        else:
            rir = rir[..., :output_length]

    return rir

