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
from torch.utils.checkpoint import checkpoint

# from tools.pyLiam.LKTimer import LKTimer
# timer=LKTimer()
import matplotlib.pyplot as plt
from backpropagatable_ISM.filters import LP_filter, BP_filter

def _batch_validate_inputs(room: torch.Tensor, mic_array: torch.Tensor, source: torch.Tensor, absorption: torch.Tensor):
    '''Only supports mono band absorption, 3D dimensions, and 1 mic for now.'''
    
    assert(room.device==source.device==mic_array.device==absorption.device) , "All inputs room, source, mic array and absorption must be on the same device."

    batch_size = room.shape[0]
    assert room.shape == (batch_size, 3), f"room batch must be a 2D Tensor (batch_size, 3). Found {room.shape}."
    assert source.shape == (batch_size, 3), f"source batch must be a 2D Tensor (batch_size, 3). Found {source.shape}."
    assert mic_array.shape == (batch_size, 1, 3), f"mic_array batch must be a 3D Tensor (batch_size, n_mics=1, 3). Found {mic_array.shape}."
    
    NUM_WALL = 6 # Shoebox room
    assert absorption.shape == (batch_size, 1, NUM_WALL), f"Absorption must be a 3D Tensor of shape (batch_size, n_bands=1, n_walls=6). Found {absorption.shape}."

    try:
        assert not torch.isnan(room).any(), "NaNs detected in room tensor"
        assert not torch.isnan(source).any(), "NaNs detected in source tensor"
        assert not torch.isnan(mic_array).any(), "NaNs detected in mic_array tensor"
        assert not torch.isnan(absorption).any(), "NaNs detected in absorption tensor"
    except AssertionError as e:
        print("room",room)
        print("source",source)
        print("mic_array",mic_array)
        print("absorption",absorption)
        raise e


def _batch_compute_image_sources(
    room: torch.Tensor,
    source: torch.Tensor,
    max_order: int,
    absorption : torch.Tensor,
    scatter: Optional[torch.Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Compute image sources in a shoebox-like room. Only one frequency band for attentuation is supported for now.

    Args:
        room (torch.Tensor): The batch of 1D Tensors to determine the room size. (batch_size, 3)
        source (torch.Tensor): The batch of coordinates of the sound source. (batch_size, 3)
        max_order (int): The maximum number of reflections of the source.
        absorption (torch.Tensor): The absorption coefficients of wall materials. (batch_size, n_band, n_walls=6)
        scatter (torch.Tensor): The scattering coefficients of wall materials. (batch_size, n_band, 6). If ``None``, it is not
            used in image source computation. (Default: ``None``)

    Returns:
        (torch.Tensor): The coordinates of all image sources within ``max_order`` number of reflections.
            Tensor with dimensions `(num_image_source, D)`.
        (torch.Tensor): The attenuation of corresponding image sources. Tensor with dimensions
            `(num_band, num_image_source)`.
    """
    batch_size = room.shape[0]
    n_band = absorption.shape[1]
    
    if scatter is None: tr = torch.sqrt(1 - absorption)
    else: tr = torch.sqrt(1 - absorption) * torch.sqrt(1 - scatter)

    # Constructing the grid
    ind = torch.arange(-max_order, max_order + 1, device=source.device)
        # Removed 2D capabilities for now
    xyz = torch.meshgrid(ind, ind, ind, indexing="ij") # Make a grid as big as the max_order
    xyz = torch.stack([c.reshape((-1,)) for c in xyz], dim=-1)
    xyz = xyz[xyz.abs().sum(dim=-1) <= max_order] # Remove the points that are too far away
    
    xyz=xyz.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, n_image_source (4991 for max order 15), 3)
    n_image_sources = xyz.shape[1]

    # compute locations of image sources
    d = room.unsqueeze(1) # (batch_size, 1, 3)
    s = source.unsqueeze(1) # (batch_size, 1, 3)
    batch_img_loc = torch.where(xyz % 2 == 1, d * (xyz + 1) - s, d * xyz + s) # (batch_size, n_image_source, 3)

    # attenuation
    exp_lo = abs(torch.floor((xyz / 2))).unsqueeze(1) # (batch_size, num_band=1, n_image_source, 3)
    exp_hi = abs(torch.floor((xyz + 1) / 2)).unsqueeze(1) # (batch_size, num_band=1, n_image_source, 3)
    t_lo = tr[:,:, ::2].unsqueeze(2).expand(-1,-1, n_image_sources, -1)  # (batch_size, num_band=1, n_image_source, left walls = 6/2 = 3)
    t_hi = tr[:,:, 1::2].unsqueeze(2).expand(-1,-1, n_image_sources, -1)  # (batch_size, num_band=1, n_image_source, right walls = 6/2 = 3)

    batch_att = torch.prod((t_lo**exp_lo) * (t_hi**exp_hi), dim=-1)  # (batch_size, num_band, num_image_source)

    assert batch_img_loc.shape == (batch_size, n_image_sources, 3), f"Expecting ({batch_size}, {n_image_sources}, 3). Found {batch_img_loc.shape}."
    assert batch_att.shape == (batch_size, n_band, n_image_sources), f"Expecting ({batch_size}, {n_band}, {n_image_sources}). Found {batch_att.shape}."

    del xyz, exp_lo, exp_hi, t_lo, t_hi, tr, d, s

    return batch_img_loc, batch_att


def batch_simulate_rir_ism(batch_room_dimensions: torch.Tensor,
                           batch_mic_position : torch.Tensor,
                           batch_source_position : torch.Tensor,
                           batch_absorption : torch.Tensor,
                           max_order : int, fs : float = 16000.0, sound_speed: float = 343.0,
                           output_length: Optional[int] = None, window_length: int = 81,
                           start_from_ir_onset : bool = False,
                           normalized_distance : bool = False
) -> Tensor:
    """
    Simulate room impulse responses (RIRs) using image source method (ISM).
    I'm just not doing multi-band processing for now
    look up torchaudio.prototype.functional.simulate_rir_ism if you're interested
    
    Args:
        batch_room_dimensions must be a 2D Tensor (batch_size, 3).
        batch_mic_position must be a 3D Tensor (batch_size, n_mics=1, 3).
        batch_source_position must be a 2D Tensor (batch_size, 3).
        batch_absorption must be a 3D Tensor of shape (batch_size, 1 or 7, n_walls=6). Walls are `"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.
        max_order (int): The maximum number of reflections of the source.
        fs (float): The sample rate of the RIRs. (Default: ``16000.0``)
        sound_speed (float): The speed of sound. (Default: ``343.0``)
        output_length (int, optional): The length of the output RIRs. (Default: ``None``)
        window_length (int): The length of the hann window. (Default: ``81``)
        lp_cutoff_frequency (int): The cutoff frequency of the low-pass filter. (Default: ``None``)

    Returns:
        (torch.Tensor): batch of rir (batch_size, max_rir_length)
    """
    _batch_validate_inputs(batch_room_dimensions, batch_mic_position, batch_source_position, batch_absorption)
    
    batch_img_location, batch_att = _batch_compute_image_sources(batch_room_dimensions, batch_source_position, max_order, batch_absorption) # returns (batch_size, n_image_source, 3[x,y,z]) , (batch_size, n_band=1, n_image_source)

    # compute distances between image sources and microphones
    vec = batch_img_location[:,:, None, :] - batch_mic_position[:, None, :, :]
    batch_dist = torch.linalg.norm(vec, dim=-1)  # (batch_size, n_image_source, n_mics=1)
    del vec
    batch_delay = batch_dist * fs / sound_speed  # (fractionnal delay) (batch_size, n_image_source, n_mics=1)
    if start_from_ir_onset:
        batch_IR_onset = fs * torch.linalg.norm(batch_mic_position.squeeze(1)-batch_source_position, dim=1) / sound_speed # squeezing n_channels for now

    #attenuate image sources
    if not normalized_distance:
        epsilon = 1e-10
        batch_img_src_att = batch_att[..., None] / (batch_dist[:, None, ...] + epsilon) # (batch_size, n_band, n_image_source, n_mics=1)
    else :
        batch_img_src_att = batch_att[..., None]
    del batch_dist, batch_att

    ##### MY OWN BATCH ISM IMPLEMENTATION ###########
    if output_length is not None: rir_length = output_length
    else: rir_length = torch.ceil(batch_delay.detach().max()).int() + window_length
    if rir_length > 6000: # for memory reasons, with rir max order 15, batch_size 9 and sample rate 48000 I can't go above 11000 (0.229 seconds)
        rir_length = 6000 # for memory reasons, with rir max order 15, batch_size 16 and sample rate 16000 I can't go above 6000 (0.393 seconds)

    #### Prepare Fractional delays
    n = torch.arange(0, rir_length, device=batch_delay.device) # (rir_length)
    # leave space for the convolution window.
    n = n - window_length//2
    n = n.unsqueeze(0).expand(batch_delay.shape[0], -1) # (batch_size, rir_length)
    if start_from_ir_onset:
        # translate the IR onset to be at t = window_length//2 + 1
        n = n + batch_IR_onset.unsqueeze(1).expand(-1,rir_length) 
    n = n.unsqueeze(2).unsqueeze(3).expand(-1, -1, batch_delay.shape[1], batch_delay.shape[2]) # (batch_size, rir_length, n_image_source, n_mics=1)
    # translate each n by the amount of delay corresponding to each image source so when convolved with the filter they'll be in the correct place in the rir before ein-summation with the absorption
    n = n - batch_delay.unsqueeze(1).expand(-1,rir_length,-1,-1) # (batch_size, rir_length, n_image_source, n_mics=1)
    del batch_delay

    #### Sub-sample backpropagatable convolution for fractional delays.
    # TODO For multiband processing, we will eventually need to create a different batch_indiv_IRS for each band using a different filter, and then sum them up.
    #      There should be a more elegant way of doing this that won't cost 7x the VRAM.
    batch_indiv_IRS=torch.where(torch.abs(n) <= window_length//2,
                                BP_filter(n, fs, 8000, 80, window_length),
                                # LP_filter(n, fs, window_length, lp_cutoff_frequency),
                                n.new_zeros(1)) # (batch_size, rir_length, n_image_source, n_mics=1) 
    del n

    # here get rid of multi-band processing and multi-channel processing (for now)
    batch_indiv_IRS = batch_indiv_IRS.squeeze(dim=3) # (batch_size, rir_length, n_image_source, n_mics=1) -> (batch_size, rir_length, n_image_source)
    batch_img_src_att = batch_img_src_att.squeeze(dim=(1,3)) # (batch_size, n_bands, n_image_source, n_mics=1) -> (batch_size, n_image_source)
    
    # Sum the img sources
    # TODO implement opt_einsum.
    batch_rir = torch.einsum('bri,bi->br', batch_indiv_IRS, batch_img_src_att) # (batch_size, rir_length)

    del batch_indiv_IRS, batch_img_src_att
    return batch_rir # (batch_size, rir_length)
    