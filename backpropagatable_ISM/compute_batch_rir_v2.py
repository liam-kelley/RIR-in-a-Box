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

from tools.pyLiam.LKTimer import LKTimer
timer=LKTimer()

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
                           max_order : int, sample_rate : float = 16000.0, sound_speed: float = 343.0,
                           output_length: Optional[int] = None, delay_filter_length: int = 81,
                           center_frequency: Optional[torch.Tensor] = None,
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
        sample_rate (float): The sample rate of the RIRs. (Default: ``16000.0``)
        sound_speed (float): The speed of sound. (Default: ``343.0``)
        output_length (int, optional): The length of the output RIRs. (Default: ``None``)
        delay_filter_length (int): The length of the delay filter. (Default: ``81``)

    Returns:
        (torch.Tensor): batch of rir (batch_size, max_rir_length)
    """
    _batch_validate_inputs(batch_room_dimensions, batch_mic_position, batch_source_position, batch_absorption)
    
    batch_img_location, batch_att = _batch_compute_image_sources(batch_room_dimensions, batch_source_position, max_order, batch_absorption) # returns (batch_size, n_image_source, 3[x,y,z]) , (batch_size, n_band=1, n_image_source)

    # compute distances between image sources and microphones
    vec = batch_img_location[:,:, None, :] - batch_mic_position[:, None, :, :]
    batch_dist = torch.linalg.norm(vec, dim=-1)  # (batch_size, n_image_source, n_mics=1)
    del vec
    batch_delay = batch_dist * sample_rate / sound_speed  # (fractionnal delay) (batch_size, n_image_source, n_mics=1)

    #attenuate image sources
    epsilon = 1e-10
    batch_img_src_att = batch_att[..., None] / (batch_dist[:, None, ...] + epsilon) # (batch_size, n_band, n_image_source, n_mics=1)
    del batch_dist, batch_att

    ##### MY OWN BATCH ISM IMPLEMENTATION ###########
    if output_length is not None: rir_length = output_length
    else: rir_length = torch.ceil(batch_delay.detach().max()).int() + delay_filter_length
    if rir_length > 6000: # for memory reasons, with rir max order 15, batch_size 9 and sample rate 48000 I can't go above 11000 (0.229 seconds)
        rir_length = 6000 # for memory reasons, with rir max order 15, batch_size 16 and sample rate 16000 I can't go above 6000 (0.393 seconds)

    #### Prepare Fractional delays
    my_arange_tensor = torch.arange(rir_length, device=batch_delay.device)-(delay_filter_length // 2) # substraction to account for the delay filter length and to have similar results as pyroomacoustics.
    my_arange_tensor = my_arange_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(batch_delay.shape[0], -1, batch_delay.shape[1], -1) # (batch_size, rir_length, n_image_source, n_mics=1)
    my_arange_tensor = my_arange_tensor-batch_delay.unsqueeze(1).expand(-1,my_arange_tensor.shape[1],-1,-1) # (batch_size, rir_length, n_image_source, n_mics=1)
    del batch_delay

    # create hann window tensor
    hann_tensor=torch.where(torch.abs(my_arange_tensor) <= delay_filter_length//2,
                            0.5 * (1 + torch.cos(math.pi * my_arange_tensor / (delay_filter_length//2))), # Hann window fix
                            my_arange_tensor.new_zeros(1)) # (batch_size, rir_length, n_image_source, n_mics=1)
    batch_indiv_IRS = torch.special.sinc(my_arange_tensor) * hann_tensor # (batch_size, rir_length, n_image_source, n_mics=1)
    del my_arange_tensor, hann_tensor

    # get rid of multi-band processing, and multi-channel processing here
    batch_indiv_IRS = batch_indiv_IRS.squeeze(dim=3) # (batch_size, rir_length, n_image_source, n_mics=1) -> (batch_size, rir_length, n_image_source)
    batch_img_src_att = batch_img_src_att.squeeze(dim=(1,3)) # (batch_size, n_bands, n_image_source, n_mics=1) -> (batch_size, n_image_source)
    # Sum the img sources
    batch_rir = torch.einsum('bri,bi->br', batch_indiv_IRS, batch_img_src_att) # (batch_size, rir_length)

    # # multi-band processing
    # if batch_absorption.shape[1] > 1:
    #     if center_frequency is None: center = torch.tensor([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0], dtype=batch_room_dimensions.dtype, device=batch_room_dimensions.device)
    #     else: center = center_frequency
    #     # n_fft is set to 512 by default.
    #     from torchaudio.functional import fftconvolve
    #     filters = make_rir_filter(center, sample_rate, n_fft=512)
    #     batch_rir = fftconvolve(batch_rir, filters.unsqueeze(0).unsqueeze(2).expand(batch_rir.shape[0],1, batch_rir.shape[2], 1), mode="same")
    # # sum up batch_rir signals of all image sources into one waveform.
    # batch_rir = batch_rir.sum(0)

    del batch_indiv_IRS, batch_img_src_att
    return batch_rir # (batch_size, rir_length)
    