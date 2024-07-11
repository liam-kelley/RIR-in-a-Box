'''
This is a reimplementation of torchaudio.prototype.functional.simulate_rir_ism
It is modified to work batchwise with pytorch tensors using autograd.

The original functions were not differentiable and therefore could not be used in a DDSP framework
It could be nice to integrate this into pytorch natively.

Usage example : TODO
'''

from typing import Optional, Tuple
import torch
from torch import Tensor
from backpropagatable_ISM.filters import LP_filter, BP_filter


def _batch_validate_shoebox_inputs(room: Tensor, mic_array: Tensor,
                                   source: Tensor, absorption: Tensor):
    '''
    This function performs sanity checks on the inputs for our batch shoebox ISM inference.
    
    This function is reimplemented from torchaudio, with more constraints due to our implementation
     only supporting mono band absorption, 3D dimensions, and 1 mic.
    '''
    # Device sanity chcek
    assert(room.device==source.device==mic_array.device==absorption.device),\
        "All inputs room, source, mic array and absorption must be on the same device."

    # Tensor sanity check
    batch_size = room.shape[0]
    assert room.shape == (batch_size, 3), f"room batch must be a 2D Tensor (batch_size, 3). Found {room.shape}."
    assert source.shape == (batch_size, 3), f"source batch must be a 2D Tensor (batch_size, 3). Found {source.shape}."
    assert mic_array.shape == (batch_size, 1, 3), f"mic_array batch must be a 3D Tensor (batch_size, n_mics=1, 3). Found {mic_array.shape}."
    NUM_WALL = 6 # Shoebox room only.
    assert absorption.shape == (batch_size, 1, NUM_WALL), f"Absorption must be a 3D Tensor of shape (batch_size, n_bands=1, n_walls=6). Found {absorption.shape}."

    # NaN sanity check
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


def _batch_compute_shoebox_image_sources(
    room: Tensor,
    source: Tensor,
    max_order: int,
    absorption : Tensor,
    scatter: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Compute image sources in a shoebox-like room.
    
    This function is adapted from torchaudio and pyroomacoustics.
    Only one frequency band for attentuation per wall is supported for now.

    Args:
        room (Tensor): The batch of 1D Tensors to determine the room size. (batch_size, 3)
        source (Tensor): The batch of coordinates (from 0 to 1) of the sound source. (batch_size, 3)
        max_order (int): The maximum number of reflections of the source.
        absorption (Tensor): The absorption coefficients of wall materials.
            (batch_size, n_band, n_walls=6)
        scatter (Tensor): The scattering coefficients of wall materials.
            (batch_size, n_band, 6). If ``None``, it is not
            used in image source computation. (Default: ``None``)

    Returns:
        (Tensor): The coordinates of all image sources within ``max_order`` number of reflections.
            Tensor with dimensions `(num_image_source, D)`.
        (Tensor): The attenuation of corresponding image sources. Tensor with dimensions
            `(num_band, num_image_source)`.
    """
    batch_size = room.shape[0]
    n_band = absorption.shape[1]

    if scatter is None:
        tr = torch.sqrt(1 - absorption)
    else:
        tr = torch.sqrt(1 - absorption) * torch.sqrt(1 - scatter)

    # Construct the grid
    ind = torch.arange(-max_order, max_order + 1, device=source.device)
    # Make a grid as big as the max_order
    xyz = torch.meshgrid(ind, ind, ind, indexing="ij")
    # Remove the grid points that correspond to too many reflections (measured via 1D distance)
    xyz = torch.stack([c.reshape((-1,)) for c in xyz], dim=-1)
    xyz = xyz[xyz.abs().sum(dim=-1) <= max_order]

    xyz = xyz.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, n_image_source (4991 for max order 15), 3)
    n_image_sources = xyz.shape[1]

    # Compute locations of image sources
    d = room.unsqueeze(1) # (batch_size, 1, 3)
    s = source.unsqueeze(1) # (batch_size, 1, 3)
    batch_img_loc = torch.where(xyz % 2 == 1, d * (xyz + 1) - s, d * xyz + s) # (batch_size, n_image_source, 3)

    # Get wall attenuation for each source by counting reflections per wall in a clever way
    exp_lo = abs(torch.floor((xyz / 2))).unsqueeze(1) # (batch_size, num_band=1, n_image_source, 3)
    exp_hi = abs(torch.floor((xyz + 1) / 2)).unsqueeze(1) # (batch_size, num_band=1, n_image_source, 3)
    t_lo = tr[:,:, ::2].unsqueeze(2).expand(-1,-1, n_image_sources, -1)  # (batch_size, num_band=1, n_image_source, left walls = 6/2 = 3)
    t_hi = tr[:,:, 1::2].unsqueeze(2).expand(-1,-1, n_image_sources, -1)  # (batch_size, num_band=1, n_image_source, right walls = 6/2 = 3)

    batch_att = torch.prod((t_lo**exp_lo) * (t_hi**exp_hi), dim=-1)  # (batch_size, num_band, num_image_source)

    # Sanity checks
    assert batch_img_loc.shape == (batch_size, n_image_sources, 3), f"Expecting ({batch_size}, {n_image_sources}, 3). Found {batch_img_loc.shape}."
    assert batch_att.shape == (batch_size, n_band, n_image_sources), f"Expecting ({batch_size}, {n_band}, {n_image_sources}). Found {batch_att.shape}."

    return batch_img_loc, batch_att


def batch_simulate_rir_ism(batch_room_dimensions: Tensor,
                           batch_mic_position : Tensor,
                           batch_source_position : Tensor,
                           batch_absorption : Tensor,
                           max_order : int, fs : float = 16000.0, sound_speed: float = 343.0,
                           output_length: Optional[int] = 3968, window_length: int = 81,
                           start_from_ir_onset : bool = False,
                           normalized_distance : bool = False
) -> Tensor:
    """
    Simulate room impulse responses (RIRs) using image source method (ISM).
    Adapted from torchaudio.prototype.functional.simulate_rir_ism.
    Not reimplemented:
        Multi-band processing
        Multi-channel microphones
    Although it shouldn't be hard to reimplement.
    
    Args:
        batch_room_dimensions must be a 2D Tensor (batch_size, 3).
        batch_mic_position must be a 3D Tensor (batch_size, n_mics=1, 3).
        batch_source_position must be a 2D Tensor (batch_size, 3).
        batch_absorption must be a 3D Tensor of shape (batch_size, n_bands=1, n_walls=6). Walls are `"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.
        max_order (int): The maximum number of reflections of the source.
        fs (float): The sample rate of the RIRs. (Default: ``16000.0``)
        sound_speed (float): The speed of sound. (Default: ``343.0``)
        output_length (int, optional): The length of the output RIRs. A fixed output length is strongly recommended due to the alternative's inconsistent backpropagation and memory issues on gpu. (Default: ``3968``)
        window_length (int): The length of the hann window. (Default: ``81``)
        start_from_ir_onset (bool): Experimental setting that saves on gpu memory... At the cost of needing individual padding for each item in the batch... Probably only useful for inference. (Default: ``False``)
        normalized_distance (bool): Experimental setting that MIGHT help convergence during training. (Default: ``False``)
        
    Returns:
        (Tensor): batch of rir (batch_size, max_rir_length)
    """
    _batch_validate_shoebox_inputs(batch_room_dimensions, batch_mic_position, batch_source_position, batch_absorption)

    batch_img_location, batch_att = _batch_compute_shoebox_image_sources(batch_room_dimensions, batch_source_position, max_order, batch_absorption) # returns (batch_size, n_image_source, [x,y,z]) , (batch_size, n_band=1, n_image_source)

    # -------------------- Distances  -------------------- #
    # Compute distances between image sources and microphones
    vec = batch_img_location[:,:, None, :] - batch_mic_position[:, None, :, :]
    batch_dist = torch.linalg.norm(vec, dim=-1)  # (batch_size, n_image_source, n_mics=1)
    del vec

    # Get exact time delay between src and mic
    batch_delay = batch_dist * fs / sound_speed  # (batch_size, n_image_source, n_mics=1)
    if start_from_ir_onset:
        batch_IR_onset = fs * torch.linalg.norm(batch_mic_position.squeeze(1)-batch_source_position, dim=1) / sound_speed # squeezing n_channels for now.

    # -------------------- Image source attenuation  -------------------- #
    if not normalized_distance:
        # Attenuate image sources according to wall reflections and distance.
        epsilon = 1e-10
        batch_img_src_att = batch_att[..., None] / (batch_dist[:, None, ...] + epsilon) # (batch_size, n_band, n_image_source, n_mics=1)
    else :
        # Experimental, only for training:
        # Only attenuate image sources according to wall refelctions.
        batch_img_src_att = batch_att[..., None]
    del batch_dist, batch_att

    # -------------------- Manage output length  -------------------- #
    if output_length is not None: rir_length = output_length
    else: rir_length = torch.ceil(batch_delay.detach().max()).int() + window_length
    if rir_length > 6000: # Safty net for variable length.
        rir_length = 6000 

    # -------------------- Prepare Fractional delays  -------------------- #
    n = torch.arange(0, rir_length, device=batch_delay.device) # (rir_length)
    
    # leave space for the convolution window, as in pyroomacoustics.
    n = n - window_length//2
    n = n.unsqueeze(0).expand(batch_delay.shape[0], -1) # (batch_size, rir_length)
    
    if start_from_ir_onset:
        # Experimental, probably only useful for inference, to save a bit on memory.
        # Translate the IR onset to be at t = window_length//2 + 1
        # Effectively, don't care about the zeroes before the signal starts, and pad them later.
        n = n + batch_IR_onset.unsqueeze(1).expand(-1,rir_length) 
    
    n = n.unsqueeze(2).unsqueeze(3).expand(-1, -1, batch_delay.shape[1], batch_delay.shape[2]) # (batch_size, rir_length, n_image_source, n_mics=1)
    
    # translate each "n" by the amount of delay corresponding to each image source
    # so when convolved with the filter they'll be in the correct place in the rir
    # before ein-summation with the absorption
    n = n - batch_delay.unsqueeze(1).expand(-1,rir_length,-1,-1) # (batch_size, rir_length, n_image_source, n_mics=1)
    del batch_delay

    # -------------------- Fractional delays  -------------------- #
    # Create for each image source a very sparse rir-length vector, with a filtered dirac at the appropriate time delay.
    # TODO For multiband processing, we will eventually need to create a different batch_indiv_IRS for each band using a different filter, and then sum them up.
    #      There should be a much more elegant way to do this that doesn't cost us 7x the VRAM, but I haven't figured it out yet.
    batch_indiv_IRS=torch.where(torch.abs(n) <= window_length//2,
                                BP_filter(n, fs, 8000, 80, window_length),
                                # LP_filter(n, fs, window_length, lp_cutoff_frequency),
                                n.new_zeros(1)) # (batch_size, rir_length, n_image_source, n_mics=1) 
    del n

    # quick fix to get rid of multi-band processing and multi-channel processing (for now)
    batch_indiv_IRS = batch_indiv_IRS.squeeze(dim=3) # (batch_size, rir_length, n_image_source, n_mics=1) -> (batch_size, rir_length, n_image_source)
    batch_img_src_att = batch_img_src_att.squeeze(dim=(1,3)) # (batch_size, n_bands, n_image_source, n_mics=1) -> (batch_size, n_image_source)
    
    # -------------------- Einsum image sources  -------------------- #
    # The images sources are summed together, while being multiplied by their respective absorption values.
    # This is done fast via ein-sum.
    # TODO implement opt_einsum.
    batch_rir = torch.einsum('bri,bi->br', batch_indiv_IRS, batch_img_src_att) # (batch_size, rir_length)

    del batch_indiv_IRS, batch_img_src_att
    return batch_rir # (batch_size, rir_length)
    