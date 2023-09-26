'''
This is a reimplementation of torchaudio.prototype.functional.simulate_rir_ism
It is modified to work with pytorch tensors with autograd.
The original function is not differentiable because of C code, and therefore cannot be used in a neural network.
'''

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

# Untouched function from torch implementation # I HAD A GLITCH! ?? in the torch where.
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
def _frac_delay(delay: torch.Tensor, delay_i: torch.Tensor, delay_filter_length: int):
    """Compute fractional delay of impulse response signal.

    Args:
        delay (torch.Tensor): The time delay Tensor in samples. (n_image_sources)
        delay_i (torch.Tensor): The integer part of delay.  (n_image_sources)
        delay_filter_length (int): The window length for sinc function.

    Returns:
        (torch.Tensor): The impulse response Tensor for all image sources. (n_image_sources, delay_filter_length)
    """
    if delay_filter_length % 2 != 1:
        raise ValueError("The filter length must be odd")

    pad = delay_filter_length // 2
    n = torch.arange(-pad, pad + 1, device=delay.device) + delay_i[..., None]
    delay = delay[..., None]

    return torch.special.sinc(n - delay) * _hann(n - delay, 2 * pad)

# Changed to ask Mono-band absorption because I didn't want to manage multi-band processing for now.
# Same for mic_array, I only want one mic.
# If a tensor of shape (1,) is fed, it will not repeat it to size n_walls.
def _validate_inputs(
    room: torch.Tensor, source: torch.Tensor, mic_array: torch.Tensor, absorption: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Validate dimensions of input arguments, and normalize different kinds of absorption into the same dimension.

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        absorption (float or torch.Tensor): The absorption coefficients of wall materials.
            If the dtype is ``float``, the absorption coefficient is identical for all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, where the values represent
            absorption coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``,
            and ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be `(7, 6)`, where 7 represents the number of octave bands.

    Returns:
        (torch.Tensor): The absorption Tensor. The shape is `(1, 6)` for single octave band case,
            or `(7, 6)` for multi octave band case.
    """
    assert(room.device==source.device==mic_array.device)
    
    if room.ndim != 1:
        raise ValueError(f"room must be a 1D Tensor. Found {room.shape}.")
    D = room.shape[0]
    if D != 3:
        raise ValueError(f"room must be a 3D room. Found {room.shape}.")
    num_wall = 6 # Shoebox room
    if source.shape[0] != D:
        raise ValueError(f"The shape of source must be `(3,)`. Found {source.shape}")
    if mic_array.ndim != 2:
        raise ValueError(f"mic_array must be a 2D Tensor. Found {mic_array.shape}.")
    if mic_array.shape[1] != D:
        raise ValueError(f"The second dimension of mic_array must be 3. Found {mic_array.shape}.")
    if mic_array.shape[0] > 1:
        raise ValueError(f"I only support single channel because that's what I need for now. Found {mic_array.shape[0]}.")
    
    if isinstance(absorption, float):
        absorption = torch.ones(1, num_wall, device=room.device) * absorption
    elif isinstance(absorption, torch.Tensor) and absorption.ndim == 0:
        assert(absorption.device==room.device)
        absorption = absorption.repeat(1, num_wall)
    elif isinstance(absorption, torch.Tensor) and absorption.ndim == 1:
        assert(absorption.device==room.device)
        if absorption.shape[0] != num_wall:
            raise ValueError(
                "The shape of absorption must be `(6,)` if it is a 1D Tensor." f"Found the shape {absorption.shape}."
            )
        absorption = absorption.unsqueeze(0)
    elif isinstance(absorption, torch.Tensor) and absorption.ndim == 2:
        raise ValueError("Multi-band absorption is not supported yet. Please make absorption a float/1D tensor.")
        # if absorption.shape != (7, num_wall):
        #     raise ValueError(
        #         "The shape of absorption must be `(7, 6)` if it is a 2D Tensor."
        #         f"Found the shape of room is {D} and shape of absorption is {absorption.shape}."
        #     )
        # absorption = absorption
    else:
        raise TypeError(f"Absorption must be a float or a 1D Tensor of shape (1,n_walls=6). Found {type(absorption)}.")
        # absorption = absorption
    return absorption

# Fixed a glitch from the original implementation where the attenuation was not squared?
# This makes Pytorch and Pyroomacoustics agree on the attenuation. (see demo in main())
def _compute_image_sources(
    room: torch.Tensor,
    source: torch.Tensor,
    max_order: int,
    absorption,
    scatter: Optional[torch.Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Compute image sources in a shoebox-like room.

    Args:
        room (torch.Tensor): The 1D Tensor to determine the room size. The shape is
            `(D,)`, where ``D`` is 2 if room is a 2D room, or 3 if room is a 3D room.
        source (torch.Tensor): The coordinate of the sound source. Tensor with dimensions
            `(D)`.
        max_order (int): The maximum number of reflections of the source.
        absorption (torch.Tensor): The absorption coefficients of wall materials.
            ``absorption`` is a Tensor with dimensions `(num_band, num_wall)`.
            The shape options are ``[(1, 4), (1, 6), (7, 4), (7, 6)]``.
            ``num_band`` is `1` if the coefficients is the same for all frequencies, or is `7`
            if the coefficients are different to different frequencies. `7` refers to the default number
            of octave bands. (See note in `simulate_rir_ism` method).
            ``num_wall`` is `4` if the room is a 2D room, representing absorption coefficients
            of ``"west"``, ``"east"``, ``"south"``, and ``"north"`` walls, respectively.
            Or it is `6` if the room is a 3D room, representing absorption coefficients
            of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.
        scatter (torch.Tensor): The scattering coefficients of wall materials.
            The shape of ``scatter`` must match that of ``absorption``. If ``None``, it is not
            used in image source computation. (Default: ``None``)

    Returns:
        (torch.Tensor): The coordinates of all image sources within ``max_order`` number of reflections.
            Tensor with dimensions `(num_image_source, D)`.
        (torch.Tensor): The attenuation of corresponding image sources. Tensor with dimensions
            `(num_band, num_image_source)`.
    """
    if scatter is None:
        tr = torch.sqrt(1 - absorption)
    else:
        tr = torch.sqrt(1 - absorption) * torch.sqrt(1 - scatter)

    ind = torch.arange(-max_order, max_order + 1, device=source.device)
    if room.shape[0] == 2:
        xyz = torch.meshgrid(ind, ind, indexing="ij")
    else:
        xyz = torch.meshgrid(ind, ind, ind, indexing="ij") # Make a grid as big as the max_order

    xyz = torch.stack([c.reshape((-1,)) for c in xyz], dim=-1)
    xyz = xyz[xyz.abs().sum(dim=-1) <= max_order] # Remove the points that are too far away

    # compute locations of image sources
    d = room[None, :]
    s = source[None, :]

    img_loc = torch.where(xyz % 2 == 1, d * (xyz + 1) - s, d * xyz + s)

    # attenuation
    exp_lo = abs(torch.floor((xyz / 2)))
    exp_hi = abs(torch.floor((xyz + 1) / 2))
    t_lo = tr[:, ::2].unsqueeze(1).repeat(1, xyz.shape[0], 1)  # (num_band, left walls)
    t_hi = tr[:, 1::2].unsqueeze(1).repeat(1, xyz.shape[0], 1)  # (num_band, right walls)
    att = torch.prod((t_lo**exp_lo) * (t_hi**exp_hi), dim=-1)  # (num_band, num_image_source)

    return img_loc, att

# New function
def _add_all_mini_irs_together(mini_irs, delay_i, rir_length,delay_filter_length):
        '''
        does what torch.ops.torchaudio._simulate_rir does, but is backpropagatable
        returns an rir of shape (rir_length)
        '''
        rir = torch.zeros(rir_length, device=mini_irs.device) # I am just assuming n_mics=1 for now because thats what I need
        for i in range(mini_irs.shape[1]): # for each image source
            mini_ir_start = delay_i[i]
            mini_ir_end = mini_ir_start + delay_filter_length
            rir[mini_ir_start:mini_ir_end] += mini_irs[0,i,0,:]
        return rir

# Modified to be backpropagatable (removed the C code)
def simulate_rir_ism(
    room: torch.Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
    max_order: int,
    absorption: Union[float, torch.Tensor],
    output_length: Optional[int] = None,
    delay_filter_length: int = 81,
    #center_frequency: Optional[torch.Tensor] = None, # Not managing multi-band processing for now
    sound_speed: float = 343.0,
    sample_rate: float = 16000.0,
    return_imgs_and_att_for_demo: bool = False
) -> Tensor:
    r"""Compute Room Impulse Response (RIR) based on the *image source method* :cite:`allen1979image`.
    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.
    Modified to be backpropagatable (not using C code anymore, for better or for worse!)

    .. devices:: CPU, CUDA (!)

    .. properties:: TorchScript

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        max_order (int): The maximum number of reflections of the source.
        absorption (float or torch.Tensor): The *absorption* :cite:`wiki:Absorption_(acoustics)`
            coefficients of wall materials for sound energy.
            If the dtype is ``float``, the absorption coefficient is identical for all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, where the values represent
            absorption coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``,
            and ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be `(7, 6)`, where 7 represents the number of octave bands.
        output_length (int or None, optional): The output length of simulated RIR signal. If ``None``,
            the length is defined as

            .. math::
                \frac{\text{max\_d} \cdot \text{sample\_rate}}{\text{sound\_speed}} + \text{delay\_filter\_length}

            where ``max_d`` is the maximum distance between image sources and microphones.
        delay_filter_length (int, optional): The filter length for computing sinc function. (Default: ``81``)
        #center_frequency (torch.Tensor, optional): The center frequencies of octave bands for multi-band walls.
        #    Only used when ``absorption`` is a 2D Tensor.
        sound_speed (float, optional): The speed of sound. (Default: ``343.0``)
        sample_rate (float, optional): The sample rate of the generated room impulse response signal.
            (Default: ``16000.0``)

    Returns:
        (torch.Tensor): The simulated room impulse response waveform. Tensor with dimensions
        `(channel, rir_length)`.

    # Note:
    #     If ``absorption`` is a 2D Tensor and ``center_frequency`` is set to ``None``, the center frequencies
    #     of octave bands are fixed to ``[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]``.
    #     Users need to tune the values of ``absorption`` to the corresponding frequencies.
    """
    absorption = _validate_inputs(room, source, mic_array, absorption) # (n_band=1, n_wall=6)
    img_location, att = _compute_image_sources(room, source, max_order, absorption) # (n_image_source, 3[x,y,z]) , (n_band=1, n_image_source)
    # compute distances between image sources and microphones
    vec = img_location[:, None, :] - mic_array[None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)  # (n_image_source, n_mics=1)

    #attenuate image sources
    img_src_att = att[..., None] / dist[None, ...]  # (n_band=1, n_image_source, n_mics=1)

    # # separate delays in integer / frac part 
    delay = dist * sample_rate / sound_speed  # time to delay in samples (fractionnal delay) (n_image_source, n_mics=1)
    delay_i = torch.ceil(delay)  # integer part (n_image_source, n_mics=1)
    
    # compute the shorts IRs corresponding to each image source
    mini_irs = img_src_att[..., None] * _frac_delay(delay, delay_i, delay_filter_length)[None, ...] # (n_band=1, n_image_source, n_mics=1, delay_filter_length)
    rir_length = int(delay_i.max() + mini_irs.shape[-1]) # full length of the rir signal
    
    # sum up rir signals of all image sources into one waveform.
    rir = _add_all_mini_irs_together(mini_irs, delay_i.type(torch.int32), rir_length,delay_filter_length) # (rir_length) # I am just assuming n_mics=1 for now
    
    # I'm just not doing multi-band processing for now
    # look up torchaudio.prototype.functional.simulate_rir_ism if you're interested

    if output_length is not None:
        if output_length > rir.shape[-1]:
            rir = torch.nn.functional.pad(rir, (0, output_length - rir.shape[-1]), "constant", 0.0)
        else:
            rir = rir[..., :output_length]

    if return_imgs_and_att_for_demo:
        return rir , img_location, att
    else:
        return rir

    # My deprecated Frequential implementation (it just sucks), it used as many frequency bands as there were frequency bins.

    # freq = torch.fft.rfftfreq(rfftfreq_samplebins, 1/sample_rate, device=source.device)
    # i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

    # exponential=torch.exp(-i*2*math.pi*freq[None,None, ...]*dist[None, ...]/sound_speed)
    # img_src_att=torch.squeeze(img_src_att)
    # exponential=torch.squeeze(exponential)
    # product=torch.mul(img_src_att/math.sqrt(2*math.pi),torch.transpose(exponential, 0, 1))
    # h_f=torch.sum(product,1)
    # irfft_h_f=torch.fft.irfft(h_f)

    # return irfft_h_f

def torch_ism(room_dimensions: torch.Tensor,
              mic_position : torch.Tensor,
              source_position : torch.Tensor,
              sample_rate : float, max_order : int = 10,
              absorption : float = 0.3, return_imgs_and_att_for_demo : bool = False):
    '''Forward pass through my backpropagatable pytorch ism implementation'''
    torch_rir = simulate_rir_ism(
        room_dimensions ,
        mic_position,
        source_position[None,:],
        max_order=max_order,
        absorption=absorption,
        sample_rate=float(sample_rate),
        return_imgs_and_att_for_demo=return_imgs_and_att_for_demo,
    )
    return torch_rir

# def multiprocessing_torch_ism(room_dimensions: torch.Tensor,
#               mic_position : torch.Tensor,
#               source_position : torch.Tensor,
#               sample_rate : float, max_order : int = 10,
#               absorption : float = 0.3):
#     '''Forward pass through my backpropagatable pytorch ism implementation'''
#     torch_rir = simulate_rir_ism(
#         room_dimensions ,
#         mic_position,
#         source_position[None,:],
#         max_order=max_order,
#         absorption=absorption,
#         sample_rate=float(sample_rate)
#     )
#     return torch_rir

def main():
    '''demo'''

    room_dimensions=torch.tensor([5.0, 5.0, 5.0])
    mic_position=torch.tensor([2.5, 2.5, 2.5])
    source_position=torch.tensor([1.0, 1.0, 1.0])
    sample_rate=16000
    max_order=18
    absorption=0.3
    rir, imgs, att = torch_ism(room_dimensions,mic_position,source_position,sample_rate, max_order=max_order,absorption=absorption, return_imgs_and_att_for_demo=True)
    rir=rir.numpy()

    import pyroomacoustics as pra
    import numpy as np
    room_dim = [5.0, 5.0, 5.0]
    wall_material = pra.Material(energy_absorption=absorption)
    
    room = pra.ShoeBox(room_dim, materials=wall_material, fs=sample_rate, max_order=max_order)
    mic_array_position = [2.5, 2.5, 2.5]
    source_position = [1.0, 1.0, 1.0] 
    room.add_source(source_position)
    mic_array = pra.MicrophoneArray(np.array([mic_array_position]).T, room.fs)
    room.add_microphone_array(mic_array)
    room.compute_rir()
    label_att = room.sources[0].damping
    label_rir = room.rir[0][0]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.title("Comparing Pyroom and Pytorch attenuation")
    plt.scatter(att,label_att, label='attenuation values')
    plt.plot([0,1],[0,1], label='regression')
    plt.plot(np.arange(0,1+1/(max_order+1),1/(max_order+1)),np.power(np.arange(0,1+1/(max_order+1),1/(max_order+1)),2), label='before squaring att in _compute_image_sources')
    plt.grid(True, ls=':', alpha=0.5)
    plt.axis('equal')
    plt.xlabel("pytorch attenuation values")
    plt.ylabel("pyroom attenuation values")
    plt.legend()
    
    plt.figure()
    plt.title("Comparing Pyroom and Pytorch RIRs")
    plt.plot(rir, alpha=0.5,c='blue',label="pytorch")
    plt.plot(label_rir, alpha=0.5, c='orange',label="pyroom")
    plt.plot(abs(rir-label_rir), c='green',label="Difference")
    plt.legend()
    plt.grid(True, ls=':', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()