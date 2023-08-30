'''
This is a reimplementation of the pyroomacoustics function compute_rir. It is modified to work with pytorch tensors with autograd.
The original function is not differentiable, and therefore cannot be used in a neural network.
Absorption is not backpropagatable yet. It is a constant for now.
'''

import math
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

rfftfreq_samplebins=24000

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
    if room.ndim != 1:
        raise ValueError(f"room must be a 1D Tensor. Found {room.shape}.")
    D = room.shape[0]
    if D != 3:
        raise ValueError(f"room must be a 3D room. Found {room.shape}.")
    num_wall = 6
    if source.shape[0] != D:
        raise ValueError(f"The shape of source must be `(3,)`. Found {source.shape}")
    if mic_array.ndim != 2:
        raise ValueError(f"mic_array must be a 2D Tensor. Found {mic_array.shape}.")
    if mic_array.shape[1] != D:
        raise ValueError(f"The second dimension of mic_array must be 3. Found {mic_array.shape}.")
    if isinstance(absorption, float):
        absorption = torch.ones(1, num_wall) * absorption
    elif isinstance(absorption, Tensor) and absorption.ndim == 1:
        if absorption.shape[0] != num_wall:
            raise ValueError(
                "The shape of absorption must be `(6,)` if it is a 1D Tensor." f"Found the shape {absorption.shape}."
            )
        absorption = absorption.unsqueeze(0)
    elif isinstance(absorption, Tensor) and absorption.ndim == 2:
        if absorption.shape != (7, num_wall):
            raise ValueError(
                "The shape of absorption must be `(7, 6)` if it is a 2D Tensor."
                f"Found the shape of room is {D} and shape of absorption is {absorption.shape}."
            )
        absorption = absorption
    else:
        absorption = absorption
    return absorption

def _compute_image_sources(
    room: torch.Tensor,
    source: torch.Tensor,
    max_order: int,
    absorption,
    scatter: Optional[torch.Tensor] = None,
    sample_rate: float = 16000.0
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

    assert(len(absorption)==7)
    freq = torch.fft.rfftfreq(rfftfreq_samplebins, 1/sample_rate)

    absorption_per_freq=[]
    for f in freq:
        power=0
        while not f <= 125*(2**power):
            power+=1
        absorption_per_freq.append(absorption[power])
        # if f < 62.5*2**power:
        #     absorption_per_freq_temp=absorption[0]
        # elif f < 125:
        #     absorption_per_freq_temp=absorption[1]
        # elif f < 250:
        #     absorption_per_freq_temp=absorption[2]
        # elif f < 500:
        #     absorption_per_freq_temp=absorption[3]
        # elif f < 1000:
        #     absorption_per_freq_temp=absorption[4]
        # elif f < 2000:
        #     absorption_per_freq_temp=absorption[5]
        # elif f < 4000:
        #     absorption_per_freq_temp=absorption[6]
        # elif f < 8000:
        #     absorption_per_freq_temp=absorption[7]

    absorption=torch.tensor(absorption_per_freq,dtype=torch.float32, device=source.device) # absorption_per_freq is a 1024 * 6 matrix.

    if scatter is None:
        tr = torch.sqrt(1 - absorption)
    else:
        tr = torch.sqrt(1 - absorption) * torch.sqrt(1 - scatter)

    ind = torch.arange(-max_order, max_order + 1, device=source.device)
    if room.shape[0] == 2:
        XYZ = torch.meshgrid(ind, ind, indexing="ij")
    else:
        XYZ = torch.meshgrid(ind, ind, ind, indexing="ij") # Make a grid as big as the max_order
    XYZ = torch.stack([c.reshape((-1,)) for c in XYZ], dim=-1)
    XYZ = XYZ[XYZ.abs().sum(dim=-1) <= max_order] # Remove the points that are too far away

    # compute locations of image sources
    d = room[None, :]
    s = source[None, :]
    img_loc = torch.where(XYZ % 2 == 1, d * (XYZ + 1) - s, d * XYZ + s) # manages the reflections

    # attenuation
    exp_lo = abs(torch.floor((XYZ / 2)))
    exp_hi = abs(torch.floor((XYZ + 1) / 2))
    t_lo = tr[:, ::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)  # (num_band, left walls)
    t_hi = tr[:, 1::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)  # (num_band, right walls)

    att = torch.prod((t_lo**exp_lo) * (t_hi**exp_hi), dim=-1)  # (num_band, num_image_source)
    
    return img_loc, att

def simulate_rir_ism(
    room: torch.Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
    max_order: int,
    absorption: Union[float, torch.Tensor],
    output_length: Optional[int] = None,
    delay_filter_length: int = 81,
    center_frequency: Optional[torch.Tensor] = None,
    sound_speed: float = 343.0,
    sample_rate: float = 16000.0,
) -> Tensor:
    r"""Compute Room Impulse Response (RIR) based on the *image source method* :cite:`allen1979image`.
    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    .. devices:: CPU

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
        center_frequency (torch.Tensor, optional): The center frequencies of octave bands for multi-band walls.
            Only used when ``absorption`` is a 2D Tensor.
        sound_speed (float, optional): The speed of sound. (Default: ``343.0``)
        sample_rate (float, optional): The sample rate of the generated room impulse response signal.
            (Default: ``16000.0``)

    Returns:
        (torch.Tensor): The simulated room impulse response waveform. Tensor with dimensions
        `(channel, rir_length)`.

    Note:
        If ``absorption`` is a 2D Tensor and ``center_frequency`` is set to ``None``, the center frequencies
        of octave bands are fixed to ``[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]``.
        Users need to tune the values of ``absorption`` to the corresponding frequencies.
    """
    absorption = _validate_inputs(room, source, mic_array, absorption)
    img_location, att = _compute_image_sources(room, source, max_order, absorption)

    # compute distances between image sources and microphones
    # dist = _compute_distances(img_location, mic_array)
    vec = img_location[:, None, :] - mic_array[None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)  # (image_source, channel)

    #attenuate image sources
    img_src_att = att[..., None] / dist[None, ...]  # (band, image_source, channel)

    freq = torch.fft.rfftfreq(rfftfreq_samplebins, 1/sample_rate, device=source.device)
    i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

    # irfft_h_f=_frac_delay_but_frequential(i, img_src_att, freq, dist, sound_speed)
    exponential=torch.exp(-i*2*math.pi*freq[None,None, ...]*dist[None, ...]/sound_speed)
    img_src_att=torch.squeeze(img_src_att)
    exponential=torch.squeeze(exponential)
    product=torch.mul(img_src_att/math.sqrt(2*math.pi),torch.transpose(exponential, 0, 1))
    h_f=torch.sum(product,1)
    irfft_h_f=torch.fft.irfft(h_f)

    return irfft_h_f