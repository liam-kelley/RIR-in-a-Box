from json import load
from backpropagatable_ISM.compute_batch_rir_v2 import BP_filter
from torch import arange, Tensor
from torch.nn.functional import conv1d

def load_config_from_file(args):
    with open(args.config_file, 'r') as file:
        config = load(file)
    for key, value in config.items():
        setattr(args, key, value)
    return args


window_length = 81
N = arange(window_length) - window_length // 2
filter = BP_filter(N, 16000, 8000, 80, window_length).unsqueeze(0).unsqueeze(0)

def filter_rir_like_rirbox(rir: Tensor):
    '''
    used for training.
    rir should be of shape (batch_size, num_samples=3968)
    '''
    rir = rir.unsqueeze(1)
    rir = conv1d(rir, filter, padding=window_length // 2)
    return rir