import torch
import math
import matplotlib.pyplot as plt

def LP_filter(n, fs, window_length, lp_cutoff_frequency):
    '''
    windowed sinc filter
    '''
    # Get hann
    hanning = 0.5 * (1 - torch.cos(2 * math.pi * (n + 1 + window_length//2) / window_length))

    # Get sinc
    nyquist_f = fs / 2
    f_c_normalized = lp_cutoff_frequency / nyquist_f
    sinc = f_c_normalized * torch.special.sinc( n * f_c_normalized)
    windowed_sinc = sinc * hanning

    return windowed_sinc

def BP_filter(n, fs, fc_high, fc_low, window_length):
    '''
    windowed sinc filter translated to be bandpass
    '''
    # Get hann
    hann_window = 0.5 * (1 - torch.cos(2 * math.pi * (n + 1 + window_length//2) / window_length))

    # Get sinc
    half_bandwidth = (fc_high - fc_low)/2
    nyquist_f = fs / 2
    f_c_normalized = half_bandwidth / nyquist_f
    sinc = 2 * f_c_normalized * torch.sinc(f_c_normalized * n) # we have to multiply by 2 because when translating, we get the energy of all negative frequencies!
    
    # Get windowed sinc
    h_lp = sinc * hann_window

    # Frequency shift to convert low pass to band pass
    fc_center = (fc_high + fc_low) / 2
    h_bp = h_lp * torch.cos(2 * math.pi * fc_center * n / fs)

    return h_bp

def create_filter_bank(fs, window_length, f_low_hi, device='cuda'):
    '''
    f_low_hi : list of tuples [(f_low, f_high), ...]
    '''
    n = torch.arange(window_length) - window_length // 2
    filter_bank = []
    for f_low, f_high in f_low_hi:
        filter_bank.append(BP_filter(n, fs, f_high, f_low, window_length))
    filter_bank = torch.stack(filter_bank).unsqueeze(1) # Adding in_channels dimension
    filter_bank = filter_bank.to(device)
    return filter_bank

def apply_filter_bank(rir, filter_bank, window_length):
    '''
    rir : torch.Tensor of shape (batch_size, num_samples)
    filter_bank : list of torch.Tensor of shape (1, window_length)
    '''
    rir = rir.unsqueeze(1)  # (batch_size, n_bands, num_samples)
    rir = torch.nn.functional.conv1d(rir, filter_bank, padding=window_length // 2)
    return rir

# Tests 3

rirs=torch.rand(8,3998, device='cuda')
window_length = 81
fs=16000
# centers : 125, 250, 500, 1000, 2000, 4000
f_low_hi=[  (88,177),
            (177,354),
            (354,707),
            (707,1414),
            (1414,2828),
            (2828,5657)]
filter_bank = create_filter_bank(fs, window_length, f_low_hi)
fig1, ax = plt.subplots(1, 1, figsize=(10, 8))
for i in range(len(filter_bank)):
    ax.plot(filter_bank[i].squeeze().cpu().numpy(), label=f"Bandpass at fc low {f_low_hi[i][0]} Hz\nand fc high {f_low_hi[i][1]} Hz")
ax.legend()
filtered_rirs = apply_filter_bank(rirs, filter_bank, window_length)
fig2, axs = plt.subplots(7, 1, figsize=(9, 9.5))
axs[0].plot(rirs[0].squeeze().cpu().numpy())
for i in range(6):
    axs[i+1].plot(filtered_rirs[0,i].squeeze().cpu().numpy())
plt.show()

# Tests 1

# window_length = 81
# n = torch.arange(window_length) - window_length // 2
# n = n[30:70]
# lp_cutoff_frequency = 8000
# fs=16000
# plt.plot(LP_filter(n, fs, window_length, lp_cutoff_frequency), label="Lowpass at {} Hz".format(lp_cutoff_frequency))
# plt.show()
# plt.title("Filter shapes".format(lp_cutoff_frequency))
# # plt.show()
# plt.plot(BP_filter(n, fs, lp_cutoff_frequency, 80, window_length), label=f"Bandpass at fc low {80} Hz\nand fc high {lp_cutoff_frequency} Hz")
# plt.legend()
# plt.show()

# Tests 2

# Old_Hann = lambda x : 0.5 * (1 + torch.cos(math.pi * x / 40))
# Old_sinc = lambda x : torch.sinc(x)
# Old_LP_filter = lambda x, fc : fc * Old_sinc(x * fc) * Old_Hann(x)

# n = torch.arange(81) - 81 // 2
# plt.plot(Old_LP_filter(n, 1))
# plt.title("Old LP filter : filtering to 8kHz")
# plt.show()