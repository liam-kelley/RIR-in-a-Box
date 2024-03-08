import torch
import math

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

# Tests

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