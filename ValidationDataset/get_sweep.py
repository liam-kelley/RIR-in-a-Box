import numpy as np

def get_validation_sweep():
    '''
    This gets the tapered sweep that was used for the validation dataset recording
    
    Inputs: /
    Returns:    tapered_sweep
                inverse_tapered_sweep
                duration of sweep in samples at 48kHz
                sample_rate
    '''

    # Parameters
    amp = 0.5 
    f1 = 100.
    f2 = 20000.
    N = 17. 
    fs = 48000.

    # Generate sweep
    T = (2**N) / fs # 2.73066 Duration of sweep.
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    K = T * w1 / np.log(w2 / w1)
    L = T / np.log(w2 / w1)
    t = np.linspace(0, T - 1 / fs, int(fs * T))
    sweep = amp * np.sin(K * (np.exp(t / L) - 1))

    # Define the percentage of the signal to taper (e.g., 1% of the signal length)
    taper_percent = 0.01
    # Calculate the length of the taper window
    taper_length = int(len(sweep) * taper_percent / 2)
    # Create a short Hanning window for tapering
    short_window = np.hanning(2 * taper_length)
    # Apply the taper to the beginning and end of the signal
    tapered_sweep = np.copy(sweep)
    tapered_sweep[:taper_length] *= short_window[:taper_length]
    tapered_sweep[-taper_length:] *= short_window[-taper_length:]

    # Inverse filter
    Inverse_filter_exp_scaling = np.exp(t / L)
    inverse_tapered_sweep = tapered_sweep[::-1]/Inverse_filter_exp_scaling

    return tapered_sweep, inverse_tapered_sweep, 2**N, fs