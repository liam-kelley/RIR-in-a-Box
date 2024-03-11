import os
import glob
import soundfile as sf
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
from tqdm import tqdm
import scipy.signal as sig
from tools.gcc_phat import gcc_phat

from librosa import resample

##################################################
#################################################

# Get tapered sweep
# Parameters
# amp = 0.5 
# f1 = 100.
# f2 = 20000.
N = 17. 
fs = 48000.
# Generate sweep
T = (2**N) / fs # 2.73066 Duration of sweep.
# w1 = 2 * np.pi * f1
# w2 = 2 * np.pi * f2
# K = T * w1 / np.log(w2 / w1)
# L = T / np.log(w2 / w1)
# t = np.linspace(0, T - 1 / fs, int(fs * T))
# sweep = amp * np.sin(K * (np.exp(t / L) - 1))
# # Taper sweep
# taper_percent = 0.01
# taper_length = int(len(sweep) * taper_percent / 2)
# short_window = np.hanning(2 * taper_length)
# tapered_sweep = np.copy(sweep)
# tapered_sweep[:taper_length] *= short_window[:taper_length]
# tapered_sweep[-taper_length:] *= short_window[-taper_length:]

# # Get sweep_times_10
# sweep_times_10 = tapered_sweep
# fs_int=int(fs)
# print(fs_int)
# print(np.zeros(fs_int))
# one_second_of_silence=np.zeros(fs_int)
# for i in range(4):
#     sweep_times_10 = np.concatenate((one_second_of_silence,sweep_times_10))
# for i in range(9):
#     sweep_times_10 = np.concatenate((sweep_times_10,one_second_of_silence))
#     sweep_times_10 = np.concatenate((sweep_times_10,tapered_sweep))
# sweep_times_10 = np.concatenate((np.zeros(20),sweep_times_10,np.zeros(20)))

# # Get Inverse filter
# Inverse_filter_exp_scaling = np.exp(t / L)
# inverse_tapered_sweep = tapered_sweep[::-1]/Inverse_filter_exp_scaling

# # Deconvolve ESS sweeps
# ideal_ir = sig.fftconvolve(sweep_times_10, inverse_tapered_sweep, mode='same')

##################################################
#################################################

# Look at audio files
path_audio_conv = "datasets/ValidationDataset/deconvolved_audio_recs"
audio_files = glob.glob(os.path.join(path_audio_conv, "*2.wav"))
print(f"Found {len(audio_files)} audio files.")

for file_path in audio_files:
    # Load audio
    audio, samplerate = sf.read(file_path)
    assert samplerate == 48000, "Samplerate is not 48000."

    peaks, _ = find_peaks(audio, distance=T, height=np.absolute(audio).max()*0.35)

    window_length_div2=40
    slice_length=35000
    sound_speed=343

    # fig, axs = plt.subplots(2,5,figsize=(19,8))
    # fig.suptitle(f'10x RIRs from {os.path.basename(file_path)}', fontsize=16)

    # sliced_adios=[]
    # for i in range(min(len(peaks),10)):
    #     ix=i//(5)
    #     iy=i%(5)

        # axs[ix,iy].set_title(f"RIR n°{i+1}")

    distance=5 # FIX THIS BY LOADING THE CORRECT DISTANCE FROM A CSV. This will be done by first creating the csv of course.
    delay_due_to_distance=distance*fs/sound_speed
    i = 7
    slicer_left=peaks[i] - window_length_div2 - delay_due_to_distance
    slicer_right=peaks[i] - window_length_div2 - delay_due_to_distance + slice_length

    sliced_audio=audio[int(slicer_left)-2000:int(slicer_right)-8000]

    sliced_audio=resample(sliced_audio, orig_sr=48000, target_sr=16000)
    sliced_audio = np.abs(sliced_audio)
    sliced_audio = sliced_audio / np.max(sliced_audio)
    sliced_audio[sliced_audio < 0.001] = 0

    # Find the first non-zero element's index
    first_non_zero_index = np.argmax(sliced_audio >= 0.001)
    # Crop the signal from the first non-zero element
    sliced_audio = sliced_audio[max(first_non_zero_index-40,0):]

        # energy_decay_curve = np.cumsum(sliced_audio)

    #     sliced_adios.append(energy_decay_curve[:8000])

    #     axs[ix,iy].plot(sliced_audio/ np.max(sliced_audio))
    #     axs[ix,iy].plot(energy_decay_curve / np.max(energy_decay_curve))
    #     axs[ix,iy].set_xlabel("time in samples")
    #     axs[ix,iy].set_xlabel("RIR absolute amplitude")

    # plt.tight_layout()
    # plt.show()

    # adios_amigos=np.array(sliced_adios)
    # adios_amigos=adios_amigos.mean(axis=0)

    # plt.plot(adios_amigos/ np.max(adios_amigos))
    

    # recovered_array = np.insert(np.diff(adios_amigos), 0,sliced_audio[0])
    # plt.plot(recovered_array)
    # plt.plot(sliced_audio)
    # plt.show()

    # save audio
    sf.write(f"datasets/ValidationDataset/estimated_rirs/audio" + os.path.basename(file_path)[6:-13] + ".wav", sliced_audio, 16000)

