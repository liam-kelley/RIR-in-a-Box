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

fs = 48000.
N = 17. 
T = (2**N) / fs # 2.73066 Duration of sweep.

# Look at audio files
path_audio_conv = "datasets/ValidationDataset/deconvolved_audio_recs"
audio_files = glob.glob(os.path.join(path_audio_conv, "*2.wav"))
print(f"Found {len(audio_files)} audio files.")

if not os.path.exists("datasets/ValidationDataset/estimated_rirs"):
    os.makedirs("datasets/ValidationDataset/estimated_rirs")

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

    distance=5 # FIX THIS BY LOADING THE CORRECT DISTANCE FROM A CSV.
    delay_due_to_distance=distance*fs/sound_speed
    i = 7
    slicer_left=peaks[i] - window_length_div2 - delay_due_to_distance
    slicer_right=peaks[i] - window_length_div2 - delay_due_to_distance + slice_length

    sliced_audio=audio[int(slicer_left)-2000:int(slicer_right)-4000]

    sliced_audio=resample(sliced_audio, orig_sr=48000, target_sr=16000)
    sliced_audio = np.abs(sliced_audio)
    sliced_audio = sliced_audio / np.max(sliced_audio)
    # sliced_audio[sliced_audio < 0.001] = 0

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
    wav_name = "audio_" + os.path.basename(file_path)[7:-13]
    if wav_name[-1] == "C":
        wav_name = wav_name[:-10] + "open_srcCopendoor"

    sf.write(f"datasets/ValidationDataset/estimated_rirs/" + wav_name  + ".wav", sliced_audio, 16000)

