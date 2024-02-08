import os
import glob
import soundfile as sf
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
from get_sweep import get_validation_sweep
from tqdm import tqdm

path_audio_conv = "deconvolved_audio_recs"
audio_files = glob.glob(os.path.join(path_audio_conv, "*.wav"))
print(f"Found {len(audio_files)} audio files.")

_, _, T, fs = get_validation_sweep()

file_path=random.choice(audio_files)
audio, samplerate = sf.read(file_path)
peaks, _ = find_peaks(audio, distance=T, height=np.absolute(audio).max()*0.35)


window_length_div2=40
slice_length=35000
sound_speed=343

fig, axs = plt.subplots(2,5,figsize=(19,8))
fig.suptitle(f'10x RIRs from {os.path.basename(file_path)}', fontsize=16)

for i in range(min(len(peaks),10)):
    ix=i//(5)
    iy=i%(5)

    axs[ix,iy].set_title(f"RIR n°{i+1}")

    distance=2 # FIX THIS BY LOADING THE CORRECT DISTANCE FROM A CSV. This will be done by first creating the csv of course.
    delay_due_to_distance=distance*fs/sound_speed

    slicer_left=peaks[i] - window_length_div2 - delay_due_to_distance
    slicer_right=peaks[i] - window_length_div2 - delay_due_to_distance + slice_length

    sliced_audio=audio[int(slicer_left):int(slicer_right)]

    axs[ix,iy].plot(np.absolute(sliced_audio))
    axs[ix,iy].set_xlabel("time in samples")
    axs[ix,iy].set_xlabel("RIR absolute amplitude")

plt.tight_layout()
plt.show()