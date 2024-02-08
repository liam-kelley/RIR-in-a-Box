from get_sweep import get_validation_sweep
import numpy as np
import os
import glob
from scipy import signal as sig
import soundfile as sf
from tqdm import tqdm

path_rec = "./audio_recordings/"
path_audio_conv = "deconvolved_audio_recs"

# Ensure the output directory exists
if not os.path.exists(path_audio_conv):
    os.makedirs(path_audio_conv)

# get inverse sweep
_, inverse_tapered_sweep, _ , _ = get_validation_sweep()

# List all audio files in the directory
audio_files = glob.glob(os.path.join(path_rec, "*.wav"))
print(f"Working on {len(audio_files)} audio files.")

mylist=[]
for file_path in tqdm(audio_files):
    # Load audio
    audio, samplerate = sf.read(file_path)
    
    for i, channel in enumerate(np.transpose(audio)):
        # Apply FFT convolution
        ir = sig.fftconvolve(channel, inverse_tapered_sweep, mode='same')

        # normalize according to max of all validation set (224.23300737839364)
        ir=ir/225
        
        # Construct the output file path
        base_name = "rir10x_" + str(os.path.basename(file_path))[22:-4] + f"_channel{i}.wav"
        output_path = os.path.join(path_audio_conv, base_name)
        
        # Save the deconvolved audio
        sf.write(output_path, ir, samplerate)

print("Deconvolution completed for all audio files.")