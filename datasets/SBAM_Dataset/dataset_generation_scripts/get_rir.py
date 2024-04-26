import pyroomacoustics as pra
import glob
import os
import soundfile as sf
import numpy as np

def get_rir(config, sbox_dim, materials, mic_pos, src_pos):

    room = pra.ShoeBox(
        sbox_dim,
        materials=materials,
        fs=config['rir_sample_rate'],
        max_order=config['rir_max_order'],
    )

    room.add_source(src_pos)
    room.add_microphone_array(np.expand_dims(mic_pos,1))
    room.compute_rir()
    rir=room.rir[0][0]

    return rir

def save_rir(config, rir):
    rir_names=glob.glob("datasets/SBAM_Dataset/rirs/rir_*.wav")
    rir_names.sort()
    if rir_names!=[]: # if there are already some rirs saved, write at the next index
        index=int(rir_names[-1].split("_")[-1].split(".")[0]) + 1
        rir_file_name = "datasets/SBAM_Dataset/rirs/rir_{:06}".format(index)+".wav"
    else:
        index=0
        rir_file_name = "datasets/SBAM_Dataset/rirs/rir_000000.wav"
    
    if not os.path.exists(os.path.dirname(rir_file_name)):
        os.makedirs(os.path.dirname(rir_file_name))

    sf.write(rir_file_name, rir, config['rir_sample_rate'])
    print("RIR saved as " + rir_file_name)

    return rir_file_name