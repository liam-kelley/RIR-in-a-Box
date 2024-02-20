from torch.utils.data import Dataset
# import soundfile as sf
import librosa
import numpy as np
import pandas as pd

from utility import mesh2ir_rir_preprocessing


class my_dataset(Dataset):
    def __init__(self, csv_file=""):
        self.csv_file=csv_file
        self.sample_rate=None
        self.data = pd.read_csv(csv_file)
        print('GWA Dataset ', csv_file ,' loaded')

    def preprocess_dataframe(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    # def get_RIR(self, full_RIR_path):
    #     # Load
    #     rir,fs = librosa.load(full_RIR_path)
    #     # Resample
    #     rir = librosa.resample(rir,orig_sr=fs,target_sr=16000)
    #     # MESH2IR preprocessing
    #     rir = mesh2ir_rir_preprocessing(rir)

    #     rir = np.array([rir]).astype('float32')
    #     return rir

    def __getitem__(self, index):
        pass

    def custom_collate_fn(batch):
        pass