import pandas as pd
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

'''
You should have run formay_csv.py and any other relevant preprocessing script before this one.
This script created these csvs
gwa_3Dfront_1m.csv
gwa_3Dfront_dp.csv
gwa_3Dfront_nondp.csv
'''

def string_to_array(s):
    '''
    Useful for what's in that dataset csv file
    '''
    # Remove square brackets and split the string
    elements = s.strip("[]").split()
    # Convert each element to float and create a numpy array
    return np.array([float(e) for e in elements])

# 1m
def create_1m_csv():
    df = pd.read_csv('./datasets/GWA_3DFRONT/gwa_3Dfront.csv')

    df["Source_Pos"] = df["Source_Pos"].apply(string_to_array)
    df["Receiver_Pos"] = df["Receiver_Pos"].apply(string_to_array)

    df["Distance"] = ((df["Receiver_Pos"]-df["Source_Pos"])**2).apply(np.sum)**0.5

    df = df[abs(df["Distance"] - 1) < 0.1]

    df.to_csv('./datasets/GWA_3DFRONT/gwa_3Dfront_1m_only.csv', index=False)

    print("done")

# Direct path only and no dp
def create_dp_csv():
    # dataset_path="./datasets/GWA_3DFRONT/gwa_3Dfront_validation.csv"
    # df = pd.read_csv(dataset_path)
    # df["Source_Pos"] = df["Source_Pos"].apply(string_to_array)
    # df["Receiver_Pos"] = df["Receiver_Pos"].apply(string_to_array)
    # df["Distance"] = ((df["Receiver_Pos"]-df["Source_Pos"])**2).apply(np.sum)**0.5
    # df["delay"] = (df["Distance"]/343) * 16000
    
    # dataset=GWA_3DFRONT_Dataset(csv_file=dataset_path, rir_std_normalization=False, gwa_scaling_compensation=True)
    # dataloader = DataLoader(dataset, shuffle=False,
    #                         num_workers=4, pin_memory=False,
    #                         collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
    # peak_onset = []
    # peak_amplitude = []
    # i = 0
    # with torch.no_grad():
    #     for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Getting all delays and first peak amplitudes"):
    #         peak_onset.append(label_origin_batch[0].item())
    #         peak_amplitude.append(label_rir_batch[0,int(label_origin_batch[0].item())].item())

    # df["peak_onset"] = peak_onset
    # df["peak_amplitude"] = peak_amplitude

    # df.to_csv("./datasets/GWA_3DFRONT/gwa_3Dfront_validation_with_peak_info.csv", index=False)

    df = pd.read_csv("./datasets/GWA_3DFRONT/gwa_3Dfront_validation_with_peak_info.csv")
    
    df['peak_amplitude'] = df['peak_amplitude'] / 0.0625029951333999

    df.to_csv("./datasets/GWA_3DFRONT/gwa_3Dfront_validation_with_peak_info_fixed.csv", index=False)

    df_dp = df[(abs(df["peak_onset"] - df["delay"]) < 20) & (abs(df["peak_amplitude"]-1) < 0.5)]
    df_dp.to_csv("./datasets/GWA_3DFRONT/gwa_3Dfront_validation_dp_only.csv", index=False)

    df_nondp = df[~((abs(df["peak_onset"] - df["delay"]) < 20) & (abs(df["peak_amplitude"]-1) < 0.5))]
    df_nondp.to_csv("./datasets/GWA_3DFRONT/gwa_3Dfront_validation_nondp_only.csv", index=False)

    print("done")

create_dp_csv()