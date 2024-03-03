import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################
def create_csv():
    # data
    dataset=GWA_3DFRONT_Dataset(csv_file="./datasets/GWA_3DFRONT/gwa_3Dfront_tiny.csv", rir_std_normalization=False)
    dataloader = DataLoader(dataset, shuffle=False,
                            num_workers=4, pin_memory=False,
                            collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)

    DEVICE = 'cpu'

    my_list = []
    i = 0
    with torch.no_grad():
        for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Inference"):
            
            label_rir = label_rir_batch[0]
            distance = torch.sqrt(torch.sum((mic_pos_batch[0]-src_pos_batch[0])**2))
            delay = (distance/343) * 16000
            measured_delay = label_origin_batch[0].int()

            delay_discrepancy = torch.abs(delay-measured_delay).item()

            # if delay_discrepancy < 20 :
            my_list.append([distance.item(), delay_discrepancy, label_rir[measured_delay].item()])
            i+=1

            # if i>100:
            #     break

    df = pd.DataFrame(my_list, columns=['distance', 'delay_error', 'peak_amplitude'])

    df.to_csv('wesh.csv', index=False)

    return(True)
    
def load_csv():
    # load dataframe from csv
    df = pd.read_csv('./wesh.csv')

    # Order by distance
    df = df.sort_values(by=['distance'])

    df = df[df['delay_error'] < 5]

    # # # Group by distance, get mean and std peak amplitude
    df_group = df.groupby('distance').agg({'delay_error': ['mean', 'std'], 'peak_amplitude': ['mean', 'std']}).reset_index()
    df_group.columns = ['distance', 'delay_error_mean', 'delay_error_std', 'peak_amplitude_mean', 'peak_amplitude_std']
    
    # filter df where peak amplitude is futher than std from its mean group-wise
    df = df.merge(df_group, on='distance')
    df = df[df['delay_error'] < 3]
    df = df[abs(df["peak_amplitude"] - df["peak_amplitude_mean"]) < df["peak_amplitude_std"]]

    # drop columns
    df = df.drop(columns=['delay_error_mean', 'delay_error_std', 'peak_amplitude_mean', 'peak_amplitude_std'])

    # # # Group by distance, get mean and std peak amplitude
    df = df.groupby('distance').agg({'delay_error': ['mean', 'std'], 'peak_amplitude': ['mean', 'std']}).reset_index()
    df.columns = ['distance', 'delay_error_mean', 'delay_error_std', 'peak_amplitude_mean', 'peak_amplitude_std']
    
    # filter df where peak amplitude is futher than std from its mean group-wise
    # df = df[df['delay_error_std'] < 0.1]
    df = df[df['peak_amplitude_std'] < 0.1]

    df["peak_normed"] = df['peak_amplitude_mean'] * (df['distance']) / 0.0625029951333999

    plt.plot(df['distance'], df['peak_normed'])    

    # distance 1 = 0.0625029951333999
    # distance 1.41 = 0.027340146047728355
    # distance 2 = 0.02348
    # distance 2.44 = 0.013
    # distance = np.array([1, 1.41, 2, 2.44])
    # peak = np.array([0.0625029951333999, 0.027340146047728355, 0.02348, 0.013])
    # peak = peak * (distance**2) / 0.0625029951333999
    # plt.plot(distance , peak)
    plt.title('This should be a straight line')
    plt.xlabel('Distance (m)')
    plt.ylabel('Peak Amplitude')
    plt.grid()
    plt.show()

    df.to_csv('wesh_filtered.csv', index=False)

    # print the box plot of peak amplitude

    plt.boxplot([df['peak_amplitude']],patch_artist=True, showmeans=True, showfliers=True)
    plt.show()

    return(True)

def main():
    create_csv()
    load_csv()

if __name__ == "__main__":
    main()
