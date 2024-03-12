import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.utility import load_all_models_for_inference, inference_on_all_models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from tools.gcc_phat import gcc_phat
import pandas as pd
import os
from matplotlib.lines import Line2D
import glob
from librosa import load
from scipy.signal import fftconvolve
import torch.nn.functional as F

def sss_mesh2ir_vs_rirbox(model_config : str, validation_csv : str, validation_iterations=0,
                                                 SCALE_MESH2IR_BY_ITS_ESTIMATED_STD = True, # If True, cancels out the std normalization used during mesh2ir's training
                                                 SCALE_MESH2IR_GWA_SCALING_COMPENSATION = True, # If true, cancels out the scaling compensation mesh2ir learned from the GWA dataset during training.
                                                 MESH2IR_USES_LABEL_ORIGIN = False, # Doesn't do anything
                                                 RESPATIALIZE_RIRBOX = False, # Of course... This helps spatialization a lot... But... it might be very unfair to use this without also using it in mesh2ir.
                                                #  FILTER_MESH2IR_IN_HYBRID = False,
                                                 ISM_MAX_ORDER = 15,
                                                 SHOW_TAU_PLOTS = False,
                                                 SHOW_SSL_PLOTS = False,
                                                 CONVOLVE_SIGNALS = False, # SLOW and doesn't significantly change performance?
                                                 ):
    '''Virtual Sound Source Localization task for the MESH2IR and RIRBOX models.'''

    print("Starting sound source spatialization validation for model: ", model_config.split("/")[-1].split(".")[0],end="\n\n")

    mesh2ir, rirbox, config, DEVICE = load_all_models_for_inference(model_config, START_FROM_IR_ONSET=RESPATIALIZE_RIRBOX, ISM_MAX_ORDER=ISM_MAX_ORDER)

    # data
    dataset=GWA_3DFRONT_Dataset(csv_file=validation_csv,rir_std_normalization=False, gwa_scaling_compensation=True, dont_load_rirs=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=10, pin_memory=False,
                            collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
    print("")

    # Get wav file paths
    wavs = glob.glob("datasets/Small_Timit/*.wav")

    # array information -> simulate the 2 ears (22.5 cm of distance according to wikipedia)
    d = 0.225
    src_distance = 1.5 # in meters
    
    azimuths = np.arange(0,360,21) # in degrees, just azimuth for now
    # azimuths = np.arange(50,180,7) # in degrees, just azimuth for now

    fs = 16000
    c = 343

    my_list=[]
    with torch.no_grad():
        # iterate over the dataset
        iterations=0
        for x_batch, edge_index_batch, batch_indexes, _, _, mic_pos_batch, _ in tqdm(dataloader, desc="Sound source spatialization validation"): 
            mic_pos_1 = mic_pos_batch.squeeze() - torch.tensor([0.0-d/2, 0, 0.0])
            mic_pos_2 = mic_pos_batch.squeeze() - torch.tensor([0.0+d/2, 0, 0.0])
            mic_pos_1 = mic_pos_1.unsqueeze(0)
            mic_pos_2 = mic_pos_2.unsqueeze(0)
            arr_pos = mic_pos_batch.squeeze().cpu().numpy()
            src_loc_from_spherical = lambda dist, azimuth : (dist * np.r_[np.cos(azimuth), np.sin(azimuth), 0]) + arr_pos

            # import speech signal
            if CONVOLVE_SIGNALS:
                signal, fs = load(path=wavs[iterations%len(wavs)], sr=fs, mono=True, duration=3.7)

            # for every angle
            tdoas_mesh2ir=[]
            tdoas_rirbox=[]
            tdoas_origins_mesh2ir=[]
            tdoas_origins_rirbox=[]
            tdoas_theoretical=[]
            for az in tqdm(azimuths,leave=False,  desc="For every azimuth"):
                # place the src with repsect to the array
                src_pos = torch.tensor(src_loc_from_spherical(src_distance, np.deg2rad(az))).unsqueeze(0)
                
                rirs_mesh2ir=[]
                rirs_rirbox=[]
                origins_mesh2ir=[]
                origins_rirbox=[]
                distances=[]
                # for both mic pos. (TODO implement multichannel backpropagatable ISM)
                for mic_pos in [mic_pos_1, mic_pos_2]:
                    # Get RIRS for our models
                    rir_mesh2ir, rir_rirbox, origin_mesh2ir, origin_rirbox,  _= inference_on_all_models(x_batch, edge_index_batch, batch_indexes,
                                                                                mic_pos.float(), src_pos.float(), 0,
                                                                                mesh2ir, rirbox, DEVICE,
                                                                                SCALE_MESH2IR_BY_ITS_ESTIMATED_STD,
                                                                                SCALE_MESH2IR_GWA_SCALING_COMPENSATION,
                                                                                MESH2IR_USES_LABEL_ORIGIN,
                                                                                RESPATIALIZE_RIRBOX)
                    
                    # print(rir_mesh2ir.shape, rir_rirbox.shape)
                    rirs_mesh2ir.append(rir_mesh2ir[0,:3968])
                    rirs_rirbox.append(rir_rirbox[0,:3968])
                    origins_mesh2ir.append(origin_mesh2ir)
                    origins_rirbox.append(origin_rirbox)
                    distances.append(np.linalg.norm(src_pos.squeeze() - mic_pos.squeeze()) / c)
                
                if CONVOLVE_SIGNALS:
                    rirs_mesh2ir.extend(rirs_rirbox)
                    del rirs_rirbox
                    signals_tensor = torch.tensor(signal, device=DEVICE).unsqueeze(0).unsqueeze(1)
                    impulse_responses_tensor = torch.nn.utils.rnn.pad_sequence(rirs_mesh2ir, batch_first=True).unsqueeze(1)
                    # Assuming impulse responses are all the same length and signals are too, or have been padded accordingly
                    results = F.conv1d(signals_tensor, impulse_responses_tensor, padding=0).squeeze().cpu().numpy()
                    signal0_mesh2ir = results[0]
                    signal1_mesh2ir = results[1]
                    signal1_rirbox = results[2]
                    signal0_rirbox = results[3]
                else:
                    signal0_mesh2ir = rirs_mesh2ir[0].cpu().numpy()
                    signal1_mesh2ir = rirs_mesh2ir[1].cpu().numpy()
                    signal0_rirbox = rirs_rirbox[0].cpu().numpy()
                    signal1_rirbox = rirs_rirbox[1].cpu().numpy()

                tau_mesh2ir, _ = gcc_phat(signal1_mesh2ir, signal0_mesh2ir, fs=fs, max_tau=None, interp=32)
                tau_rirbox, _ = gcc_phat(signal1_rirbox, signal0_rirbox, fs=fs, max_tau=None, interp=32)

                tau_origin_mesh2ir = (origins_mesh2ir[1] - origins_mesh2ir[0])/fs
                tau_origin_rirbox = (origins_rirbox[1] - origins_rirbox[0])/fs

                if SHOW_TAU_PLOTS:
                    fig, axs = plt.subplots(2, figsize=(9,9))
                    axs[0].set_title(f"MESH2IR : difference in RIR for Angle {az} degrees")
                    axs[1].set_title(f"RIRBOX : difference in RIR for Angle {az} degrees")
                    for j, rirs in enumerate([rirs_mesh2ir, rirs_rirbox]):#, rirs_hybrid]):
                        axs[j].plot(rirs[1].cpu().numpy(), alpha=0.5, label="mic 1")
                        axs[j].plot(rirs[0].cpu().numpy(), alpha=0.5, label="mic 2")
                    for j, tau in enumerate([tau_mesh2ir, tau_rirbox]):#, tau_hybrid]):
                        axs[j].axvline(abs(tau*1e6), ls="dashed", color="black", label="tau * 1e6")
                    for ax in axs:
                        ax.legend()
                        ax.grid(ls="dashed", alpha=0.5)
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Amplitude")
                    plt.tight_layout()
                    plt.show()
                
                tdoas_mesh2ir.append(tau_mesh2ir)
                tdoas_rirbox.append(tau_rirbox)
                tdoas_origins_mesh2ir.append(tau_origin_mesh2ir.cpu().numpy())
                tdoas_origins_rirbox.append(tau_origin_rirbox.cpu().numpy())
                tdoas_theoretical.append(distances[1] - distances[0])
        
            if SHOW_SSL_PLOTS:
                plt.plot(azimuths, -np.array(tdoas_mesh2ir), label='mesh2ir')
                plt.plot(azimuths, -np.array(tdoas_rirbox), label='rirbox')
                plt.plot(azimuths, -np.array(tdoas_origins_mesh2ir), label='mesh2ir_origins')
                plt.plot(azimuths, -np.array(tdoas_origins_rirbox), label='rirbox_origins')
                plt.plot(azimuths, -np.array(tdoas_theoretical), label='Theoretical')
                plt.xlabel('Azimuth (degrees)')
                plt.ylabel('TDOA (s)')
                plt.legend()
                plt.title('TDOA vs Azimuth')
                plt.tight_layout()
                plt.show()

            mse_mesh2ir = np.mean(np.sqrt((np.array(tdoas_mesh2ir) - np.array(tdoas_theoretical))**2))
            mse_rirbox = np.mean(np.sqrt((np.array(tdoas_rirbox) - np.array(tdoas_theoretical))**2))

            mse_origins_mesh2ir = np.mean(np.sqrt((np.array(tdoas_origins_mesh2ir) - np.array(tdoas_theoretical))**2))
            mse_origins_rirbox = np.mean(np.sqrt((np.array(tdoas_origins_rirbox) - np.array(tdoas_theoretical))**2))

            my_list.append([mse_mesh2ir, mse_rirbox,mse_origins_mesh2ir, mse_origins_rirbox])
            
            iterations +=1
            if iterations == validation_iterations:
                break
    
    my_list = np.array(my_list)
    df = pd.DataFrame(my_list, columns=["mse_mesh2ir", "mse_rirbox", "mse_origins_mesh2ir", "mse_origins_rirbox"])

    save_path = "./validation/results_sss/" + config['SAVE_PATH'].split("/")[-2] + "/" + config['SAVE_PATH'].split("/")[-1].split(".")[0] + ".csv"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path)

    print("Validation results saved at: ", save_path)

def view_results_sss_mesh2ir_vs_rirbox(results_csv="./validation/results_sss/***.csv"):
    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(1,1, figsize=(7, 5))
    fig.suptitle(f'Sound Source Spatialization validation\nMESH2IR vs {results_csv.split("/")[-1].split(".")[0]}')

    # Prepare the data for the box plot
    model_names = ["Baseline", "RIRBOX"]

    mean_marker = Line2D([], [], color='w', marker='^', markerfacecolor='green', markersize=10, label='Mean')

    # EDR
    ax.boxplot([df["mse_mesh2ir"], df["mse_rirbox"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    ax.set_ylabel('Mean TODA Error')
    ax.legend(handles=[mean_marker], loc='lower left')
    ax.set_ylim(0,0.0005)

    ax.grid(ls="--", alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()