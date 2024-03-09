import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.utility import load_all_models_for_inference, inference_on_all_models
from tools.pyLiam.LKTimer import LKTimer
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from scipy.signal import stft

rirbox_config = "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_HIQMRSTFT_EDR_superfast_noDistInLatent.json"
# rirbox_config = "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model3_dp_HIQMRSTFT_EDR_superfast.json"
DATASET_PATH = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"

TOA_SYNCHRONIZATION = False
SCALE_MESH2IR_BY_ITS_ESTIMATED_STD = True # If True, cancels out the std normalization used during mesh2ir's training
SCALE_MESH2IR_GWA_SCALING_COMPENSATION = True # If true, cancels out the scaling compensation mesh2ir learned from the GWA dataset during training.
MESH2IR_USES_LABEL_ORIGIN = False
RESPATIALIZE_RIRBOX = False # This both activates the respaitialization of the rirbox and the start from ir onset
ISM_MAX_ORDER = 18
plot_stft = False

mesh2ir, rirbox, config, DEVICE = load_all_models_for_inference(rirbox_config, START_FROM_IR_ONSET=RESPATIALIZE_RIRBOX, ISM_MAX_ORDER=ISM_MAX_ORDER)

# data
dataset=GWA_3DFRONT_Dataset(csv_file=DATASET_PATH,
                            rir_std_normalization=False, gwa_scaling_compensation=True)
dataloader = DataLoader(dataset, shuffle=True,
                        num_workers=4, pin_memory=False,
                        collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)

def format_text(batch_of_coords):
    return [float(str(coord)[:5]) for coord in batch_of_coords.squeeze().cpu().numpy().tolist()]

with torch.no_grad():
    for x_batch, edge_index_batch, batch_indexes, label_rir, label_origin, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Inference"):
        
        rir_mesh2ir, rir_rirbox,\
                origin_mesh2ir, origin_rirbox, \
                virtual_shoebox= inference_on_all_models(x_batch, edge_index_batch, batch_indexes,
                                                        mic_pos_batch, src_pos_batch, label_origin,
                                                        mesh2ir, rirbox, DEVICE,
                                                        SCALE_MESH2IR_BY_ITS_ESTIMATED_STD,
                                                        SCALE_MESH2IR_GWA_SCALING_COMPENSATION,
                                                        MESH2IR_USES_LABEL_ORIGIN,
                                                        RESPATIALIZE_RIRBOX)

        ############################ Plotting #############################

        rirbox_distance=torch.linalg.norm(virtual_shoebox[1][0]-virtual_shoebox[2][0])
        gt_distance=torch.linalg.norm(mic_pos_batch[0]-src_pos_batch[0])

        rir_mesh2ir = rir_mesh2ir.squeeze()[:3968]
        rir_rirbox = rir_rirbox.squeeze()
        label_rir = label_rir.squeeze()

        # Get EDCs
        nrg_mesh2ir = torch.pow(rir_mesh2ir, 2)
        nrg_rirbox = torch.pow(rir_rirbox, 2)
        nrg_label = torch.pow(label_rir, 2)
        edc_mesh2ir = torch.flip(torch.cumsum(torch.flip(nrg_mesh2ir, dims=[-1]), dim=-1), dims=[-1])
        edc_rirbox = torch.flip(torch.cumsum(torch.flip(nrg_rirbox, dims=[-1]), dim=-1), dims=[-1])
        edc_label = torch.flip(torch.cumsum(torch.flip(nrg_label, dims=[-1]), dim=-1), dims=[-1])
        edc_mesh2ir = edc_mesh2ir / edc_mesh2ir[0]
        edc_rirbox = edc_rirbox / edc_rirbox[0]
        edc_label = edc_label / edc_label[0] 

        # preprocessing for plotting
        rir_mesh2ir = abs(rir_mesh2ir.cpu().numpy())
        rir_rirbox = abs(rir_rirbox.cpu().numpy())
        label_rir = abs(label_rir.cpu().numpy())

        origin_mesh2ir = origin_mesh2ir.cpu().numpy()
        origin_rirbox = origin_rirbox.cpu().numpy()
        label_origin = label_origin.cpu().numpy()

        y_max = max(1.0, max(rir_mesh2ir), max(rir_rirbox), max(label_rir))
        x_mesh2ir = np.arange(0, len(edc_mesh2ir), 1)
        x_rirbox = np.arange(0, len(edc_rirbox), 1)
        x_label = np.arange(0, len(edc_label), 1)

        edc_mesh2ir = (y_max*edc_mesh2ir).cpu().numpy()
        edc_rirbox = (y_max*edc_rirbox).cpu().numpy()
        edc_label = (y_max*edc_label).cpu().numpy()

        ############## PLOTTING ##############

        fig, axs = plt.subplots(3, 1, figsize=(9, 9.5))
        fig.suptitle('RIR comparison between MESH2IR and RIRBOX')

        # MESH2IR
        axs[0].set_title('MESH2IR')
        if TOA_SYNCHRONIZATION:
            raise NotImplementedError("TOA Synchronization deprecated")
            axs[0].plot(rir_mesh2ir[max(0,int(label_origin[0]-41)):], label="MESH2IR", color='blue')
        else: 
            axs[0].plot(rir_mesh2ir, label="MESH2IR", color='blue')
            axs[0].axvline(x=origin_mesh2ir, ymax=max(rir_mesh2ir)/y_max,color='red', linestyle='--', label=f'IR Onset : {origin_mesh2ir[0]:.2f}')
            axs[0].plot(edc_mesh2ir, label="MESH2IR EDC", color='darkblue', linestyle='dotted')
            axs[0].fill_between(x_mesh2ir, edc_mesh2ir, color='darkblue', alpha=0.1)
        axs[0].set_xlim(0, 4096)
        axs[0].set_ylim(0.0, y_max)
        axs[0].grid(ls="--", alpha=0.5)
        axs[0].legend()

        # RIRBOX
        axs[1].set_title('RIRBOX : ' + " ".join(rirbox_config.split('/')[-1].split('.')[0].split('_')[1:]))
        if TOA_SYNCHRONIZATION:
            raise NotImplementedError("TOA Synchronization deprecated")
            axs[1].plot(rir_rirbox[max(0,int(origin_rirbox[0]-41)):], label="RIRBOX", color='orange')
        else:
            axs[1].plot(rir_rirbox, label="RIRBOX", color='orange')
            axs[1].axvline(x=origin_rirbox, ymax=max(rir_rirbox)/y_max,color='red', linestyle='--', label=f'IR Onset : {origin_rirbox[0]:.2f}')
            axs[1].plot(edc_rirbox, label="RIRBOX EDC", color='darkorange', linestyle='dotted')
            axs[1].fill_between(x_rirbox, edc_rirbox, color='darkorange', alpha=0.1)
        axs[1].set_xlim(0, 4096)
        axs[1].set_ylim(0.0, y_max)
        axs[1].grid(ls="--", alpha=0.5)
        axs[1].legend()
        axs[1].text(0.5, 0.85, f"Room dim {format_text(virtual_shoebox[0])}", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].text(0.5, 0.78, f"Absorption {format_text(virtual_shoebox[3])[3:]}", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].text(0.5, 0.71, f"Mic pos {format_text(virtual_shoebox[1])}", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].text(0.5, 0.64, f"Src pos {format_text(virtual_shoebox[2])}", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].text(0.5, 0.57, f"Mic-Src Distance : {float(str(rirbox_distance.cpu().item())[:5])}", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].text(0.5, 0.50, f"Respatialize RIRBOX : {RESPATIALIZE_RIRBOX}", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)


        axs[2].set_title('GT')
        if TOA_SYNCHRONIZATION:
            raise NotImplementedError("TOA Synchronization deprecated")
            axs[2].plot(label_rir[max(0,int(label_origin[0]-41)):], label="GT", color='green')
        else:
            axs[2].plot(label_rir, label="GT", color='green')
            axs[2].axvline(x=label_origin, ymax = max(label_rir)/y_max, color='red', linestyle='--', label=f'IR Onset : {label_origin[0]:.2f}')
            axs[2].plot(edc_label, label="GT EDC", color='darkgreen', linestyle='dotted')
            axs[2].fill_between(x_label, edc_label, color='darkgreen', alpha=0.1)
        axs[2].set_xlim(0, 4096)
        axs[2].set_ylim(0.0, y_max)
        axs[2].grid(ls="--", alpha=0.5)
        axs[2].legend()
        axs[2].text(0.5, 0.75, f"GT Mic pos {format_text(mic_pos_batch)}\nGT Src pos {format_text(src_pos_batch)}\nGT Mic-Src Distance : {float(str(gt_distance.cpu().item())[:5])}", # \nIR onset : {label_origin[0]}
                    horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)

        plt.tight_layout()
        plt.show()

        if plot_stft:
            # # plot rirs with subplots
            fig, axs = plt.subplots(3, 1, figsize=(9, 9))

            f, t, Zxx = stft(abs(np.array(rir_mesh2ir, dtype=float)), 16000, nperseg=256)
            axs[0].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            axs[0].set_title("MESH2IR STFT Magnitude")
            axs[0].set_ylabel('Frequency [Hz]')
            axs[0].set_xlabel('Time [sec]')

            f, t, Zxx = stft(abs(np.array(rir_rirbox, dtype=float)), 16000, nperseg=256)
            axs[1].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            axs[1].set_title("RIRBOX STFT Magnitude")
            axs[1].set_ylabel('Frequency [Hz]')
            axs[1].set_xlabel('Time [sec]')

            f, t, Zxx = stft(abs(np.array(label_rir, dtype=float)), 16000, nperseg=256)
            axs[2].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            axs[2].set_title("GT STFT Magnitude")
            axs[2].set_ylabel('Frequency [Hz]')
            axs[2].set_xlabel('Time [sec]')
            plt.show()