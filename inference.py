import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET, STAGE1_G, MESH2IR_FULL
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR, RIRBox_FULL, RIRBox_MESH2IR_Hybrid
from models.utility import load_mesh_net, load_GAN, load_mesh_to_shoebox
from training.utility import filter_rir_like_rirbox
from tools.pyLiam.LKTimer import LKTimer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.signal import stft

rirbox_path = "models/RIRBOX/ablation3_different_datasets/rirbox_model2_MRSTFT_MSDist_MLPDEPTH4_dp.pth" # Ideally would just use 1 config file for everything
dataset_path = "datasets/GWA_3DFRONT/gwa_3Dfront_validation_nondp_only.csv"
plot_stft = False
RIRBOX_MAX_ORDER = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOA_SYNCHRONIZATION = True
SCALE_MESH2IR_BY_ITS_ESTIMATED_STD = True # If True, cancels out the std normalization used during mesh2ir's training
SCALE_MESH2IR_GWA_SCALING_COMPENSATION = True # If true, cancels out the scaling compensation mesh2ir learned from the GWA dataset during training.
MESH2IR_USES_LABEL_ORIGIN = False
RESPATIALIZE_RIRBOX = True
FILTER_MESH2IR_IN_HYBRID = False

############################################ Data #############################################

# data
dataset=GWA_3DFRONT_Dataset(csv_file=dataset_path,
                            rir_std_normalization=False, gwa_scaling_compensation=True)
dataloader = DataLoader(dataset, shuffle=True,
                        num_workers=4, pin_memory=False,
                        collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)

############################################ Models #############################################

def print_model_params(model : torch.nn.Module):
    # get the total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")


# Init baseline
mesh_net = MESH_NET()
mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth").eval().to(DEVICE)
net_G = STAGE1_G()
net_G = load_GAN(net_G, "./models/MESH2IR/netG_epoch_175.pth").eval().to(DEVICE)
mesh2ir = MESH2IR_FULL(mesh_net, net_G).eval().to(DEVICE)
print("")

# Init RIRBox
mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=2, MLP_Depth=4).eval().to(DEVICE)
mesh_to_shoebox = load_mesh_to_shoebox(mesh_to_shoebox, rirbox_path)
shoebox_to_rir = ShoeboxToRIR(16000, max_order=RIRBOX_MAX_ORDER, rir_length=3968, start_from_ir_onset=True).eval().to(DEVICE)
rirbox = RIRBox_FULL(mesh_to_shoebox, shoebox_to_rir, return_sbox=True).eval().to(DEVICE)
print("")

# Init Hybrid Model (Under construction)
hybrid = RIRBox_MESH2IR_Hybrid(mesh_to_shoebox, shoebox_to_rir).eval().to(DEVICE)
print("")

############################################ Inference #############################################

with torch.no_grad():
    for x_batch, edge_index_batch, batch_indexes, label_rir, label_origin, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Inference"):
        # Moving data to device
        x_batch = x_batch.to(DEVICE)
        edge_index_batch = edge_index_batch.to(DEVICE)
        batch_indexes = batch_indexes.to(DEVICE)
        mic_pos_batch = mic_pos_batch.to(DEVICE)
        src_pos_batch = src_pos_batch.to(DEVICE)

        # Find Ground Truth theoretical direct path onset
        distance = torch.linalg.norm(mic_pos_batch[0]-src_pos_batch[0])
        dp_onset_in_samples = int(distance*16000/343)

        ############################ Forward passes #############################

        # MESH2IR
        rir_mesh2ir = mesh2ir(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
        assert(rir_mesh2ir.shape[1] == 4096)
        if SCALE_MESH2IR_BY_ITS_ESTIMATED_STD: rir_mesh2ir = rir_mesh2ir*torch.mean(rir_mesh2ir[:,-64:], dim=1).unsqueeze(1).expand(-1, 4096)
        if SCALE_MESH2IR_GWA_SCALING_COMPENSATION: rir_mesh2ir = rir_mesh2ir / 0.0625029951333999
            
        rir_mesh2ir = rir_mesh2ir[:,:3968]
        if MESH2IR_USES_LABEL_ORIGIN: origin_mesh2ir = label_origin
        else : origin_mesh2ir = torch.tensor([GWA_3DFRONT_Dataset._estimate_origin(rir_mesh2ir.cpu().numpy())])
        
        # RIRBOX
        rir_rirbox, origin_rirbox, latent_vector = rirbox(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
        if RESPATIALIZE_RIRBOX: rir_rirbox, origin_rirbox = ShoeboxToRIR.respatialize_rirbox(rir_rirbox, dp_onset_in_samples)
        virtual_shoebox = ShoeboxToRIR.extract_shoebox_from_latent_representation(latent_vector)

        # Hybrid model
        if FILTER_MESH2IR_IN_HYBRID :
            rir_mesh2ir_filtered = filter_rir_like_rirbox(rir_mesh2ir)
            hybrid_rir, hybrid_origin = hybrid(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch, rir_mesh2ir_filtered, origin_mesh2ir)
        else:
            hybrid_rir, hybrid_origin = hybrid(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch, rir_mesh2ir, origin_mesh2ir)
        if RESPATIALIZE_RIRBOX: hybrid_rir, hybrid_origin = ShoeboxToRIR.respatialize_rirbox(hybrid_rir, dp_onset_in_samples)

        ############################ Plotting #############################

        def format_text(batch_of_coords):
            return [float(str(coord)[:5]) for coord in batch_of_coords.squeeze().cpu().numpy().tolist()]

        # preprocessing for plotting
        rir_mesh2ir = abs(rir_mesh2ir[0].cpu().numpy())
        rir_rirbox = abs(rir_rirbox[0].cpu().numpy())
        label_rir = abs(label_rir[0].cpu().numpy())
        hybrid_rir = abs(hybrid_rir[0].cpu().numpy())

        origin_mesh2ir = origin_mesh2ir.cpu().numpy()
        origin_rirbox = origin_rirbox.cpu().numpy()
        label_origin = label_origin.cpu().numpy()
        hybrid_origin = hybrid_origin.cpu().numpy()

        # # plot rirs with subplots
        fig, axs = plt.subplots(4, 1, figsize=(9, 9.5))
        fig.suptitle('RIR comparison between MESH2IR and RIRBOX')
        axs[0].set_title('MESH2IR')
        if not TOA_SYNCHRONIZATION:
            axs[0].plot(rir_mesh2ir, label="MESH2IR", color='blue')
            axs[0].axvline(x=origin_mesh2ir, color='red', linestyle='--', label='Origin')
            axs[0].axvline(x=dp_onset_in_samples, color='black', alpha=0.5, linestyle='--', label='DP Origin')
        else:
            axs[0].plot(rir_mesh2ir[max(0,int(label_origin[0]-41)):], label="MESH2IR", color='blue')
        axs[0].set_xlim(0, 4096)
        axs[0].set_ylim(0.0, 1.0)
        axs[0].grid(ls="--", alpha=0.5)
        axs[0].legend()

        axs[1].set_title('RIRBOX : ' + " ".join(rirbox_path.split('/')[-1].split('.')[0].split('_')[1:]))
        if not TOA_SYNCHRONIZATION:
            axs[1].plot(rir_rirbox, label="RIRBOX", color='orange')
            axs[1].axvline(x=origin_rirbox, color='red', linestyle='--', label='Origin')
            axs[1].axvline(x=dp_onset_in_samples, color='black', alpha=0.5, linestyle='--', label='DP Origin')
        else:
            axs[1].plot(rir_rirbox[max(0,int(origin_rirbox[0]-41)):], label="RIRBOX", color='orange')
        axs[1].set_xlim(0, 4096)
        axs[1].set_ylim(0.0, 1.0)
        axs[1].grid(ls="--", alpha=0.5)
        axs[1].legend()
        axs[1].text(0.7, 0.75, f"Room dim {format_text(virtual_shoebox[0])}\nMic pos {format_text(virtual_shoebox[1])}\nSrc pos {format_text(virtual_shoebox[2])}\nAbsorption {format_text(virtual_shoebox[3])[3:]}",
                    horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)

        axs[2].set_title('GT')
        if not TOA_SYNCHRONIZATION:
            axs[2].plot(label_rir, label="GT", color='green')
            axs[2].axvline(x=label_origin, color='red', linestyle='--', label='Origin')
            axs[2].axvline(x=dp_onset_in_samples, color='black', alpha=0.5, linestyle='--', label='DP Origin')
        else:
            axs[2].plot(label_rir[max(0,int(label_origin[0]-41)):], label="GT", color='green')
        axs[2].set_xlim(0, 4096)
        axs[2].set_ylim(0.0, 1.0)
        axs[2].grid(ls="--", alpha=0.5)
        axs[2].legend()
        axs[2].text(0.7, 0.85, f"GT Mic pos {format_text(mic_pos_batch)}\nGT Src pos {format_text(src_pos_batch)}",
                    horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)

        axs[3].set_title('(RIRBOX + MESH2IR) Hybrid. Mixing point based on estimated RIRBOX room dimensions.')
        if not TOA_SYNCHRONIZATION:
            axs[3].plot(hybrid_rir, label="Hybrid", color='C3')
            axs[3].axvline(x=hybrid_origin, color='red', linestyle='--', label='Origin')
            axs[3].axvline(x=dp_onset_in_samples, color='black', alpha=0.5, linestyle='--', label='DP Origin')
        else:
            axs[3].plot(hybrid_rir[max(0,int(hybrid_origin[0]-41)):], label="Hybrid", color='C3')
        axs[3].set_xlim(0, 4096)
        axs[3].set_ylim(0.0, 1.0)
        axs[3].grid(ls="--", alpha=0.5)
        axs[3].legend()
        axs[3].text(0.7, 0.75, f"Room dim {format_text(virtual_shoebox[0])}\nMic pos {format_text(virtual_shoebox[1])}\nSrc pos {format_text(virtual_shoebox[2])}\nAbsorption {format_text(virtual_shoebox[3])[3:]}",
                    horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)

        plt.tight_layout()
        plt.show()

        if plot_stft:
            # # plot rirs with subplots
            fig, axs = plt.subplots(4, 1, figsize=(9, 9))

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

            f, t, Zxx = stft(abs(np.array(hybrid_rir, dtype=float)), 16000, nperseg=256)
            axs[3].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            axs[3].set_title("Hybrid STFT Magnitude")
            axs[3].set_ylabel('Frequency [Hz]')
            axs[3].set_xlabel('Time [sec]')
            plt.tight_layout()
            plt.show()