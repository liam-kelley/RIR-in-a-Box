import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET, STAGE1_G, MESH2IR_FULL
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR, RIRBox_FULL
from models.utility import load_mesh_net, load_GAN, load_mesh_to_shoebox
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
from tools.pyLiam.LKTimer import LKTimer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import argparse


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
rirbox_path = "./models/RIRBOX/RIRBOX_Model2_Finetune_apricot-frost-16.pth"
ribox_max_order = 15

############################################ Inits #############################################

# Init baseline
mesh_net = MESH_NET()
mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth").eval().to(DEVICE)
net_G = STAGE1_G()
net_G = load_GAN(net_G, "./models/MESH2IR/netG_epoch_175.pth").eval().to(DEVICE)
mesh2ir = MESH2IR_FULL(mesh_net, net_G).eval().to(DEVICE)
print("")

# Init Rirbox
mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=2)
if rirbox_path is not None:
    mesh_to_shoebox = load_mesh_to_shoebox(mesh_to_shoebox, rirbox_path)
shoebox_to_rir = ShoeboxToRIR(16000, max_order=ribox_max_order)
rirbox = RIRBox_FULL(mesh_to_shoebox, shoebox_to_rir).eval().to(DEVICE)
print("")

# data
dataset=GWA_3DFRONT_Dataset()
dataloader = DataLoader(dataset, shuffle=True,
                        num_workers=4, pin_memory=False,
                        collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)

############################################ Inference #############################################

with torch.no_grad():
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Inference"):
        # Moving data to device
        x_batch = x_batch.to(DEVICE)
        edge_index_batch = edge_index_batch.to(DEVICE)
        batch_indexes = batch_indexes.to(DEVICE)
        mic_pos_batch = mic_pos_batch.to(DEVICE)
        src_pos_batch = src_pos_batch.to(DEVICE)
        # label_rir_batch = label_rir_batch.to(DEVICE)
        # label_origin_batch = label_origin_batch.to(DEVICE)

        # Forward passes
        rir_mesh2ir = mesh2ir(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
        origin_mesh2ir = GWA_3DFRONT_Dataset._estimate_origin(rir_mesh2ir.cpu().numpy())

        rir_rirbox, origin_rirbox = rirbox(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)

        # preprocessing for plotting
        rir_mesh2ir=abs(rir_mesh2ir[0].cpu().numpy())
        origin_mesh2ir=origin_mesh2ir
        rir_rirbox = abs(rir_rirbox[0].cpu().numpy())
        origin_rirbox = origin_rirbox.cpu().numpy()
        label_rir = abs(label_rir_batch[0].cpu().numpy())
        label_origin = label_origin_batch.cpu().numpy()

        # # plot rirs with subplots
        fig, axs = plt.subplots(3, 1, figsize=(9, 9))
        fig.suptitle('RIR comparison between MESH2IR and RIRBOX')
        axs[0].plot(rir_mesh2ir, label="MESH2IR", color='blue')
        axs[0].axvline(x=origin_mesh2ir, color='red', linestyle='--', label='Origin')
        axs[0].set_title('MESH2IR')
        axs[0].grid(ls="--", alpha=0.5)
        axs[0].legend()
        axs[0].set_xlim(0, 4096)
        axs[1].plot(rir_rirbox, label="RIRBOX", color='orange')
        axs[1].axvline(x=origin_rirbox, color='red', linestyle='--', label='Origin')
        axs[1].set_title('RIRBOX')
        axs[1].legend()
        axs[1].grid(ls="--", alpha=0.5)
        axs[1].set_xlim(0, 4096)
        axs[2].plot(label_rir, label="GT", color='green')
        axs[2].axvline(x=label_origin, color='red', linestyle='--', label='Origin')
        axs[2].set_title('GT')
        axs[2].grid(ls="--", alpha=0.5)
        axs[2].legend()
        axs[2].set_xlim(0, 4096)
        plt.tight_layout()
        plt.show()