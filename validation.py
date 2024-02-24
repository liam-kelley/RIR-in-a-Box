import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET, STAGE1_G, MESH2IR_FULL
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR, RIRBox_FULL
from models.utility import load_mesh_net, load_GAN, load_mesh_to_shoebox
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
from pyLiam.LKTimer import LKTimer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def validation_metric_accuracy_mesh2ir_vs_rirbox():
    '''
    Validation of the metric accuracy of the MESH2IR and RIRBOX models on the GWA_3DFRONT dataset.
    '''

    ############################################ Config ############################################

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 4 # Batch evaluation

    print("PARAMETERS:")
    print("    > BATCH_SIZE = ", BATCH_SIZE)
    if DEVICE == 'cuda':
        if not torch.cuda.is_available():
            DEVICE = 'cpu'
            print("    CUDA not available, using CPU")
    print("    > DEVICE = ", DEVICE)

    ############################################ Inits #############################################

    # Init baseline
    mesh_net = MESH_NET()
    mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth")
    net_G = STAGE1_G()
    net_G = load_GAN(net_G, "./models/MESH2IR/netG_epoch_175.pth")
    mesh2ir = MESH2IR_FULL(mesh_net, net_G).eval().to(DEVICE)
    print("")

    # Init Rirbox
    mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=2)
    mesh_to_shoebox = load_mesh_to_shoebox(mesh_to_shoebox, "./models/RIRBox/mesh_to_shoebox_epoch_1.pth")
    shoebox_to_rir = ShoeboxToRIR(16000, max_order=10)
    rirbox = RIRBox_FULL(mesh_to_shoebox, shoebox_to_rir).eval().to(DEVICE)
    print("")

    ################################################################################################
    ############################################ Metric validation #################################
    ################################################################################################

    # data
    dataset=GWA_3DFRONT_Dataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True,
                            collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
    print("")

    # metrics
    edc=EnergyDecay_Loss(frequency_wise=False,
                        synchronize_TOA=True,
                        normalize_dp=False,
                        normalize_decay_curve=True,
                        deemphasize_early_reflections=True,
                        pad_to_same_length=False,
                        crop_to_same_length=True).to(DEVICE)
    mrstft=MRSTFT_Loss(sample_rate=16000,
                    device=DEVICE,
                    synchronize_TOA=True,
                    deemphasize_early_reflections=True,
                    normalize_dp=True,
                    pad_to_same_length=False,
                    crop_to_same_length=True).to(DEVICE)
    acm=AcousticianMetrics_Loss(sample_rate=16000,
                                synchronize_TOA=True, 
                                crop_to_same_length=True,
                                normalize_dp=False,
                                frequency_wise=False,
                                normalize_total_energy=False,
                                pad_to_same_length=False,
                                MeanAroundMedian_pruning=False).to(DEVICE)
    print("")


    # create empty dataframe with columns for each loss for each model
    df = pd.DataFrame(columns=["mesh2ir_edr", "mesh2ir_mrstft", "mesh2ir_c80", "mesh2ir_D", "mesh2ir_rt60",
                                "rirbox_edr", "rirbox_mrstft", "rirbox_c80", "rirbox_D", "rirbox_rt60"])

    # iterate over the dataset
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Metric validation"):
        # Moving data to device
        x_batch = x_batch.to(DEVICE)
        edge_index_batch = edge_index_batch.to(DEVICE)
        batch_indexes = batch_indexes.to(DEVICE)
        mic_pos_batch = mic_pos_batch.to(DEVICE)
        src_pos_batch = src_pos_batch.to(DEVICE)

        # Forward passes
        rir_mesh2ir = mesh2ir(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
        origin_mesh2ir = torch.tensor(np.repeat(41, BATCH_SIZE))

        rir_rirbox, origin_rirbox = rirbox(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)

        # Compute losses
        loss_mesh2ir_edr = edc(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
        loss_mesh2ir_mrstft = mrstft(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
        loss_mesh2ir_c80, loss_mesh2ir_D, loss_mesh2ir_rt60, _ = acm(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)

        loss_rirbox_edr = edc(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
        loss_rirbox_mrstft = mrstft(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
        loss_rirbox_c80, loss_rirbox_D, loss_rirbox_rt60, _ = acm(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)

        # Append to dataframe
        df = df.append({"mesh2ir_edr": loss_mesh2ir_edr.item(), "mesh2ir_mrstft": loss_mesh2ir_mrstft.item(),
                        "mesh2ir_c80": loss_mesh2ir_c80.item(), "mesh2ir_D": loss_mesh2ir_D.item(),
                        "mesh2ir_rt60": loss_mesh2ir_rt60.item(),
                        "rirbox_edr": loss_rirbox_edr.item(), "rirbox_mrstft": loss_rirbox_mrstft.item(),
                        "rirbox_c80": loss_rirbox_c80.item(), "rirbox_D": loss_rirbox_D.item(),
                        "rirbox_rt60": loss_rirbox_rt60.item()}, ignore_index=True)
        
    # Save dataframe
    df.to_csv("./validation_results/metric_accuracy_mesh2ir_vs_rirbox.csv")

def view_results_metric_accuracy_mesh2ir_vs_rirbox():
    df = pd.read_csv("./validation_results/metric_accuracy_mesh2ir_vs_rirbox.csv")

    # Please perform average of the losses for each model and print the results as a table

    # Average of the losses for each model
    df = df.mean()
    print(df)

def main():
    validation_metric_accuracy_mesh2ir_vs_rirbox()
    view_results_metric_accuracy_mesh2ir_vs_rirbox()

if __name__ == "__main__":
    main()
