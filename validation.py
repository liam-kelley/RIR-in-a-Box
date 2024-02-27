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

def validation_metric_accuracy_mesh2ir_vs_rirbox(rirbox_path=None, validation_iterations=10):
    '''
    Validation of the metric accuracy of the MESH2IR and RIRBOX models on the GWA_3DFRONT dataset.
    '''

    ############################################ Config ############################################

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1 # Batch validation ?
    RIRBOX_MAX_ORDER = 15 # please make sure it's the same as the one used during training TODO do this better
    DATALOADER_NUM_WORKERS = 12
    synchronizing_TOA_for_mesh2ir = False # No TOA synchronizing for mesh3ir, yes for rirbox

    print("PARAMETERS:")
    print("    > BATCH_SIZE = ", BATCH_SIZE)
    print("    > DEVICE = ", DEVICE)

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
    shoebox_to_rir = ShoeboxToRIR(16000, max_order=RIRBOX_MAX_ORDER)
    rirbox = RIRBox_FULL(mesh_to_shoebox, shoebox_to_rir).eval().to(DEVICE)
    print("")

    ################################################################################################
    ############################################ Metric validation #################################
    ################################################################################################

    # data
    dataset=GWA_3DFRONT_Dataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=DATALOADER_NUM_WORKERS, pin_memory=False,
                            collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
    print("")

    # metrics
    edc_mesh2ir=EnergyDecay_Loss(frequency_wise=False,
                        synchronize_TOA=synchronizing_TOA_for_mesh2ir, # No TOA synchronizing for mesh3ir, yes for rirbox
                        normalize_dp=False,
                        normalize_decay_curve=False,
                        deemphasize_early_reflections=False, # No deemphasis for validation, only for training
                        pad_to_same_length=False,
                        crop_to_same_length=True).to(DEVICE)
    edc_rirbox=EnergyDecay_Loss(frequency_wise=False,
                        synchronize_TOA=True,         # No TOA synchronizing for mesh3ir, yes for rirbox  
                        normalize_dp=False,
                        normalize_decay_curve=False,
                        deemphasize_early_reflections=False, # No deemphasis for validation, only for training
                        pad_to_same_length=False,
                        crop_to_same_length=True).to(DEVICE)
    
    mrstft_mesh2ir=MRSTFT_Loss(sample_rate=16000,
                    device=DEVICE,
                    synchronize_TOA=synchronizing_TOA_for_mesh2ir,   # No TOA synchronizing for mesh3ir, yes for rirbox 
                    deemphasize_early_reflections=False, # No deemphasis for validation, only for training
                    normalize_dp=False,
                    pad_to_same_length=False,
                    crop_to_same_length=True).to(DEVICE)
    mrstft_rirbox=MRSTFT_Loss(sample_rate=16000,
                    device=DEVICE,
                    synchronize_TOA=True,   # No TOA synchronizing for mesh3ir, yes for rirbox
                    deemphasize_early_reflections=False, # No deemphasis for validation, only for training
                    normalize_dp=False,
                    pad_to_same_length=False,
                    crop_to_same_length=True).to(DEVICE)
    
    acm_mesh2ir=AcousticianMetrics_Loss(sample_rate=16000,
                                synchronize_TOA=synchronizing_TOA_for_mesh2ir,  # No TOA synchronizing for mesh3ir, yes for rirbox
                                crop_to_same_length=True,
                                normalize_dp=False,
                                frequency_wise=False,
                                normalize_total_energy=False,
                                pad_to_same_length=False,
                                MeanAroundMedian_pruning=False).to(DEVICE)
    acm_rirbox=AcousticianMetrics_Loss(sample_rate=16000,
                                synchronize_TOA=True,    # No TOA synchronizing for mesh3ir, yes for rirbox
                                crop_to_same_length=True,
                                normalize_dp=False,
                                frequency_wise=False,
                                normalize_total_energy=False,
                                pad_to_same_length=False,
                                MeanAroundMedian_pruning=False).to(DEVICE)
    print("")

    with torch.no_grad():
        my_list = []

        i = 0
        # iterate over the dataset
        for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Metric validation"):
            # Moving data to device
            x_batch = x_batch.to(DEVICE)
            edge_index_batch = edge_index_batch.to(DEVICE)
            batch_indexes = batch_indexes.to(DEVICE)
            mic_pos_batch = mic_pos_batch.to(DEVICE)
            src_pos_batch = src_pos_batch.to(DEVICE)
            label_rir_batch = label_rir_batch.to(DEVICE)
            label_origin_batch = label_origin_batch.to(DEVICE)

            # Forward passes
            rir_mesh2ir = mesh2ir(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
            origin_mesh2ir = torch.tensor([GWA_3DFRONT_Dataset._estimate_origin(rir_mesh2ir.cpu().numpy())]).to(DEVICE)

            rir_rirbox, origin_rirbox = rirbox(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)

            # Compute losses
            loss_mesh2ir_edr = edc_mesh2ir(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_mrstft = mrstft_mesh2ir(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_c80, loss_mesh2ir_D, loss_mesh2ir_rt60, _ = acm_mesh2ir(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)

            loss_rirbox_edr = edc_rirbox(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_mrstft = mrstft_rirbox(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_c80, loss_rirbox_D, loss_rirbox_rt60, _ = acm_rirbox(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)

            # # plot rirs with subplots
            plot_rirs = False
            if plot_rirs:
                fig, axs = plt.subplots(3, 1, figsize=(9, 9))
                fig.suptitle('RIR comparison between MESH2IR and RIRBOX')
                axs[0].plot(rir_mesh2ir[0].cpu().numpy(), label="MESH2IR", color='blue')
                axs[0].axvline(x=origin_mesh2ir.cpu().numpy(), color='red', linestyle='--', label='Origin')
                axs[0].set_title('MESH2IR')
                axs[0].grid(ls="--", alpha=0.5)
                axs[0].legend()
                axs[0].set_xlim(0, 4096)
                axs[1].plot(rir_rirbox[0].cpu().numpy(), label="RIRBOX", color='orange')
                axs[1].axvline(x=origin_rirbox.cpu().numpy(), color='red', linestyle='--', label='Origin')
                axs[1].set_title('RIRBOX')
                axs[1].legend()
                axs[1].grid(ls="--", alpha=0.5)
                axs[1].set_xlim(0, 4096)
                axs[2].plot(abs(label_rir_batch[0].cpu().numpy()), label="GT", color='green')
                axs[2].axvline(x=label_origin_batch.cpu().numpy(), color='red', linestyle='--', label='Origin')
                axs[2].set_title('GT')
                axs[2].grid(ls="--", alpha=0.5)
                axs[2].legend()
                axs[2].set_xlim(0, 4096)
                plt.tight_layout()
                plt.show()

            # Append to dataframe
            my_list.append([loss_mesh2ir_edr.cpu().item(),
                            loss_rirbox_edr.cpu().item(),
                            loss_mesh2ir_mrstft.cpu().item(),
                            loss_rirbox_mrstft.cpu().item(),
                            loss_mesh2ir_c80.cpu().item(),
                            loss_rirbox_c80.cpu().item(),
                            loss_mesh2ir_D.cpu().item(),
                            loss_rirbox_D.cpu().item(),
                            loss_mesh2ir_rt60.cpu().item(),
                            loss_rirbox_rt60.cpu().item()])

            i += 1
            if i == validation_iterations:
                break

    # Save as dataframe
    df = pd.DataFrame(my_list, columns=["mesh2ir_edr", "rirbox_edr",
                                        "mesh2ir_mrstft", "rirbox_mrstft",
                                        "mesh2ir_c80", "rirbox_c80",
                                        "mesh2ir_D", "rirbox_D",
                                        "mesh2ir_rt60", "rirbox_rt60"])
    df = df.apply(np.sqrt) # removes the square from the MSEs
    df.to_csv("./validation_results/metric_accuracy_mesh2ir_vs_rirbox.csv")

def view_results_metric_accuracy_mesh2ir_vs_rirbox():
    df = pd.read_csv("./validation_results/metric_accuracy_mesh2ir_vs_rirbox.csv")

    df_mean = df.mean()
    df_std = df.std()

    fig, axs = plt.subplots(1,5, figsize=(12, 5))
    fig.suptitle('Metric accuracy comparison between MESH2IR and RIRBOX')

    model_names = ["Baseline", "RIRBOX"]
    colors = ['C0', 'C1']

    # EDR
    axs[0].bar(model_names, [df_mean["mesh2ir_edr"], df_mean["rirbox_edr"]],
                yerr=[df_std["mesh2ir_edr"], df_std["rirbox_edr"]],color=colors, capsize=20)
    axs[0].set_title('EDR')
    axs[0].set_ylabel('Mean Error')

    # MRSTFT
    axs[1].bar(model_names, [df_mean["mesh2ir_mrstft"], df_mean["rirbox_mrstft"]],
                yerr=[df_std["mesh2ir_mrstft"], df_std["rirbox_mrstft"]],color=colors, capsize=20)
    axs[1].set_title('MRSTFT')
    axs[1].set_ylabel('Mean Error')

    # C80
    axs[2].bar(model_names, [df_mean["mesh2ir_c80"], df_mean["rirbox_c80"]],
                yerr=[df_std["mesh2ir_c80"], df_std["rirbox_c80"]],color=colors, capsize=20)
    axs[2].set_title('C80')
    axs[2].set_ylabel('Mean Error')

    # # delete axs 2
    # fig.delaxes(axs[2])

    # D
    axs[3].bar(model_names, [df_mean["mesh2ir_D"], df_mean["rirbox_D"]],
                yerr=[df_std["mesh2ir_D"], df_std["rirbox_D"]],color=colors, capsize=20)
    axs[3].set_title('D')
    axs[3].set_ylabel('Mean Error')

    # RT60
    axs[4].bar(model_names, [df_mean["mesh2ir_rt60"], df_mean["rirbox_rt60"]],
                yerr=[df_std["mesh2ir_rt60"], df_std["rirbox_rt60"]],color=colors, capsize=20)
    axs[4].set_title('RT60')
    axs[4].set_ylabel('Mean Error')

    for ax in axs:
        ax.grid(ls="--", alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rirbox_path', type=str, default="./models/RIRBOX/RIRBOX_Model2_Finetune_worldly-lion-25.pth",
                        help='Path to rirbox model to validate.')
    args, _ = parser.parse_known_args()
    
    validation_metric_accuracy_mesh2ir_vs_rirbox(rirbox_path=args.rirbox_path, validation_iterations=10000)
    view_results_metric_accuracy_mesh2ir_vs_rirbox()

if __name__ == "__main__":
    main()
