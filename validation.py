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

def validation_metric_accuracy_mesh2ir_vs_rirbox(rirbox_path=None, validation_iterations=10, plot_rirs=False):
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
    mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth").eval().to(DEVICE)
    net_G = STAGE1_G()
    net_G = load_GAN(net_G, "./models/MESH2IR/netG_epoch_175.pth").eval().to(DEVICE)
    mesh2ir = MESH2IR_FULL(mesh_net, net_G).eval().to(DEVICE)
    print("")

    # Init Rirbox
    mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=2)
    if rirbox_path is not None:
        mesh_to_shoebox = load_mesh_to_shoebox(mesh_to_shoebox, rirbox_path)
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
            origin_mesh2ir = torch.tensor(np.repeat(41, BATCH_SIZE)).to(DEVICE)

            rir_rirbox, origin_rirbox = rirbox(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)

            # Compute losses
            loss_mesh2ir_edr = edc(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_mrstft = mrstft(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_c80, loss_mesh2ir_D, loss_mesh2ir_rt60, _ = acm(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)

            loss_rirbox_edr = edc(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_mrstft = mrstft(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_c80, loss_rirbox_D, loss_rirbox_rt60, _ = acm(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)

            # # plot rirs with subplots
            if plot_rirs and (i == 0 or i == 1):
                fig, axs = plt.subplots(3, 1, figsize=(9, 9))
                fig.suptitle('RIR comparison between MESH2IR and RIRBOX')
                axs[0].plot(abs(rir_mesh2ir[0].cpu().numpy()), label="MESH2IR", color='blue')
                axs[0].set_title('MESH2IR')
                axs[0].grid(ls="--", alpha=0.5)
                axs[0].set_xlim(0, 4096)
                axs[1].plot(abs(rir_rirbox[0].cpu().numpy()), label="RIRBOX", color='orange')
                axs[1].set_title('RIRBOX')
                axs[1].grid(ls="--", alpha=0.5)
                axs[1].set_xlim(0, 4096)
                axs[2].plot(abs(label_rir_batch[0].cpu().numpy()), label="GT", color='green')
                axs[2].set_title('GT')
                axs[2].grid(ls="--", alpha=0.5)
                axs[2].set_xlim(0, 4096)
                # Show plot
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

    # Perform average of the losses for each model and print the results as a table
    df_mean = df.mean()
    print(df_mean)

    # Plot the results as a different bar plot for each metric
    # use subplots
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Metric accuracy comparison between MESH2IR and RIRBOX')

    # EDR
    axs[0, 0].bar(["MESH2IR", "RIRBOX"], [df_mean["mesh2ir_edr"], df_mean["rirbox_edr"]])
    axs[0, 0].set_title('EDR')
    axs[0, 0].set_ylabel('Mean Error')

    # MRSTFT
    axs[0, 1].bar(["MESH2IR", "RIRBOX"], [df_mean["mesh2ir_mrstft"], df_mean["rirbox_mrstft"]])
    axs[0, 1].set_title('MRSTFT')
    axs[0, 1].set_ylabel('Mean Error')

    # C80
    axs[0, 2].bar(["MESH2IR", "RIRBOX"], [df_mean["mesh2ir_c80"], df_mean["rirbox_c80"]])
    axs[0, 2].set_title('C80')
    axs[0, 2].set_ylabel('Mean Error')

    # D
    axs[1, 0].bar(["MESH2IR", "RIRBOX"], [df_mean["mesh2ir_D"], df_mean["rirbox_D"]])
    axs[1, 0].set_title('D')
    axs[1, 0].set_ylabel('Mean Error')

    # RT60
    axs[1, 1].bar(["MESH2IR", "RIRBOX"], [df_mean["mesh2ir_rt60"], df_mean["rirbox_rt60"]])
    axs[1, 1].set_title('RT60')
    axs[1, 1].set_ylabel('Mean Error')

    # Remove last subplot
    fig.delaxes(axs[1, 2])

    # Show plot
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rirbox_path', type=str, default="./training/rirbox_model2_finetune.json", help='Path to rirbox model to validate.')
    args, _ = parser.parse_known_args()
    
    validation_metric_accuracy_mesh2ir_vs_rirbox(rirbox_path=args.rirbox_path,
                                                 validation_iterations=20, plot_rirs=False)
    view_results_metric_accuracy_mesh2ir_vs_rirbox()

if __name__ == "__main__":
    main()
