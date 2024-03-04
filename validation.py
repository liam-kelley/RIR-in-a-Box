import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET, STAGE1_G, MESH2IR_FULL
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR, RIRBox_FULL, RIRBox_MESH2IR_Hybrid
from models.utility import load_mesh_net, load_GAN, load_mesh_to_shoebox
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
from training.utility import filter_rir_like_rirbox
from tools.pyLiam.LKTimer import LKTimer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D
from json import load

def validation_metric_accuracy_mesh2ir_vs_rirbox(model_config="./training/ablation2/rirbox_model3_MRSTFT_EDR_MLPDEPTH4.json",
                                                 validation_iterations=10):
    '''
    Validation of the metric accuracy of the MESH2IR and RIRBOX models on the GWA_3DFRONT dataset.
    '''

    ############################################ Config ############################################

    TOA_SYNCHRONIZATION = True
    SCALE_MESH2IR_BY_ITS_ESTIMATED_STD = True # If True, cancels out the std normalization used during mesh2ir's training
    SCALE_MESH2IR_GWA_SCALING_COMPENSATION = True # If true, cancels out the scaling compensation mesh2ir learned from the GWA dataset during training.
    MESH2IR_USES_LABEL_ORIGIN = False
    RESPATIALIZE_RIRBOX = True
    FILTER_MESH2IR_IN_HYBRID = False

    with open(model_config, 'r') as file: config = load(file)
    config['RIRBOX_MAX_ORDER'] = 15 # Feel free to increase this value maybe?

    config["DATALOADER_NUM_WORKERS"] = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['BATCH_SIZE'] = 1
    if config['SAVE_PATH'] == "": config['SAVE_PATH'] = "./models/RIRBOX/"+ model_config.split("/")[-2] + "/"+ model_config.split("/")[-1].split(".")[0] + ".pth"

    print("PARAMETERS:")
    for key, value in config.items():
        print(f"    > {key} = {value}")
    print("")

    ############################################ Models #############################################

    # Init baseline
    mesh_net = MESH_NET()
    mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth").eval().to(DEVICE)
    net_G = STAGE1_G()
    net_G = load_GAN(net_G, "./models/MESH2IR/netG_epoch_175.pth").eval().to(DEVICE)
    mesh2ir = MESH2IR_FULL(mesh_net, net_G).eval().to(DEVICE)
    print("")

    # Init Rirbox
    mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=config['RIRBOX_MODEL_ARCHITECTURE'], MLP_Depth=config['MLP_DEPTH'])
    if config['SAVE_PATH'] is not None: mesh_to_shoebox = load_mesh_to_shoebox(mesh_to_shoebox, config['SAVE_PATH'])
    shoebox_to_rir = ShoeboxToRIR(16000, max_order=config['RIRBOX_MAX_ORDER'], rir_length=3968, start_from_ir_onset=True)
    rirbox = RIRBox_FULL(mesh_to_shoebox, shoebox_to_rir).eval().to(DEVICE)
    print("")

    # Init Hybrid Model
    hybrid = RIRBox_MESH2IR_Hybrid(mesh_to_shoebox, shoebox_to_rir).eval().to(DEVICE)
    print("")

    ################################################################################################
    ############################################ Metric validation #################################
    ################################################################################################

    # data
    dataset=GWA_3DFRONT_Dataset(csv_file="datasets/GWA_3DFRONT/gwa_3Dfront_validation_dp_only.csv",rir_std_normalization=False)
    dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=False,
                            num_workers=config['DATALOADER_NUM_WORKERS'], pin_memory=False,
                            collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
    print("")

    # metrics
    edc=EnergyDecay_Loss(frequency_wise=True,
                            synchronize_TOA=TOA_SYNCHRONIZATION, 
                            normalize_dp=False,
                            normalize_decay_curve=False,
                            deemphasize_early_reflections=False,
                            pad_to_same_length=False,
                            crop_to_same_length=True).to(DEVICE)
    
    mrstft=MRSTFT_Loss(sample_rate=16000,device=DEVICE,
                            synchronize_TOA=TOA_SYNCHRONIZATION,
                            deemphasize_early_reflections=False,
                            normalize_dp=False,
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

            # Find Ground Truth theoretical direct path onset
            distance = torch.linalg.norm(mic_pos_batch[0]-src_pos_batch[0])
            dp_onset_in_samples = int(distance*16000/343)

            ############################ Forward passes #############################

            # MESH2IR
            rir_mesh2ir = mesh2ir(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
            assert(rir_mesh2ir.shape[1] == 4096)
            if SCALE_MESH2IR_BY_ITS_ESTIMATED_STD: rir_mesh2ir = rir_mesh2ir*torch.mean(rir_mesh2ir[:,-64:], dim=1).unsqueeze(1).expand(-1, 4096)
            if SCALE_MESH2IR_GWA_SCALING_COMPENSATION: rir_mesh2ir = rir_mesh2ir / 0.0625029951333999
            
            rir_mesh2ir = rir_mesh2ir[:3968]
            if MESH2IR_USES_LABEL_ORIGIN: origin_mesh2ir = label_origin_batch
            else : origin_mesh2ir = torch.tensor([GWA_3DFRONT_Dataset._estimate_origin(rir_mesh2ir.cpu().numpy())]).to(DEVICE)

            # RIRBOX
            rir_rirbox, origin_rirbox, latent_vector = rirbox(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
            if RESPATIALIZE_RIRBOX: rir_rirbox, origin_rirbox = ShoeboxToRIR.respatialize_rirbox(rir_rirbox, dp_onset_in_samples)
            virtual_shoebox = ShoeboxToRIR.extract_shoebox_from_latent_representation(latent_vector)
            
            # Hybrid model
            if FILTER_MESH2IR_IN_HYBRID :
                rir_mesh2ir_filtered = filter_rir_like_rirbox(rir_mesh2ir)
                hybrid_rir, origin_hybrid = hybrid(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch, rir_mesh2ir_filtered, origin_mesh2ir)
            else:
                hybrid_rir, origin_hybrid = hybrid(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch, rir_mesh2ir, origin_mesh2ir)
            if RESPATIALIZE_RIRBOX: hybrid_rir, origin_hybrid = ShoeboxToRIR.respatialize_rirbox(hybrid_rir, dp_onset_in_samples)

            ############################ Get losses #############################

            # Compute losses
            loss_mesh2ir_edr = edc(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_mrstft = mrstft(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_c80, loss_mesh2ir_D, loss_mesh2ir_rt60, _ = acm(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)

            loss_rirbox_edr = edc(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_mrstft = mrstft(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_c80, loss_rirbox_D, loss_rirbox_rt60, _ = acm(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)

            loss_hybrid_edr = edc(hybrid_rir, origin_hybrid, label_rir_batch, label_origin_batch)
            loss_hybrid_mrstft = mrstft(hybrid_rir, origin_hybrid, label_rir_batch, label_origin_batch)
            loss_hybrid_c80, loss_hybrid_D, loss_hybrid_rt60, _ = acm(hybrid_rir, origin_hybrid, label_rir_batch, label_origin_batch)

            # Append to dataframe
            my_list.append([loss_mesh2ir_edr.cpu().item(),
                            loss_rirbox_edr.cpu().item(),
                            loss_hybrid_edr.cpu().item(),
                            loss_mesh2ir_mrstft.cpu().item(),
                            loss_rirbox_mrstft.cpu().item(),
                            loss_hybrid_mrstft.cpu().item(),
                            loss_mesh2ir_c80.cpu().item(),
                            loss_rirbox_c80.cpu().item(),
                            loss_hybrid_c80.cpu().item(),
                            loss_mesh2ir_D.cpu().item(),
                            loss_rirbox_D.cpu().item(),
                            loss_hybrid_D.cpu().item(),
                            loss_mesh2ir_rt60.cpu().item(),
                            loss_rirbox_rt60.cpu().item(),
                            loss_hybrid_rt60.cpu().item()])

            i += 1
            if i == validation_iterations:
                break

    # Save as dataframe
    df = pd.DataFrame(my_list, columns=["mesh2ir_edr", "rirbox_edr", "hybrid_edr",
                                        "mesh2ir_mrstft", "rirbox_mrstft", "hybrid_mrstft",
                                        "mesh2ir_c80", "rirbox_c80", "hybrid_c80",
                                        "mesh2ir_D", "rirbox_D", "hybrid_D",
                                        "mesh2ir_rt60", "rirbox_rt60", "hybrid_rt60"])
    df = df.apply(np.sqrt) # removes the square from the MSEs
    df.to_csv("./validation_results/" + config['SAVE_PATH'].split("/")[-1].split(".")[0] + ".csv")

def view_results_metric_accuracy_mesh2ir_vs_rirbox(results_csv="./validation_results/model.csv"):
    df = pd.read_csv(results_csv)

    df_mean = df.mean()
    df_std = df.std()

    fig, axs = plt.subplots(1,5, figsize=(14, 5))
    fig.suptitle(f'Metric accuracy validation. MESH2IR vs {results_csv.split("/")[-1].split(".")[0]}')

    # Prepare the data for the box plot
    model_names = ["Baseline", "RIRBOX"]#, "Hybrid"]
    colors = ['C0', 'C1', 'C2']

    mean_marker = Line2D([], [], color='w', marker='^', markerfacecolor='green', markersize=10, label='Mean')

    # EDR
    # axs[0].bar(model_names, [df_mean["mesh2ir_edr"], df_mean["rirbox_edr"], df_mean["hybrid_edr"]],
    #             yerr=[df_std["mesh2ir_edr"], df_std["rirbox_edr"], df_std["hybrid_edr"]],
    #             color=colors, capsize=20)
    # axs[0].boxplot([df["mesh2ir_edr"], df["rirbox_edr"], df["hybrid_edr"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[0].boxplot([df["mesh2ir_edr"], df["rirbox_edr"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[0].set_title('EDR')
    axs[0].set_ylabel('EDR Error')
    # Add green triangle to the legend that says it represents mean
    # Create a custom legend entry for the mean marker
    axs[0].legend(handles=[mean_marker])

    # MRSTFT
    # axs[1].bar(model_names, [df_mean["mesh2ir_mrstft"], df_mean["rirbox_mrstft"], df_mean["hybrid_mrstft"]],
    #             yerr=[df_std["mesh2ir_mrstft"], df_std["rirbox_mrstft"], df_std["hybrid_mrstft"]],
    #             color=colors, capsize=20)
    # axs[1].set_title('MRSTFT')
    # axs[1].set_ylabel('Mean Error')
    # axs[1].boxplot([df["mesh2ir_mrstft"], df["rirbox_mrstft"], df["hybrid_mrstft"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[1].boxplot([df["mesh2ir_mrstft"], df["rirbox_mrstft"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[1].set_title('MRSTFT')
    axs[1].set_ylabel('MRSTFT Error')
    axs[1].legend(handles=[mean_marker])

    # C80
    # axs[2].bar(model_names, [df_mean["mesh2ir_c80"], df_mean["rirbox_c80"], df_mean["hybrid_c80"]],
    #             yerr=[df_std["mesh2ir_c80"], df_std["rirbox_c80"], df_std["hybrid_c80"]],
    #             color=colors, capsize=20)
    # axs[2].set_title('C80')
    # axs[2].set_ylabel('Mean Error')
    # axs[2].boxplot([df["mesh2ir_c80"], df["rirbox_c80"], df["hybrid_c80"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[2].boxplot([df["mesh2ir_c80"], df["rirbox_c80"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[2].set_title('C80')
    axs[2].set_ylabel('C80 Error')
    axs[2].legend(handles=[mean_marker])

    # # delete axs 2
    # fig.delaxes(axs[2])

    # D
    # axs[3].bar(model_names, [df_mean["mesh2ir_D"], df_mean["rirbox_D"], df_mean["hybrid_D"]],
    #             yerr=[df_std["mesh2ir_D"], df_std["rirbox_D"], df_std["hybrid_D"]],
    #             color=colors, capsize=20)
    # axs[3].set_title('D')
    # axs[3].set_ylabel('Mean Error')
    # axs[3].boxplot([df["mesh2ir_D"], df["rirbox_D"], df["hybrid_D"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[3].boxplot([df["mesh2ir_D"], df["rirbox_D"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[3].set_title('D')
    axs[3].set_ylabel('D Error')
    axs[3].legend(handles=[mean_marker])

    # RT60
    # axs[4].bar(model_names, [df_mean["mesh2ir_rt60"], df_mean["rirbox_rt60"], df_mean["hybrid_rt60"]],
    #             yerr=[df_std["mesh2ir_rt60"], df_std["rirbox_rt60"], df_std["hybrid_rt60"]],
    #             color=colors, capsize=20)
    # axs[4].set_title('RT60')
    # axs[4].set_ylabel('Mean Error')
    # axs[4].boxplot([df["mesh2ir_rt60"], df["rirbox_rt60"], df["hybrid_rt60"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[4].boxplot([df["mesh2ir_rt60"], df["rirbox_rt60"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[4].set_title('RT60')
    axs[4].set_ylabel('RT60 Error')
    axs[4].legend(handles=[mean_marker])

    for ax in axs:
        ax.grid(ls="--", alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config['SAVE_PATH']', type=str, default="./models/RIRBOX/RIRBOX_Model2_Finetune_worldly-lion-25.pth",
    #                     help='Path to rirbox model to validate.')
    # args, _ = parser.parse_known_args()

    model_configs = [
        "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MSDist_MLPDEPTH4_1m.json",
        "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MSDist_MLPDEPTH4_dp.json",
        "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MSDist_MLPDEPTH4_1m.json",
        "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MSDist_MLPDEPTH4_dp.json",
        "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MLPDEPTH4_1m.json",
        "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MLPDEPTH4_dp.json",
        "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MLPDEPTH4_1m.json",
        "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MLPDEPTH4_dp.json",
        # "./training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH2.json",
        # "./training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH3.json",
        # "./training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH4.json",
        # "./training/ablation2/rirbox_model3_MRSTFT_EDR_MLPDEPTH4.json",
        # "./training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH2.json",
        # "./training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH3.json",
        # "./training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH4.json",
        # "./training/ablation2/rirbox_model2_MRSTFT_EDR_MLPDEPTH4.json",
    ]
    
    # for model_config in model_configs:
    #     validation_metric_accuracy_mesh2ir_vs_rirbox(model_config=model_config, validation_iterations=2815)
    
    results_csvs = [
        "./validation_results/rirbox_model2_MRSTFT_MSDist_MLPDEPTH4_1m.csv",
        "./validation_results/rirbox_model3_MRSTFT_MSDist_MLPDEPTH4_1m.csv",
        "./validation_results/rirbox_model2_MRSTFT_MSDist_MLPDEPTH4_dp.csv",
        "./validation_results/rirbox_model3_MRSTFT_MSDist_MLPDEPTH4_dp.csv",
        "./validation_results/rirbox_model2_MRSTFT_MLPDEPTH4_1m.csv",
        "./validation_results/rirbox_model3_MRSTFT_MLPDEPTH4_1m.csv",
        "./validation_results/rirbox_model2_MRSTFT_MLPDEPTH4_dp.csv",
        "./validation_results/rirbox_model3_MRSTFT_MLPDEPTH4_dp.csv",
        # "./validation_results/rirbox_model3_MRSTFT_MLPDEPTH2.csv",
        # "./validation_results/rirbox_model3_MRSTFT_MLPDEPTH3.csv",
        # "./validation_results/rirbox_model3_MRSTFT_MLPDEPTH4.csv",
        # "./validation_results/rirbox_model3_MRSTFT_EDR_MLPDEPTH4.csv",
        # "./validation_results/rirbox_model2_MRSTFT_MLPDEPTH2.csv",
        # "./validation_results/rirbox_model2_MRSTFT_MLPDEPTH3.csv",
        # "./validation_results/rirbox_model2_MRSTFT_MLPDEPTH4.csv",
        # "./validation_results/rirbox_model2_MRSTFT_EDR_MLPDEPTH4.csv",
    ]

    for results_csv in results_csvs:
        view_results_metric_accuracy_mesh2ir_vs_rirbox(results_csv=results_csv)

if __name__ == "__main__":
    main()
