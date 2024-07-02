import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from datasets.ValidationDataset.dataset import HL2_Dataset
from torch.utils.data import DataLoader
from models.utility import load_all_models_for_inference, inference_on_all_models
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss, DRR_Loss, RIR_MSE_Loss
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def metric_accuracy_mesh2ir_vs_rirbox_GWA(model_config : str, validation_csv : str, validation_iterations=0,
                                                 SCALE_MESH2IR_BY_ITS_ESTIMATED_STD = True, # If True, cancels out the std normalization used during mesh2ir's training
                                                 SCALE_MESH2IR_GWA_SCALING_COMPENSATION = True, # If true, cancels out the scaling compensation mesh2ir learned from the GWA dataset during training.
                                                 MESH2IR_USES_LABEL_ORIGIN = False,
                                                 RESPATIALIZE_RIRBOX = False, # This both activates the respaitialization of the rirbox and the start from ir onset
                                                 ISM_MAX_ORDER = 18
                                                 ):
    ''' Validation of the metric accuracy of the MESH2IR and RIRBOX models on the GWA_3DFRONT dataset.'''

    print("Starting metric accuracy validation for model: ", model_config.split("/")[-1].split(".")[0],end="\n\n")

    mesh2ir, rirbox, config, DEVICE = load_all_models_for_inference(model_config,
                                                                    START_FROM_IR_ONSET=RESPATIALIZE_RIRBOX,
                                                                    ISM_MAX_ORDER=ISM_MAX_ORDER)

    # data
    dataset=GWA_3DFRONT_Dataset(csv_file=validation_csv, rir_std_normalization=False, gwa_scaling_compensation=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=10, pin_memory=False,
                            collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
    print("")

    # metrics
    edc=EnergyDecay_Loss(frequency_wise=True,
                            synchronize_TOA=False,
                            pad_to_same_length=True,
                            crop_to_same_length=False).to(DEVICE)
    mrstft=MRSTFT_Loss(sample_rate=dataset.sample_rate,
                        device=DEVICE,
                        synchronize_TOA=False,
                        pad_to_same_length=True,
                        crop_to_same_length=False,
                        hi_q_temporal=True).to(DEVICE)
    acm=AcousticianMetrics_Loss(sample_rate=16000,
                                synchronize_TOA=True,
                                crop_to_same_length=False,
                                pad_to_same_length=True).to(DEVICE)
    drr=DRR_Loss().to(DEVICE)
    rir_mse=RIR_MSE_Loss(synchronize_TOA=False).to(DEVICE)
    mse = torch.nn.MSELoss()

    print("")

    with torch.no_grad():
        my_list = []
        i = 0
        # iterate over the dataset
        for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Metric validation"):

            rir_mesh2ir, rir_rirbox,\
                origin_mesh2ir, origin_rirbox, \
                virtual_shoebox= inference_on_all_models(x_batch, edge_index_batch, batch_indexes,
                                                        mic_pos_batch, src_pos_batch, label_origin_batch,
                                                        mesh2ir, rirbox, DEVICE,
                                                        SCALE_MESH2IR_BY_ITS_ESTIMATED_STD,
                                                        SCALE_MESH2IR_GWA_SCALING_COMPENSATION,
                                                        MESH2IR_USES_LABEL_ORIGIN,
                                                        RESPATIALIZE_RIRBOX)

            label_rir_batch = label_rir_batch.to(DEVICE)
            label_origin_batch = label_origin_batch.to(DEVICE)

            # Compute losses
            loss_mesh2ir_edr = edc(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_mrstft = mrstft(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_c80, loss_mesh2ir_D, loss_mesh2ir_rt60, _ = acm(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_drr = drr(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_ir_onset = mse(origin_mesh2ir,label_origin_batch)
            loss_mesh2ir_rir_mse = rir_mse(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)

            loss_rirbox_edr = edc(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_mrstft = mrstft(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_c80, loss_rirbox_D, loss_rirbox_rt60, _ = acm(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_drr = drr(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_ir_onset = mse(origin_rirbox,label_origin_batch)
            loss_rirbox_rir_mse = rir_mse(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)

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
                            loss_rirbox_rt60.cpu().item(),

                            loss_mesh2ir_drr.cpu().item(),
                            loss_rirbox_drr.cpu().item(),

                            loss_mesh2ir_ir_onset.cpu().item(),
                            loss_rirbox_ir_onset.cpu().item(),

                            loss_mesh2ir_rir_mse.cpu().item(),
                            loss_rirbox_rir_mse.cpu().item()
                            ])

            i += 1
            if i == validation_iterations:
                break

    # Save as dataframe
    df = pd.DataFrame(my_list, columns=["mesh2ir_edr",
                                        "rirbox_edr",
                                        
                                        "mesh2ir_mrstft",
                                        "rirbox_mrstft",
                                        
                                        "mesh2ir_c80", 
                                        "rirbox_c80", 
                                        
                                        "mesh2ir_D", 
                                        "rirbox_D", 
                                        
                                        "mesh2ir_rt60", 
                                        "rirbox_rt60", 
                                        
                                        "mesh2ir_drr",
                                        "rirbox_drr",

                                        "mesh2ir_ir_onset",
                                        "rirbox_ir_onset",

                                        "mesh2ir_rir_mse",
                                        "rirbox_rir_mse"
                                        ])
    df = df.apply(np.sqrt) # removes the square from the MSEs
    save_path = "./validation/results_acc_gwa/" + config['SAVE_PATH'].split("/")[-2] + "/" + config['SAVE_PATH'].split("/")[-1].split(".")[0] + ".csv"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path)

    print("Validation results saved at: ", save_path)


def metric_accuracy_mesh2ir_vs_rirbox_HL2(model_config : str, validation_csv : str, validation_iterations=0,
                                                 SCALE_MESH2IR_BY_ITS_ESTIMATED_STD = True, # If True, cancels out the std normalization used during mesh2ir's training
                                                 SCALE_MESH2IR_GWA_SCALING_COMPENSATION = True, # If true, cancels out the scaling compensation mesh2ir learned from the GWA dataset during training.
                                                 MESH2IR_USES_LABEL_ORIGIN = False,
                                                 RESPATIALIZE_RIRBOX = False, # This both activates the respaitialization of the rirbox and the start from ir onset
                                                 ISM_MAX_ORDER = 18
                                                 ):
    ''' Validation of the metric accuracy of the MESH2IR and RIRBOX models on the HL2 dataset.'''

    print("Starting metric accuracy validation for model: ", model_config.split("/")[-1].split(".")[0],end="\n\n")

    mesh2ir, rirbox, config, DEVICE = load_all_models_for_inference(model_config,
                                                                    START_FROM_IR_ONSET=RESPATIALIZE_RIRBOX,
                                                                    ISM_MAX_ORDER=ISM_MAX_ORDER)

    # data
    dataset=HL2_Dataset(csv_file=validation_csv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=1, pin_memory=False,
                            collate_fn=HL2_Dataset.custom_collate_fn)
    print("")

    # metrics
    edc=EnergyDecay_Loss(frequency_wise=True,
                            synchronize_TOA=False,
                            pad_to_same_length=True,
                            crop_to_same_length=False,
                            normalize_decay_curve=True).to(DEVICE)
    mrstft=MRSTFT_Loss(sample_rate=dataset.sample_rate,
                        device=DEVICE,
                        synchronize_TOA=False,
                        pad_to_same_length=True,
                        crop_to_same_length=False,
                        hi_q_temporal=True).to(DEVICE)
    acm=AcousticianMetrics_Loss(sample_rate=16000,
                                synchronize_TOA=True,
                                crop_to_same_length=False,
                                pad_to_same_length=True).to(DEVICE)
    rir_mse=RIR_MSE_Loss(synchronize_TOA=False).to(DEVICE)
    drr=DRR_Loss().to(DEVICE)
    mse = torch.nn.MSELoss()

    print("")

    with torch.no_grad():
        my_list = []
        i = 0
        # iterate over the dataset
        for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Metric validation"):

            rir_mesh2ir, rir_rirbox,\
                origin_mesh2ir, origin_rirbox, \
                virtual_shoebox= inference_on_all_models(x_batch, edge_index_batch, batch_indexes,
                                                        mic_pos_batch, src_pos_batch, label_origin_batch,
                                                        mesh2ir, rirbox, DEVICE,
                                                        SCALE_MESH2IR_BY_ITS_ESTIMATED_STD,
                                                        SCALE_MESH2IR_GWA_SCALING_COMPENSATION,
                                                        MESH2IR_USES_LABEL_ORIGIN,
                                                        RESPATIALIZE_RIRBOX)

            label_rir_batch = label_rir_batch.to(DEVICE)
            label_origin_batch = label_origin_batch.to(DEVICE)

            # Compute losses
            loss_mesh2ir_edr = edc(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_mrstft = mrstft(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_c80, loss_mesh2ir_D, loss_mesh2ir_rt60, _ = acm(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_drr = drr(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            loss_mesh2ir_ir_onset = mse(origin_mesh2ir,label_origin_batch)
            loss_mesh2ir_rir_mse = rir_mse(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)

            loss_rirbox_edr = edc(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_mrstft = mrstft(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_c80, loss_rirbox_D, loss_rirbox_rt60, _ = acm(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_drr = drr(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            loss_rirbox_ir_onset = mse(origin_rirbox,label_origin_batch)
            loss_rirbox_rir_mse = rir_mse(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)

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
                            loss_rirbox_rt60.cpu().item(),

                            loss_mesh2ir_drr.cpu().item(),
                            loss_rirbox_drr.cpu().item(),

                            loss_mesh2ir_ir_onset.cpu().item(),
                            loss_rirbox_ir_onset.cpu().item(),

                            loss_mesh2ir_rir_mse.cpu().item(),
                            loss_rirbox_rir_mse.cpu().item()
                            ])

            i += 1
            if i == validation_iterations:
                break

    # Save as dataframe
    df = pd.DataFrame(my_list, columns=["mesh2ir_edr",
                                        "rirbox_edr",
                                        
                                        "mesh2ir_mrstft",
                                        "rirbox_mrstft",
                                        
                                        "mesh2ir_c80", 
                                        "rirbox_c80", 
                                        
                                        "mesh2ir_D", 
                                        "rirbox_D", 
                                        
                                        "mesh2ir_rt60", 
                                        "rirbox_rt60", 
                                        
                                        "mesh2ir_drr",
                                        "rirbox_drr",

                                        "mesh2ir_ir_onset",
                                        "rirbox_ir_onset",

                                        "mesh2ir_rir_mse",
                                        "rirbox_rir_mse"
                                        ])
    df = df.apply(np.sqrt) # removes the square from the MSEs
    save_path = "./validation/results_acc_hl2/" + config['SAVE_PATH'].split("/")[-2] + "/" + config['SAVE_PATH'].split("/")[-1].split(".")[0] + ".csv"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path)

    print("Validation results saved at: ", save_path)


def view_results_metric_accuracy_mesh2ir_vs_rirbox(results_csv="validation/results_acc_hl2/*.csv"):
    df = pd.read_csv(results_csv)

    df_mean = df.mean()
    df_std = df.std()

    fig, axs = plt.subplots(1,8, figsize=(16, 4))
    fig.suptitle(f'Metric accuracy validation. MESH2IR vs {results_csv.split("/")[-1].split(".")[0]}')

    # Prepare the data for the box plot
    model_names = ["Baseline", "RIRBOX"]
    colors = ['C0', 'C1', 'C2']

    mean_marker = Line2D([], [], color='w', marker='^', markerfacecolor='green', markersize=10, label='Mean')

    # EDR
    axs[0].boxplot([df["mesh2ir_edr"], df["rirbox_edr"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[0].set_title('EDR')
    axs[0].set_ylabel('EDR Error')
    axs[0].legend(handles=[mean_marker], loc="upper right")

    # MRSTFT
    axs[1].boxplot([df["mesh2ir_mrstft"], df["rirbox_mrstft"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[1].set_title('MRSTFT')
    axs[1].set_ylabel('MRSTFT Error')
    axs[1].legend(handles=[mean_marker], loc="upper right")

    # C80
    axs[2].boxplot([df["mesh2ir_c80"], df["rirbox_c80"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[2].set_title('C80')
    axs[2].set_ylabel('C80 Error')
    axs[2].legend(handles=[mean_marker], loc="upper right")
    # D
    axs[3].boxplot([df["mesh2ir_D"], df["rirbox_D"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[3].set_title('D')
    axs[3].set_ylabel('D Error')
    axs[3].legend(handles=[mean_marker], loc="upper right")

    # RT60
    axs[4].boxplot([df["mesh2ir_rt60"], df["rirbox_rt60"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[4].set_title('RT60')
    axs[4].set_ylabel('RT60 Error')
    axs[4].legend(handles=[mean_marker], loc="upper right")

    # DRR
    axs[5].boxplot([df["mesh2ir_drr"], df["rirbox_drr"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[5].set_title('DRR')
    axs[5].set_ylabel('DRR Error')
    axs[5].legend(handles=[mean_marker], loc="upper right")

    # IR ONSET
    axs[6].boxplot([df["mesh2ir_ir_onset"], df["rirbox_ir_onset"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[6].set_title('IR ONSET')
    axs[6].set_ylabel('IR ONSET Error')
    axs[6].legend(handles=[mean_marker], loc="upper right")

    # Simple MSE for Phase Rebuttal
    axs[7].boxplot([df["mesh2ir_rir_mse"], df["rirbox_rir_mse"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[7].set_title('MSE')
    axs[7].set_ylabel('MSE Error')
    axs[7].legend(handles=[mean_marker], loc="upper right")

    for ax in axs:
        ax.grid(ls="--", alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

def  view_results_metric_accuracy_mesh2ir_vs_rirbox_multiple_models(results_csvs):
    for prefix in ["hl2","gwa"]:

        df_list = []
        # df_std_list = []
        for results_csv in results_csvs:
            df = pd.read_csv("./validation/results_acc_"+ prefix+ "/" + results_csv)
            df_list.append(df)

        # if "mesh2ir_drr" in df.columns:
        #     fig, axs = plt.subplots(1,7, figsize=(10, 4))
        # else:
        #     fig, axs = plt.subplots(1,5, figsize=(10, 4))
            
        fig, axs = plt.subplots(1,5, figsize=(10, 4))
        
        fig.suptitle(f'Metrics accuracy on {prefix.upper()} dataset.', fontsize=20)

        # Prepare the data for the box plot
        model_names = ["M2IR", "RBx1","RBx2"]
        # for results_csv in results_csvs:
        #     model_names.append("RIRBox" + results_csv.split("/")[-1].split(".")[0].split("_")[1][-1])

        # mean_marker = Line2D([], [], color='w', marker='^', markerfacecolor='green', markersize=10, label='Mean')

        # EDR
        means=[df["mesh2ir_edr"]] + [df["rirbox_edr"] for df in df_list]
        axs[0].boxplot(means, labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
        axs[0].set_title('EDR', fontsize=18)
        axs[0].set_ylabel('EDR Error [Absolute]', fontsize=14)
        # axs[0].legend(handles=[mean_marker], loc="upper right")

        # MRSTFT
        means=[df["mesh2ir_mrstft"]] + [df["rirbox_mrstft"] for df in df_list]
        axs[1].boxplot(means, labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
        axs[1].set_title('MRSTFT', fontsize=18)
        axs[1].set_ylabel('MRSTFT Error [Absolute]', fontsize=14)
        # axs[1].legend(handles=[mean_marker], loc="upper right")

        # # C80
        # means=[df["mesh2ir_c80"]] + [df["rirbox_c80"] for df in df_list]
        # axs[2].boxplot(means, labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
        # axs[2].set_title('C80', fontsize=18)
        # axs[2].set_ylabel('C80 Error')
        # axs[2].legend(handles=[mean_marker], loc="upper right")
        # # D
        # means=[df["mesh2ir_D"]] + [df["rirbox_D"] for df in df_list]
        # axs[3].boxplot(means, labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
        # axs[3].set_title('D', fontsize=18)
        # axs[3].set_ylabel('D Error')
        # axs[3].legend(handles=[mean_marker], loc="upper right")

        # RT60
        means=[df["mesh2ir_rt60"]] + [df["rirbox_rt60"] for df in df_list]
        axs[2].boxplot(means, labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
        axs[2].set_title('RT60', fontsize=18)
        axs[2].set_ylabel('RT60 Error [s]', fontsize=14)
        # axs[2].legend(handles=[mean_marker], loc="upper right")

        # if "mesh2ir_drr" in df.columns:
        # DRR
        means=[df["mesh2ir_drr"]] + [df["rirbox_drr"] for df in df_list]
        axs[3].boxplot(means, labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
        axs[3].set_title('DRR', fontsize=18)
        axs[3].set_ylabel('DRR Error [Absolute]', fontsize=14)
        # axs[3].legend(handles=[mean_marker], loc="upper right")

        # IR ONSET
        means=[df["mesh2ir_ir_onset"]] + [df["rirbox_ir_onset"] for df in df_list]
        axs[4].boxplot(means, labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
        axs[4].set_title('IR ONSET', fontsize=18)
        axs[4].set_ylabel('IR ONSET Error [samples]', fontsize=14)
        # axs[4].legend(handles=[mean_marker], loc="upper right")

        for ax in axs:
            ax.grid(ls="--", alpha=0.5, axis='y')
            ax.tick_params(axis='x', labelrotation=45, labelsize=14)
            ax.tick_params(axis='y', labelrotation=45, labelsize=12)

        plt.tight_layout()
        plt.show()
