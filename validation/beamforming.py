import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from datasets.ValidationDataset.dataset import HL2_Dataset
from torch.utils.data import DataLoader
from models.utility import load_all_models_for_inference, inference_on_all_models
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss, DRR_Loss
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

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

    # # data
    # dataset=HL2_Dataset(csv_file=validation_csv)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
    #                         num_workers=1, pin_memory=False,
    #                         collate_fn=HL2_Dataset.custom_collate_fn)
    # print("")

    # # metrics
    # edc=EnergyDecay_Loss(frequency_wise=True,
    #                         synchronize_TOA=False,
    #                         pad_to_same_length=True,
    #                         crop_to_same_length=False,
    #                         normalize_decay_curve=True).to(DEVICE)
    # mrstft=MRSTFT_Loss(sample_rate=dataset.sample_rate,
    #                     device=DEVICE,
    #                     synchronize_TOA=False,
    #                     pad_to_same_length=True,
    #                     crop_to_same_length=False,
    #                     hi_q_temporal=True).to(DEVICE)
    # acm=AcousticianMetrics_Loss(sample_rate=16000,
    #                             synchronize_TOA=True,
    #                             crop_to_same_length=False,
    #                             pad_to_same_length=True).to(DEVICE)
    # drr=DRR_Loss().to(DEVICE)
    # mse = torch.nn.MSELoss()

    # print("")

    # with torch.no_grad():
    #     my_list = []
    #     i = 0
    #     # iterate over the dataset
    #     for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Metric validation"):

    #         rir_mesh2ir, rir_rirbox,\
    #             origin_mesh2ir, origin_rirbox, \
    #             virtual_shoebox= inference_on_all_models(x_batch, edge_index_batch, batch_indexes,
    #                                                     mic_pos_batch, src_pos_batch, label_origin_batch,
    #                                                     mesh2ir, rirbox, DEVICE,
    #                                                     SCALE_MESH2IR_BY_ITS_ESTIMATED_STD,
    #                                                     SCALE_MESH2IR_GWA_SCALING_COMPENSATION,
    #                                                     MESH2IR_USES_LABEL_ORIGIN,
    #                                                     RESPATIALIZE_RIRBOX)

    #         label_rir_batch = label_rir_batch.to(DEVICE)
    #         label_origin_batch = label_origin_batch.to(DEVICE)

    #         # Compute losses
    #         loss_mesh2ir_edr = edc(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
    #         loss_mesh2ir_mrstft = mrstft(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
    #         loss_mesh2ir_c80, loss_mesh2ir_D, loss_mesh2ir_rt60, _ = acm(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
    #         loss_mesh2ir_drr = drr(rir_mesh2ir, origin_mesh2ir, label_rir_batch, label_origin_batch)
            
    #         loss_mesh2ir_ir_onset = mse(origin_mesh2ir,label_origin_batch)

    #         loss_rirbox_edr = edc(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
    #         loss_rirbox_mrstft = mrstft(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
    #         loss_rirbox_c80, loss_rirbox_D, loss_rirbox_rt60, _ = acm(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
    #         loss_rirbox_drr = drr(rir_rirbox, origin_rirbox, label_rir_batch, label_origin_batch)
            
            
    #         loss_rirbox_ir_onset = mse(origin_rirbox,label_origin_batch)

    #         # Append to dataframe
    #         my_list.append([loss_mesh2ir_edr.cpu().item(),
    #                         loss_rirbox_edr.cpu().item(),

    #                         loss_mesh2ir_mrstft.cpu().item(),
    #                         loss_rirbox_mrstft.cpu().item(),

    #                         loss_mesh2ir_c80.cpu().item(),
    #                         loss_rirbox_c80.cpu().item(),

    #                         loss_mesh2ir_D.cpu().item(),
    #                         loss_rirbox_D.cpu().item(),

    #                         loss_mesh2ir_rt60.cpu().item(),
    #                         loss_rirbox_rt60.cpu().item(),

    #                         loss_mesh2ir_drr.cpu().item(),
    #                         loss_rirbox_drr.cpu().item(),

    #                         loss_mesh2ir_ir_onset.cpu().item(),
    #                         loss_rirbox_ir_onset.cpu().item()
    #                         ])

    #         i += 1
    #         if i == validation_iterations:
    #             break

    # # Save as dataframe
    # df = pd.DataFrame(my_list, columns=["mesh2ir_edr",
    #                                     "rirbox_edr",
                                        
    #                                     "mesh2ir_mrstft",
    #                                     "rirbox_mrstft",
                                        
    #                                     "mesh2ir_c80", 
    #                                     "rirbox_c80", 
                                        
    #                                     "mesh2ir_D", 
    #                                     "rirbox_D", 
                                        
    #                                     "mesh2ir_rt60", 
    #                                     "rirbox_rt60", 
                                        
    #                                     "mesh2ir_drr",
    #                                     "rirbox_drr",

    #                                     "mesh2ir_ir_onset",
    #                                     "rirbox_ir_onset"
    #                                     ])
    # df = df.apply(np.sqrt) # removes the square from the MSEs
    # save_path = "./validation/results_acc_hl2/" + config['SAVE_PATH'].split("/")[-2] + "/" + config['SAVE_PATH'].split("/")[-1].split(".")[0] + ".csv"
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path))
    # df.to_csv(save_path)

    # print("Validation results saved at: ", save_path)
