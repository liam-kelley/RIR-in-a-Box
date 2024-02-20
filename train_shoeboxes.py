import torch.optim as optim
from tqdm import tqdm
import os

from datasets.ShoeboxMeshDataset.mesh_dataset import GraphDataset, plot_mesh_from_edge_index_batch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from models.rirbox_models import ShoeboxToRIR, GraphToShoeboxEncoder

# from torch.nn import MSELoss
from shoebox_loss import Shoebox_Loss # TODO fix this
from edc_loss import EDC_Loss
from RIRMetricsLoss import RIRMetricsLoss
from torch import cat,stack, unsqueeze, Tensor, tensor , isnan, clamp, save as model_save
from torch.cuda import empty_cache

from pyLiam.LKTimer import LKTimer

import wandb #https://wandb.ai/home

import torch.autograd.profiler as profiler

############################################ Config ############################################

LEARNING_RATE = 1e-3
EPOCHS = 25
BATCH_SIZE =  16
DEVICE='cuda'

SHOEBOXES=True

RIR_MAX_ORDER = 15 # dataset max order is 15

EDR_DEEMPHASIZE_EARLY_REFLECTIONS=True # very important. EDR also has significant performance improvements.
MRSTFT_CARE_ABOUT_ORIGIN=True


# OVERALL_LAMBDA_SHOEBOX = 1.0
# LAMBDA_ROOM_SIZE = 1 # 0.125
# LAMBDA_MIC = 5
# LAMBDA_SRC = 5
# LAMBDA_MIC_SRC_VECTOR = 10
# LAMBDA_SRC_MIC_VECTOR = 10
# LAMBDA_MIC_SRC_DISTANCE = 25
# LAMBDA_ABSORPTION = 1
OVERALL_LAMBDA_SHOEBOX = 1
LAMBDA_ROOM_SIZE = 100 # 0.3 # 0.125
LAMBDA_MIC = 5
LAMBDA_SRC = 5
LAMBDA_MIC_SRC_VECTOR = 0
LAMBDA_SRC_MIC_VECTOR = 0
LAMBDA_MIC_SRC_DISTANCE = 25 # Seemingly no effect on convergence.
LAMBDA_ABSORPTION = 100 # 1

OVERALL_LAMBDA_RIR = 1.0
LAMBDA_EDR = 400 # 250
LAMBDA_D = 0 # 8.3
LAMBDA_C80 = 500
LAMBDA_MRSTFT = 6.25

do_wandb=True

plot=False
plot_every=50 
if plot: plot_i=0
else: plot_i=1

print("BATCH_SIZE = ", BATCH_SIZE)
print("LEARNING_RATE = ", LEARNING_RATE)
print("EPOCHS = ", EPOCHS)
print("DEVICE = ", DEVICE)
print("wandb = ",do_wandb)

################################################################################################

if do_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="GraphToRIR11",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "Normal, OracleMic/OracleSrc",
            "dataset": "GraphDataset",
            "epochs": EPOCHS,
            "batch size": BATCH_SIZE,
            "device": DEVICE,
            "RIR max order": RIR_MAX_ORDER,
            "Shoeboxes only": SHOEBOXES,
            "lambda room_size" : LAMBDA_ROOM_SIZE,
            "lambda mic" : LAMBDA_MIC,
            "lambda src" : LAMBDA_SRC,
            "lambda mic_src_vector" : LAMBDA_MIC_SRC_VECTOR,
            "lambda src_mic_vector" : LAMBDA_SRC_MIC_VECTOR,
            "lambda mic_src_distance" : LAMBDA_MIC_SRC_DISTANCE,
            "lambda absorption" : LAMBDA_ABSORPTION,
            "lambda edr" : LAMBDA_EDR,
            "lambda d" : LAMBDA_D,
            "lambda c80" : LAMBDA_C80,
            "lambda mrstft" : LAMBDA_MRSTFT,
            "edr deemphasize early reflections" : EDR_DEEMPHASIZE_EARLY_REFLECTIONS,
            "mrstft care about origin" : MRSTFT_CARE_ABOUT_ORIGIN
        }
    )

# data
dataset=GraphDataset("meshdataset/shoebox_mesh_dataset.csv", filter={})
print(f"Dataset sample_rate = {dataset.sample_rate}")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=GraphDataset.custom_collate_fn)

# models
encoder = GraphToShoeboxEncoder(training=True).to(DEVICE)
BoxToRIR = ShoeboxToRIR(dataset.sample_rate, max_order=RIR_MAX_ORDER).to(DEVICE)#.to('cpu') # This doesn't train, it just computes the RIRs


# Shoebox training loss modules
shoebox=Shoebox_Loss(lambdas={"room_dim":LAMBDA_ROOM_SIZE,"mic":LAMBDA_MIC,"src":LAMBDA_SRC,
                              "mic_src_vector":LAMBDA_MIC_SRC_VECTOR,"src_mic_vector":LAMBDA_SRC_MIC_VECTOR,
                              "mic_src_distance":LAMBDA_MIC_SRC_DISTANCE, "absorption":1},
                              return_separate_losses=True).to(DEVICE)#.to('cpu')
edr=EDC_Loss(deemphasize_early_reflections=EDR_DEEMPHASIZE_EARLY_REFLECTIONS,plot=False, edr=True).to(DEVICE)
rirmetricsloss=RIRMetricsLoss(lambda_param={'d': 1, 'c80': 1, 'mrstft': 1},
                              sample_rate=dataset.sample_rate, mrstft_care_about_origin=False,
                              return_separate_losses=True).to(DEVICE)

# optimizer
optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

# utility
timer = LKTimer(print_time=False)
edr_loss, d_loss, c80_loss, mrstft_loss = tensor([0.0]), tensor([0.0]), tensor([0.0]), tensor([0.0])

# Training
for epoch in range(EPOCHS):
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch,\
        room_dim_batch, mic_pos_batch, source_pos_batch, label_absorption_batch, _ in tqdm(dataloader, desc="Epoch "+str(epoch+1)+ " completion"):

        optimizer.zero_grad()

        x_batch=x_batch.to(DEVICE)
        edge_index_batch=edge_index_batch.to(DEVICE)
        batch_indexes=batch_indexes.to(DEVICE)
        mic_pos_batch=mic_pos_batch.to(DEVICE)
        source_pos_batch=source_pos_batch.to(DEVICE)
        room_dim_batch=room_dim_batch.to(DEVICE)
        label_absorption_batch=label_absorption_batch.to(DEVICE)

        with timer.time("GNN forward pass"):
            shoebox_z_batch = encoder(x_batch, edge_index_batch ,batch_indexes, mic_pos_batch, source_pos_batch)#.to('cpu')
        
        # check for NaNs in encoder output
        if isnan(shoebox_z_batch).any():
            print("Skipping this batch due to NaN shoebox z batch!")
            continue

        if plot and plot_i%plot_every == 0 :
            print("plotting Input mesh")
            plot_mesh_from_edge_index_batch(x_batch , edge_index_batch, batch_indexes, show=False)
        del x_batch, edge_index_batch, batch_indexes

        with timer.time("Computing shoebox losses"):
            sig_mic_pos_batch=mic_pos_batch/room_dim_batch
            sig_source_pos_batch = source_pos_batch/room_dim_batch
            label_shoebox_z_batch=cat((room_dim_batch, sig_mic_pos_batch, sig_source_pos_batch, unsqueeze(label_absorption_batch,dim=1)), dim=1)
            room_dimensions_loss, mic_loss, source_loss, mic_source_vector_loss,\
                source_mic_vector_loss, mic_source_distance_loss, absorption_loss = shoebox(shoebox_z_batch[:,:10], label_shoebox_z_batch[:,:10]) #Shoebox loss only checks dimensions of room
            loss =  OVERALL_LAMBDA_SHOEBOX * LAMBDA_ROOM_SIZE * room_dimensions_loss+\
                    OVERALL_LAMBDA_SHOEBOX * LAMBDA_MIC * mic_loss +\
                    OVERALL_LAMBDA_SHOEBOX * LAMBDA_SRC * source_loss +\
                    OVERALL_LAMBDA_SHOEBOX * LAMBDA_MIC_SRC_VECTOR * mic_source_vector_loss +\
                    OVERALL_LAMBDA_SHOEBOX * LAMBDA_SRC_MIC_VECTOR * source_mic_vector_loss +\
                    OVERALL_LAMBDA_SHOEBOX * LAMBDA_MIC_SRC_DISTANCE * mic_source_distance_loss +\
                    OVERALL_LAMBDA_SHOEBOX * LAMBDA_ABSORPTION * absorption_loss
        
        losses1 = [room_dimensions_loss, mic_loss, source_loss,
                  mic_source_vector_loss, source_mic_vector_loss, absorption_loss]
        for i in range(len(losses1)): losses1[i] = losses1[i].detach().cpu()

        if plot and plot_i%plot_every == 0 :
            print("Plotting intermediate shoeboxes")
            encoder.plot_intermediate_shoeboxes(shoebox_z_batch, label_shoebox_z_batch, show=True)
        del label_shoebox_z_batch,  room_dim_batch, source_pos_batch, mic_pos_batch, label_absorption_batch

        if OVERALL_LAMBDA_RIR > 0.0 :
            with timer.time("Getting pytorch rir"):
                shoebox_rir_batch, shoebox_origin_batch = BoxToRIR(shoebox_z_batch) # shoebox_rir_batch is a list of tensors (batch_size, rir_lengths(i)) , shoebox_origin_batch is a (batch_size) tensor)
            del shoebox_z_batch

            # check for NaNs in rir batch and origin
            if isnan(shoebox_rir_batch).any():
                print("Skipping this batch due to NaN shoebox_rir_batch!")
                continue
            if isnan(shoebox_origin_batch).any():
                print("Skipping this batch due to NaN rirs shoebox_origin_batch!")
                continue

            with timer.time("Computing RIR losses"):
                edr_loss = edr(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch, plot_i)
                rirml = rirmetricsloss(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
                d_loss, c80_loss, mrstft_loss = rirml['d'], rirml['c80'], rirml['mrstft']
                d_loss = clamp(d_loss, max=150)
                loss = loss + OVERALL_LAMBDA_RIR * LAMBDA_EDR * edr_loss
                loss = loss + OVERALL_LAMBDA_RIR * LAMBDA_D *  d_loss.detach() \
                            + OVERALL_LAMBDA_RIR * LAMBDA_C80 * c80_loss \
                            + OVERALL_LAMBDA_RIR * LAMBDA_MRSTFT * mrstft_loss
            
            del shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch
            del rirml
            losses2 = [edr_loss, d_loss, c80_loss, mrstft_loss]
            for i in range(len(losses2)): losses2[i] = losses2[i].detach().cpu()

        # check for NaNs in losses
        if isnan(loss).any():
            print("Skipping this batch due to NaN loss!")
            continue 

        with timer.time("Computing backward"):
            loss.backward()
            optimizer.step()

        if do_wandb:
            wandb.log({"Epoch": epoch+1, "total loss": loss.item()})
            for key, value in  {"Room dim loss": room_dimensions_loss,
                                "Mic loss": mic_loss,
                                "Src loss": source_loss,
                                "Mic src vector loss": mic_source_vector_loss,
                                "Src mic vector loss": source_mic_vector_loss,
                                "Mic src distance loss": mic_source_distance_loss,
                                "Absorption loss": absorption_loss,
                                "EDR MSE loss": edr_loss,
                                "D MSE loss": d_loss,
                                "C80 MSE loss ": c80_loss,
                                "MRSTFT loss": mrstft_loss,
                                "OVERALL_LAMBDA_SHOEBOX": OVERALL_LAMBDA_SHOEBOX,
                                "OVERALL_LAMBDA_RIR": OVERALL_LAMBDA_RIR}.items():
                if isinstance(value, Tensor) : wandb.log({key: value.item()})
                elif isinstance(value, float): wandb.log({key: value})
            wandb.log(timer.get_logs())

        if plot : plot_i+=1

    # Curriculum learning.
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    if epoch == 5:
        LAMBDA_ROOM_SIZE = 0.3
        LAMBDA_ABSORPTION = 1
        LAMBDA_MRSTFT = 13.5
        LEARNING_RATE/=2
    elif epoch == 10:
        LAMBDA_MIC = 1
        LAMBDA_SRC = 1
        LAMBDA_MIC_SRC_DISTANCE = 5
        LAMBDA_EDR = 600
        LAMBDA_C80 = 750
        LEARNING_RATE/=2
    elif epoch == 15:
        LAMBDA_MIC = 0
        LAMBDA_SRC = 0
        LAMBDA_MIC_SRC_DISTANCE = 0
        LAMBDA_EDR = 800
        LAMBDA_C80 = 1000
        LEARNING_RATE/=2

if do_wandb:
    # finish the wandb run
    wandb.finish()