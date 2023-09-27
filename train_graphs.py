import torch.optim as optim
from tqdm import tqdm
import os

from mesh_dataset import GraphDataset, plot_mesh_from_edge_index_batch
from torch.utils.data import DataLoader
from encoders import ShoeboxToRIR, GraphToShoeboxEncoder

# from torch.nn import MSELoss
from shoebox_loss import Shoebox_Loss
from edc_loss import EDC_Loss
from RIRMetricsLoss import RIRMetricsLoss
from torch import cat,stack, unsqueeze, Tensor , save as model_save
from torch.cuda import empty_cache

from pyLiam.LKTimer import LKTimer

import wandb #https://wandb.ai/home

############################################ Config ############################################

LEARNING_RATE = 4e-3
EPOCHS = 50
BATCH_SIZE = 32
DEVICE='cuda'

SHOEBOXES=True

RIR_MAX_ORDER = 15 # dataset max order is 15

EDR_DEEMPHASIZE_EARLY_REFLECTIONS=True
MRSTFT_CARE_ABOUT_ORIGIN=False

shoebox_lambda_multiplier = 1.000
LAMBDA_ROOM_SIZE = 0.125
LAMBDA_MIC = 1
LAMBDA_SRC = 1
LAMBDA_MIC_SRC_VECTOR = 1
LAMBDA_SRC_MIC_VECTOR = 1
LAMBDA_ABSORPTION = 1

rir_lambda_multiplier = 1.000
LAMBDA_EDR = 250
LAMBDA_D = 8.3
LAMBDA_C80=500
LAMBDA_MRSTFT=1.25

do_wandb=True

plot=False
plot_every=50
if plot: plot_i=0
else: plot_i=1

save=False
save_every=50
if save: save_i=0
save_path="saved_models/GraphToRIR"


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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=GraphDataset.custom_collate_fn)

# models
encoder = GraphToShoeboxEncoder(training=True).to(DEVICE)
BoxToRIR = ShoeboxToRIR(dataset.sample_rate, max_order=RIR_MAX_ORDER).to(DEVICE)#.to('cpu')

# Shoebox training loss modules
shoebox=Shoebox_Loss(lambdas={"room_dim":LAMBDA_ROOM_SIZE,"mic":LAMBDA_MIC,"src":LAMBDA_SRC,
                              "mic_src_vector":LAMBDA_MIC_SRC_VECTOR,"src_mic_vector":LAMBDA_SRC_MIC_VECTOR,
                              "absorption":1}, return_separate_losses=True).to(DEVICE)#.to('cpu')
edr=EDC_Loss(deemphasize_early_reflections=EDR_DEEMPHASIZE_EARLY_REFLECTIONS,plot=False, edr=True).to(DEVICE)
rirmetricsloss=RIRMetricsLoss(lambda_param={'d': 1, 'c80': 1, 'mrstft': 1}, sample_rate=dataset.sample_rate, mrstft_care_about_origin=False, return_separate_losses=True).to(DEVICE)

# optimizer
optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

# utility
timer = LKTimer(print_time=False)
best_losses = [[float('inf'), "empty"] for _ in range(3)] # Init Used to save best 3 models from all epochs
edr_loss, d_loss, c80_loss, mrstft_loss = Tensor([0.0]), Tensor([0.0]), Tensor([0.0]), Tensor([0.0])

# Training
for epoch in range(EPOCHS):
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch,\
        room_dim_batch, mic_pos_batch, source_pos_batch, label_absorption_batch, _ in tqdm(dataloader, desc="Epoch "+str(epoch+1)+ " completion"):

        optimizer.zero_grad()
        
        with timer.time("GNN forward pass"):
            shoebox_z_batch = encoder(x_batch.to(DEVICE), edge_index_batch.to(DEVICE) ,batch_indexes.to(DEVICE), mic_pos_batch.to(DEVICE), source_pos_batch.to(DEVICE))#.to('cpu')

        with timer.time("Shoebox losses"):
            sig_mic_pos_batch=mic_pos_batch/room_dim_batch
            sig_source_pos_batch = source_pos_batch/room_dim_batch
            label_shoebox_z_batch=cat((room_dim_batch, sig_mic_pos_batch, sig_source_pos_batch, unsqueeze(label_absorption_batch,dim=1)), dim=1).to(DEVICE)

            room_dimensions_loss, mic_loss, source_loss, mic_source_vector_loss,\
                source_mic_vector_loss, absorption_loss = shoebox(shoebox_z_batch[:,:10].to(DEVICE), label_shoebox_z_batch[:,:10].to(DEVICE)) #Shoebox loss only checks dimensions of room
            
            loss =  shoebox_lambda_multiplier * LAMBDA_ROOM_SIZE * room_dimensions_loss+\
                    shoebox_lambda_multiplier * LAMBDA_MIC * mic_loss +\
                    shoebox_lambda_multiplier * LAMBDA_SRC * source_loss +\
                    shoebox_lambda_multiplier * LAMBDA_MIC_SRC_VECTOR * mic_source_vector_loss +\
                    shoebox_lambda_multiplier * LAMBDA_SRC_MIC_VECTOR * source_mic_vector_loss +\
                    shoebox_lambda_multiplier * LAMBDA_ABSORPTION * absorption_loss

        if plot and plot_i%plot_every == 0 :
            print("plotting Input mesh")
            plot_mesh_from_edge_index_batch(x_batch , edge_index_batch, batch_indexes, show=False)
            print("Plotting intermediate shoeboxes")
            encoder.plot_intermediate_shoeboxes(shoebox_z_batch, label_shoebox_z_batch, show=True)

        with timer.time("Getting pytorch rir"):
            if rir_lambda_multiplier > 0.0 :
                shoebox_rir_batch, shoebox_origin_batch = BoxToRIR(shoebox_z_batch) # shoebox_rir_batch is a list of tensors (batch_size, rir_lengths(i)), shoebox_origin_batch is a (batch_size) tensor)

        with timer.time("RIR losses"):
            if rir_lambda_multiplier > 0.0 :
                edr_loss = edr(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch, device=DEVICE, plot_i=plot_i).to(DEVICE)#.to('cpu')
                rirml = rirmetricsloss(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
                d_loss, c80_loss, mrstft_loss = rirml['d'], rirml['c80'], rirml['mrstft']

                loss = loss + rir_lambda_multiplier * LAMBDA_EDR * edr_loss \
                    + rir_lambda_multiplier * LAMBDA_D * d_loss \
                    + rir_lambda_multiplier * LAMBDA_C80 * c80_loss \
                    + rir_lambda_multiplier * LAMBDA_MRSTFT * mrstft_loss

        with timer.time("Computing backward"):
            loss.backward()
            optimizer.step()

        empty_cache()

        # for key, value in {"shoebox": intermediate_shoebox_loss,"edr": edr_loss, "d": d_loss, "c80": c80_loss, "mrstft": mrstft_loss}.items():
        #     if value != None:
        #         print(key, value.item(), end=" , ")
        # print("\nTotal loss :",loss.item())

        """Log gradients and weights of the model."""
        # if do_wandb:
            # for name, param in encoder.named_parameters():
            #     wandb.log({f'gradient_{name}': wandb.Histogram(param.grad.detach().cpu().numpy())})
                # wandb.log({f'weight_{name}': wandb.Histogram(param.data)})

        if do_wandb:
            wandb.log({"Epoch": epoch+1, "total loss": loss.item()})
            for key, value in  {"Room dim loss": room_dimensions_loss,
                                "Mic loss": mic_loss,
                                "Src loss": source_loss,
                                "Mic src vector loss": mic_source_vector_loss,
                                "Src mic vector loss": source_mic_vector_loss,
                                "Absorption loss": absorption_loss,
                                "EDR MSE loss": edr_loss,
                                "D MSE loss": d_loss,
                                "C80 MSE loss ": c80_loss,
                                "MRSTFT loss": mrstft_loss,
                                "shoebox_lambda_multiplier": shoebox_lambda_multiplier,
                                "rir_lambda_multiplier": rir_lambda_multiplier}.items():
                if isinstance(value,Tensor) : wandb.log({key: value.item()})
                elif isinstance(value, float): wandb.log({key: value})
            wandb.log(timer.get_logs())

        if plot : plot_i+=1

        if save:
            if save_i%save_every == 0:
                # Check if current model loss is better than the worst among top 3
                if not any([elem[0] == float('inf') for elem in best_losses]):
                    worst_loss, worst_loss_filename = best_losses[[elem[0] for elem in best_losses].index(max([elem[0] for elem in best_losses]))]
                    print(f"Current worst loss is {worst_loss} from {worst_loss_filename}")
                else:
                    worst_loss, worst_loss_filename = float('inf'), "empty"
                if loss < worst_loss:
                    # Replace worst loss with current loss and save model
                    if worst_loss_filename and os.path.exists(worst_loss_filename): os.remove(worst_loss_filename)
                    # save as one of the best models
                    filename = os.path.join(save_path, f"best_model_loss{loss.item()}.pth")
                    model_save({
                        'epoch': epoch,
                        'model_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, filename)
                    best_losses.remove([worst_loss, worst_loss_filename])
                    best_losses.append([loss, filename])
                    print(f"A new best model with loss {loss.item()} is saved!!")
            save_i+=1

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    if epoch > 7:
        if shoebox_lambda_multiplier > 0.0 : shoebox_lambda_multiplier-=1/(0.5*EPOCHS)
        else: shoebox_lambda_multiplier = 0.0
        if rir_lambda_multiplier < 1.0 : rir_lambda_multiplier += 1/(0.5*EPOCHS)
        else : rir_lambda_multiplier = 1.0

if save:
    # Save the latest model
    model_save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, os.path.join(save_path, f"latest_model_loss{loss.item()}.pth"))

if do_wandb:
    # finish the wandb run
    wandb.finish()