import torch.optim as optim
from tqdm import tqdm
import os

from mesh_dataset import GraphDataset, plot_mesh_from_edge_index_batch
from torch.utils.data import DataLoader
from encoder import ShoeboxToRIR, GraphToShoeboxEncoder
from RIRMetricsExperiments import make_both_rir_comparable
from RIRMetricsLoss import RIRMetricsLoss, plot_rir_metrics
from torch.nn import MSELoss
from torch import cat, save as model_save

from LKTimer import LKTimer

import wandb #https://wandb.ai/home

############################################ Options ############################################

LEARNING_RATE = 1e-3
EPOCHS = 5
BATCH_SIZE = 16
DEVICE='cuda'

SHOEBOXES=True

RIR_MAX_ORDER = 12

LAMBDA_D = 0
LAMBDA_C80 = 0
LAMBDA_RT60 = 1
LAMBDA_CENTER_TIME = 0
LAMBDA_MRSTFT = 1
LAMBDA_MS_ENV = 1
LAMBDA_SHOEBOX = 1
MRSTFT_CARE_ABOUT_ORIGIN=True
MS_ENV_FILTERING=True
MS_ENV_CARE_ABOUT_ORIGIN=True

do_wandb=True

plot=False
plot_every=10
if plot: plot_i=0

save=True
save_every=5
if save: save_i=0
save_path="saved_models/GraphToRIR"


################################################################################################

if do_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="GraphToRIR",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "(Conv/Pool)*3, cat pooled layers, FC*2, Sigmoid",
        "dataset": "GraphDataset1",
        "epochs": EPOCHS,
        "batch size": BATCH_SIZE,
        "device": DEVICE,
        "RIR max order": RIR_MAX_ORDER,
        "Shoeboxes only": SHOEBOXES,
        "lambda d" : LAMBDA_D,
        "lambda c80" : LAMBDA_C80,
        "lambda rt60" : LAMBDA_RT60,
        "lambda center_time" : LAMBDA_CENTER_TIME,
        "lambda mrstft" : LAMBDA_MRSTFT,
        "lambda ms_env" : LAMBDA_MS_ENV,
        "lambda shoebox" : LAMBDA_SHOEBOX,
        "mrstft care_about_origin": MRSTFT_CARE_ABOUT_ORIGIN,
        "ms_env filtering": MS_ENV_FILTERING,
        "ms_env care_about_origin": MS_ENV_CARE_ABOUT_ORIGIN,
        }
    )

dataset=GraphDataset("meshdataset/shoebox_mesh_dataset.csv", filter={})

dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=GraphDataset.custom_collate_fn)
encoder = GraphToShoeboxEncoder().to(DEVICE)
BoxToRIR = ShoeboxToRIR(dataset.sample_rate, max_order=RIR_MAX_ORDER).to(DEVICE)

loss_function = RIRMetricsLoss(mrstft_care_about_origin=MRSTFT_CARE_ABOUT_ORIGIN,
                               ms_env_filtering=MS_ENV_FILTERING, ms_env_care_about_origin=MS_ENV_CARE_ABOUT_ORIGIN,
                               lambda_param={'d': LAMBDA_D, 'c80': LAMBDA_C80, 'rt60':LAMBDA_RT60,
                                             'center_time': LAMBDA_CENTER_TIME, 'mrstft': LAMBDA_MRSTFT, 'ms_env': LAMBDA_MS_ENV})
mse=MSELoss().to(DEVICE)

best_losses = [[float('inf'), "empty"] for _ in range(3)] # Init Used to save best 3 models from all epochs

optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

timer = LKTimer()

for epoch in range(EPOCHS):
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch,\
        room_dim_batch,mic_pos_batch, source_pos_batch, absorption_batch, scattering_batch in tqdm(dl, desc="Epoch "+str(epoch+1)+ " completion"):

        if plot and plot_i%plot_every == 0 : plot_mesh_from_edge_index_batch(x_batch , edge_index_batch, batch_indexes, show=False)

        optimizer.zero_grad()

        x_batch = x_batch.to(DEVICE)
        edge_index_batch = edge_index_batch.to(DEVICE)
        batch_indexes = batch_indexes.to(DEVICE)
        
        with timer.time("GNN forward pass"):
            shoebox_z_batch = encoder(x_batch, edge_index_batch ,batch_indexes)

        if plot and plot_i%plot_every == 0 : print("Plotting intermediate shoeboxes"); encoder.plot_intermediate_shoeboxes(shoebox_z_batch, label_shoebox_z_batch, show=False)

        with timer.time("Getting pytorch rir"):
            shoebox_rir_batch, shoebox_origin_batch = BoxToRIR(shoebox_z_batch)
        
        # These are too big for (my) gpu... so yeah
        # also the next steps are just way faster to do on cpu for some reason...
        with timer.time("Move to cpu"):
            shoebox_rir_batch = shoebox_rir_batch.to('cpu')
            shoebox_origin_batch = shoebox_origin_batch.to('cpu')
            label_rir_batch = label_rir_batch.to('cpu')
            label_origin_batch = label_origin_batch.to('cpu')

        with timer.time("Making rir comparable"):
            shoebox_rir_batch, label_rir_batch = make_both_rir_comparable(shoebox_rir_batch, label_rir_batch)

        if plot and plot_i%plot_every == 0 : print("Plotting RIR metrics"); plot_rir_metrics(shoebox_rir_batch, label_rir_batch, shoebox_origin_batch, label_origin_batch)

        with timer.time("Computing losses"):
            label_shoebox_z_batch=cat((room_dim_batch/10.0, mic_pos_batch/room_dim_batch, source_pos_batch/room_dim_batch), dim=1).to(DEVICE)
            intermediate_shoebox_loss = mse(shoebox_z_batch, label_shoebox_z_batch)
            loss = loss_function(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch) + intermediate_shoebox_loss.cpu() * LAMBDA_SHOEBOX

        with timer.time("Computing backward"):
            loss.backward()
            optimizer.step()

        if do_wandb:
            wandb.log({"Epoch": epoch+1, "total loss": loss})
            wandb.log(loss_function.loss_dict) # log all the individual losses
            wandb.log({"Shoebox MSE loss": intermediate_shoebox_loss})
            wandb.log(timer.get_logs()) # log the times

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