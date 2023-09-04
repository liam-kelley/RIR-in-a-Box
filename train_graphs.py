import torch.optim as optim

from mesh_dataset import GraphDataset
from torch.utils.data import DataLoader
from encoder import ShoeboxToRIR, GraphToShoeboxEncoder
from RIRMetricsExperiments import make_both_rir_comparable
from RIRMetricsLoss import RIRMetricsLoss, plot_rir_metrics

import time

import wandb #https://wandb.ai/home

LEARNING_RATE = 1e-4
EPOCHS = 5
BATCH_SIZE = 16
DEVICE='cuda'

SHOEBOXES=True

RIR_MAX_ORDER = 12

LAMBDA_D = 0
LAMBDA_C80 = 0
LAMBDA_RT60 = 1
LAMBDA_CENTER_TIME = 0
LAMBDA_MRSTFT = 10
LAMBDA_MS_ENV = 1
MRSTFT_CARE_ABOUT_ORIGIN=True
MS_ENV_FILTERING=True
MS_ENV_CARE_ABOUT_ORIGIN=True

do_wandb=True

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
optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

timer={}
for epoch in range(EPOCHS):
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, source_pos_batch, absorption_batch, scattering_batch in dl:
        optimizer.zero_grad()

        x_batch = x_batch.to(DEVICE)
        edge_index_batch = edge_index_batch.to(DEVICE)
        batch_indexes = batch_indexes.to(DEVICE)
        
        # unused for now...
        # mic_pos_batch = mic_pos_batch.to(DEVICE)
        # source_pos_batch = source_pos_batch.to(DEVICE)
        
        print("GNN forward pass")
        timer['GNN forward pass']=time.time()
        shoebox_intermediate_x = encoder(x_batch, edge_index_batch ,batch_indexes)
        timer['GNN forward pass']=time.time()-timer['GNN forward pass']

        print("Getting pytorch rir")
        timer['rir calculation']=time.time()
        shoebox_rir_batch, shoebox_origin_batch = BoxToRIR(shoebox_intermediate_x)
        timer['rir calculation']=time.time()-timer['rir calculation']
        
        # These are too big for gpu... so yeah
        # also its just way faster to do on cpu for some reason...
        shoebox_rir_batch = shoebox_rir_batch.to('cpu')
        shoebox_origin_batch = shoebox_origin_batch.to('cpu')
        label_rir_batch = label_rir_batch.to('cpu')
        label_origin_batch = label_origin_batch.to('cpu')

        print("Making rir comparable")
        timer['make comparable']=time.time()
        shoebox_rir_batch, label_rir_batch = make_both_rir_comparable(shoebox_rir_batch, label_rir_batch)
        timer['make comparable']=time.time()-timer['make comparable']

        print("plotting")
        # plot_rir_metrics(shoebox_rir_batch, label_rir_batch, shoebox_origin_batch, label_origin_batch)

        print("Computing losses")
        timer["compute losses"]=time.time()
        loss = loss_function(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
        timer["compute losses"]=time.time()-timer["compute losses"]

        print("Computing backward")
        timer["compute backward"]=time.time()
        loss.backward()
        optimizer.step()
        timer["compute backward"]=time.time()-timer["compute backward"]

        if do_wandb:
            wandb.log({"Epoch": epoch+1, "total loss": loss})
            wandb.log(loss_function.loss_dict) # log all the individual losses
            wandb.log(timer) # log the times

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

if do_wandb:
    # finish the wandb run
    wandb.finish()