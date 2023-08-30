import torch.optim as optim
from torch.utils.data import DataLoader
from torch import compile

from complex_room_dataset import RoomDataset, dataset_generation
from RIRMetricsExperiments import make_both_rir_comparable
from RIRMetricsLoss import RIRMetricsLoss, plot_rir_metrics
from encoder import RoomToShoeboxEncoder

import time

import wandb #https://wandb.ai/home

LEARNING_RATE = 3e-3
EPOCHS = 10
BATCH_SIZE = 16
DEVICE='cuda'

RIR_MAX_ORDER = 10
ROOM_N_VERTEX = 4

LAMBDA_D = 1
LAMBDA_C80 = 1
LAMBDA_RT60 = 1
LAMBDA_CENTER_TIME = 1
LAMBDA_MRSTFT = 10
LAMBDA_MS_ENV = 1
MRSTFT_CARE_ABOUT_ORIGIN=True
MS_ENV_FILTERING=True
MS_ENV_CARE_ABOUT_ORIGIN=True

do_wandb=False

if do_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RoomToShoeboxEncoderSecondRuns",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "DNN with 2 hidden layers and Pytorch RIR calculation",
        "dataset": "SimpleRoomDataset",
        "epochs": EPOCHS,
        "batch size": BATCH_SIZE,
        "device": DEVICE,
        "RIR max order": RIR_MAX_ORDER,
        "Room number of vertexes": ROOM_N_VERTEX,
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

dataset=RoomDataset('complex_room_dataset.csv',pre_filtering={'dim':3,'n_vertex':ROOM_N_VERTEX}) # input_0, _ , _ = dataset[0] ; input_size = input_0.shape[0] # ROOM_N_VERTEX * 2 + 3 + 3
dl=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=1)
encoder = RoomToShoeboxEncoder(input_size = ROOM_N_VERTEX*2 + 3 + 3, sample_rate=48000, max_order=RIR_MAX_ORDER).to(DEVICE)
loss_function = RIRMetricsLoss(mrstft_care_about_origin=MRSTFT_CARE_ABOUT_ORIGIN,
                               ms_env_filtering=MS_ENV_FILTERING, ms_env_care_about_origin=MS_ENV_CARE_ABOUT_ORIGIN,
                               lambda_param={'d': LAMBDA_D, 'c80': LAMBDA_C80, 'rt60':LAMBDA_RT60,
                                             'center_time': LAMBDA_CENTER_TIME, 'mrstft': LAMBDA_MRSTFT, 'ms_env': LAMBDA_MS_ENV})
optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

import torch
timer={}
for epoch in range(EPOCHS):
    for input_features_batch, label_rir_batch, label_origin_batch in dl:
        optimizer.zero_grad()

        input_features_batch = input_features_batch.to(DEVICE)
        label_rir_batch = label_rir_batch.to(DEVICE)
        label_origin_batch = label_origin_batch.to(DEVICE)

        print("Getting pytorch rir")
        timer['rir calculation']=time.time()
        shoebox_rir_batch, shoebox_origin_batch = encoder(input_features_batch)
        timer['rir calculation']=time.time()-timer['rir calculation']
        
        print("Making rir comparable")
        timer['make comparable']=time.time()
        shoebox_rir_batch, label_rir_batch = make_both_rir_comparable(shoebox_rir_batch, label_rir_batch)
        timer['make comparable']=time.time()-timer['make comparable']

        print("plotting")
        plot_rir_metrics(shoebox_rir_batch, label_rir_batch, shoebox_origin_batch, label_origin_batch)

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