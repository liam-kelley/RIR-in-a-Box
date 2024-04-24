import wandb
import torch
import torch.optim as optim
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR
from models.utility import load_mesh_net
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
from training.utility import filter_rir_like_rirbox, print_config_parameters
from tools.pyLiam.LKTimer import LKTimer
from tools.pyLiam.LKMemCheck import LKMemCheck
from tqdm import tqdm
import argparse
from json import load
import time
import gc
import copy
import os
from torch.nn import MSELoss


############################################ Config ############################################

# get arparse args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="training/configs/default/rirbox_default.json", help='Path to configuration file.')
parser.add_argument('--nworkers', type=int, default=10)
parser.add_argument('--dowandb', action='store_true', help='Enable the feature')
args, _ = parser.parse_known_args()

# load config
with open(args.config, 'r') as file: config = load(file)

# update config with device, nworkers and correct save path
DEVICE = config['DEVICE'] if torch.cuda.is_available() else 'cpu'
config["DATALOADER_NUM_WORKERS"] = args.nworkers
config["DO_WANDB"] =  args.dowandb
if config["SAVE_PATH"] == "":
    config["SAVE_PATH"] = "./models/RIRBOX/" + args.config.split("/")[-2] + "/" + args.config.split("/")[-1].split('.')[0] + ".pth"

print_config_parameters(config)

if config["DO_WANDB"]:
    wandb.init(project="RIRBoxFinal2",config=config)
    print("")

############################################ Inits ############################################

# data
dataset=GWA_3DFRONT_Dataset(csv_file=config['train_dataset'], rir_length = config['RIR_LENGTH'],
                            rir_std_normalization=False, gwa_scaling_compensation=True)
dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=config["SHUFFLE_DATASET"],
                        num_workers=config["DATALOADER_NUM_WORKERS"], pin_memory=False,
                        collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
print("")

# models
mesh_net = MESH_NET()
if config['PRETRAINED_MESHNET']: mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth")
if not config['TRAIN_MESHNET']: mesh_net = mesh_net.eval()
mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net,
                                model=config['RIRBOX_MODEL_ARCHITECTURE'],
                                MLP_Depth=config["MLP_DEPTH"],
                                hidden_size=config['HIDDEN_LAYER_SIZE'],
                                dropout_p=config['DROPOUT_P'],
                                random_noise=config["RANDOM_NOISE_AFTER_MESHNET"],
                                distance_in_latent_vector=config["DIST_IN_LATENT_VECTOR"]).train().to(DEVICE)
shoebox_to_rir = ShoeboxToRIR(sample_rate=dataset.sample_rate,
                              max_order=config['ISM_MAX_ORDER'],
                              rir_length=config['RIR_LENGTH'],
                              start_from_ir_onset=config['SYNC_IR_ONSET_IN_LOSSES'],
                              normalized_distance=False).eval().to(DEVICE)
print("")

# losses
edc=EnergyDecay_Loss(frequency_wise=True,
                     synchronize_TOA=config['SYNC_IR_ONSET_IN_LOSSES'],
                     normalize_decay_curve=config['NORMALIZE_DECAY_CURVE'],
                     normalize_by_distance=config['NORMALIZE_BY_DIST_IN_LOSSES'],
                     pad_to_same_length=False,
                     crop_to_same_length=True).to(DEVICE)
mrstft=MRSTFT_Loss(sample_rate=dataset.sample_rate,
                   device=DEVICE,
                   synchronize_TOA=config['SYNC_IR_ONSET_IN_LOSSES'],
                   normalize_by_distance=config['NORMALIZE_BY_DIST_IN_LOSSES'],
                   pad_to_same_length=False,
                   crop_to_same_length=True,
                   hi_q_temporal=config['MRSTFT_HI_Q_TEMPORAL']).to(DEVICE)
acm=AcousticianMetrics_Loss(sample_rate=dataset.sample_rate,
                            synchronize_TOA=True, 
                            crop_to_same_length=True,
                            pad_to_same_length=False).to(DEVICE)
mse = MSELoss().to(DEVICE)

loss_edr = torch.tensor([0.0])
loss_mrstft = torch.tensor([0.0])
loss_c80 = torch.tensor([0.0])
loss_D = torch.tensor([0.0])
loss_rt60 = torch.tensor([0.0])
loss_mic_src_distance = torch.tensor([0.0])
print("")

# optimizer
if not config['TRAIN_MESHNET'] : mesh_to_shoebox.meshnet.requires_grad = False
optimizer = optim.Adam(mesh_to_shoebox.parameters(), lr=config['LEARNING_RATE'])
optimizer.zero_grad()
if config['DO_SCHEDULING']:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['EPOCHS']//3, gamma=0.3)

# utility
timer = LKTimer(print_time=False)
# lkmc = LKMemCheck(print_during_mem_check=False, print_at_last_mem_check=True, always_reset_lists=True)
# lkmc.disable()
iterations = 0
time_load = 0
time_load_remember = 1
time_start_load = time.time()

# Training
for epoch in range(config['EPOCHS']):
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Epoch "+str(epoch+1)+ " completion"):
        iterations += 1
        time_end_load = time.time()
        time_load += time_end_load - time_start_load      

        with timer.time("Move data to device"):
            # Moving data to device
            x_batch = x_batch.to(DEVICE)
            edge_index_batch = edge_index_batch.to(DEVICE)
            batch_indexes = batch_indexes.to(DEVICE)
            mic_pos_batch = mic_pos_batch.to(DEVICE)
            src_pos_batch = src_pos_batch.to(DEVICE)

        with timer.time("GNN forward pass"):
            latent_shoebox_batch = mesh_to_shoebox(x_batch, edge_index_batch ,batch_indexes, mic_pos_batch, src_pos_batch)

        # Freeing memory
        del x_batch, edge_index_batch, batch_indexes

        with timer.time("Computing mic src distance loss"):
            _, sbox_mic_position, sbox_src_position, _ = ShoeboxToRIR.extract_shoebox_from_latent_representation(latent_shoebox_batch)
            mic_src_distance_loss = mse(torch.linalg.norm(mic_pos_batch - src_pos_batch, dim=1), torch.linalg.norm(sbox_mic_position-sbox_src_position, dim=1))

        del mic_pos_batch, src_pos_batch

        with timer.time("Getting shoebox rir"):
            shoebox_rir_batch, shoebox_origin_batch = shoebox_to_rir(latent_shoebox_batch) # shoebox_rir_batch (batch_size, rir_length) , shoebox_origin_batch (batch_size)

        # Freeing memory
        del latent_shoebox_batch

        # filter label rir for comparability to rirbox in the losses.
        label_rir_batch = filter_rir_like_rirbox(label_rir_batch)

        # Moving data to device
        label_rir_batch = label_rir_batch.to(DEVICE)
        label_origin_batch = label_origin_batch.to(DEVICE)

        with timer.time("Computing RIR losses"):
            loss_edr = edc(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            loss_mrstft = mrstft(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            loss_c80, loss_D, loss_rt60, _ = acm(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            loss_c80 = torch.clamp(loss_c80, max=40)
            loss_rt60 = torch.clamp(loss_rt60, max=1.0)
            del _
            total_loss = loss_edr * config['EDC_LOSS_WEIGHT']\
                        + loss_mrstft * config['MRSTFT_LOSS_WEIGHT']\
                        + loss_c80 * config['C80_LOSS_WEIGHT']\
                        + loss_D * config['D_LOSS_WEIGHT']\
                        + loss_rt60 * config['RT60_LOSS_WEIGHT']\
                        + mic_src_distance_loss * config['MIC_SRC_DISTANCE_LOSS_WEIGHT']
        
        # Freeing memory
        del shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch

        losses = [loss_edr, loss_mrstft, loss_c80 , loss_D , loss_rt60]
        for i in range(len(losses)): losses[i] = losses[i].detach().cpu()

        # if there are nan values in the latent_shoebox_batch, then skip this iteration
        if torch.isnan(total_loss).any():
            print("NAN values in loss. Skipping this iteration")
            # lkmc.next_iteration()
            continue

        with timer.time("Computing backward"):
            total_loss.backward()
            
        with timer.time("Optimizer step"):
            optimizer.step()
            optimizer.zero_grad()

        if config["DO_WANDB"]:
            wandb.log({"Epoch": epoch+1, "total loss": total_loss.item()})
            for key, value in {"loss_edr":loss_edr,
                               "loss_mrstft":loss_mrstft,
                               "loss_c80":loss_c80,
                               "loss_D":loss_D,
                               "loss_rt60":loss_rt60,
                               "loss_mic_src_distance":mic_src_distance_loss}.items():
                if isinstance(value, torch.Tensor) : wandb.log({key: value.item()})
                elif isinstance(value, float): wandb.log({key: value})
            wandb.log(timer.get_logs())
            if iterations % config["DATALOADER_NUM_WORKERS"] == 0:
                wandb.log({"Loading data": time_load/config["DATALOADER_NUM_WORKERS"]})
            total_time = sum([ timer.get_logs()[key] for key in timer.get_logs()]) + time_load_remember
            wandb.log({"Total time": total_time})
        
        # print(iterations)
        if iterations % config["DATALOADER_NUM_WORKERS"] == 0:
            # print("Loading data: ", time_load/config["DATALOADER_NUM_WORKERS"])
            time_load_remember = time_load/config["DATALOADER_NUM_WORKERS"]
            time_load = 0
        
        del total_loss, losses

        if iterations >= config['MAX_ITERATIONS']:
            break

        time_start_load = time.time()

    if config['DO_SCHEDULING']:
        scheduler.step()

# Save the model to ./models/RIRBOX
# check if save directory exists
if not os.path.exists(os.path.dirname(config['SAVE_PATH'])):
    os.makedirs(os.path.dirname(config['SAVE_PATH']))

torch.save(mesh_to_shoebox.state_dict(), config['SAVE_PATH'])

print("Training completed")

if config["DO_WANDB"]:
    # finish the wandb run
    wandb.finish()