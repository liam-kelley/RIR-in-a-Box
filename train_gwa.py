import wandb
import torch
import torch.optim as optim
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR
from models.utility import load_mesh_net, load_GAN
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
from training.utility import filter_rir_like_rirbox
from tools.pyLiam.LKTimer import LKTimer
from tools.pyLiam.LKMemCheck import LKMemCheck
from tqdm import tqdm
import argparse
from json import load
import time
import gc
import copy
import os

############################################ Config ############################################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="./training/default/rirbox_model2_finetune_default.json", help='Path to configuration file.')
args, _ = parser.parse_known_args()
with open(args.config, 'r') as file: config = load(file)

DEVICE = config['DEVICE']
if not torch.cuda.is_available(): DEVICE = 'cpu'
DATALOADER_NUM_WORKERS = 10
config["DATALOADER_NUM_WORKERS"] = DATALOADER_NUM_WORKERS
if config["SAVE_PATH"] == "": config["SAVE_PATH"] = "./models/RIRBOX" + args.config[10:]

print("PARAMETERS:")
for key, value in config.items():
    print(f"    > {key} = {value}")
print("")

if config['do_wandb']:
    wandb.init(project="RIRBox3",config=config)
    print("")

# og_config = copy.deepcopy(config)
# config['EDC_LOSS_WEIGHT'] = 0.0
# config['C80_LOSS_WEIGHT'] = 0.0
# config['D_LOSS_WEIGHT'] = 0.0
# config['RT60_LOSS_WEIGHT'] = 0.0

############################################ Inits ############################################

# data
dataset=GWA_3DFRONT_Dataset(csv_file="./datasets/GWA_3DFRONT/gwa_3Dfront_train.csv", rir_std_normalization=False)
dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
                        num_workers=DATALOADER_NUM_WORKERS, pin_memory=False,
                        collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
print("")

# models
mesh_net = MESH_NET()
if config['PRETRAINED_MESHNET']:
    mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth")
mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=config['RIRBOX_MODEL_ARCHITECTURE']).to(DEVICE)
shoebox_to_rir = ShoeboxToRIR(dataset.sample_rate, max_order=config['ISM_MAX_ORDER'], rir_length=3968, start_from_ir_onset=True).to(DEVICE)
print("")

# losses
edc=EnergyDecay_Loss(frequency_wise=True,
                     synchronize_TOA=config['EDC_TOA_SYNC'],
                     normalize_dp=False,
                     normalize_decay_curve=False,
                     deemphasize_early_reflections=False,
                     pad_to_same_length=False,
                     crop_to_same_length=True).to(DEVICE)
mrstft=MRSTFT_Loss(sample_rate=dataset.sample_rate,
                   device=DEVICE,
                   synchronize_TOA=True,
                   deemphasize_early_reflections=False,
                   normalize_dp=False,
                   pad_to_same_length=False,
                   crop_to_same_length=True).to(DEVICE)
acm=AcousticianMetrics_Loss(sample_rate=dataset.sample_rate,
                            synchronize_TOA=True, 
                            crop_to_same_length=True,
                            normalize_dp=False,
                            frequency_wise=False,
                            normalize_total_energy=False,
                            pad_to_same_length=False,
                            MeanAroundMedian_pruning=False).to(DEVICE)

loss_edr, loss_mrstft, loss_c80, loss_D, loss_rt60 = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
print("")

# optimizer
if not config['TRAIN_MESHNET'] : mesh_to_shoebox.meshnet.requires_grad = False
optimizer = optim.Adam(mesh_to_shoebox.parameters(), lr=config['LEARNING_RATE'])

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

        optimizer.zero_grad()

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
        del x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch

        with timer.time("Getting shoebox rir"):
            shoebox_rir_batch, shoebox_origin_batch = shoebox_to_rir(latent_shoebox_batch) # shoebox_rir_batch is a list of tensors (batch_size, TODO rir_lengths(i)) , shoebox_origin_batch is a (batch_size) tensor)

        # Freeing memory
        del latent_shoebox_batch

        # Moving data to device
        label_rir_batch = label_rir_batch.to(DEVICE)
        label_origin_batch = label_origin_batch.to(DEVICE)

        # filter label rir for comparability to rirbox in the losses.
        label_rir_batch = filter_rir_like_rirbox(label_rir_batch)

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
                        + loss_rt60 * config['RT60_LOSS_WEIGHT']
        
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
            optimizer.step()

        if config['do_wandb']:
            wandb.log({"Epoch": epoch+1, "total loss": total_loss.item()})
            for key, value in {"loss_edr":loss_edr,
                               "loss_mrstft":loss_mrstft,
                               "loss_c80":loss_c80,
                               "loss_D":loss_D,
                               "loss_rt60":loss_rt60}.items():
                if isinstance(value, torch.Tensor) : wandb.log({key: value.item()})
                elif isinstance(value, float): wandb.log({key: value})
            wandb.log(timer.get_logs())
            if iterations % DATALOADER_NUM_WORKERS == 0:
                wandb.log({"Loading data": time_load/DATALOADER_NUM_WORKERS})
            total_time = sum([ timer.get_logs()[key] for key in timer.get_logs()]) + time_load_remember
            wandb.log({"Total time": total_time})
        
        # print(iterations)
        if iterations % DATALOADER_NUM_WORKERS == 0:
            # print("Loading data: ", time_load/DATALOADER_NUM_WORKERS)
            time_load_remember = time_load/DATALOADER_NUM_WORKERS
            time_load = 0
        
        del total_loss, losses

        # if iterations == config['MAX_ITERATIONS'] // 2:
        #     config['LEARNING_RATE'] = config['LEARNING_RATE'] / 2
        #     config['EDC_LOSS_WEIGHT'] = og_config['EDC_LOSS_WEIGHT']
        #     config['C80_LOSS_WEIGHT'] = og_config['C80_LOSS_WEIGHT']
        #     config['D_LOSS_WEIGHT'] = og_config['D_LOSS_WEIGHT']
        #     config['RT60_LOSS_WEIGHT'] = og_config['RT60_LOSS_WEIGHT']

        if iterations >= config['MAX_ITERATIONS']:
            break

        time_start_load = time.time()

# Save the model to ./models/RIRBOX
# check if save directory exists
if not os.path.exists(config['SAVE_PATH']):
    os.makedirs(config['SAVE_PATH'])

torch.save(mesh_to_shoebox.state_dict(), config['SAVE_PATH'])

print("Training completed")

if config['do_wandb']:
    # finish the wandb run
    wandb.finish()