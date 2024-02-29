import wandb
import torch
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET
from models.rirbox_models import MeshToShoebox_Model_2, ShoeboxToRIR
from models.utility import load_mesh_net
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
import torch.optim as optim
from tools.pyLiam.LKTimer import LKTimer
from tools.pyLiam.LKMemCheck import LKMemCheck
from tqdm import tqdm
import argparse
from json import load
import time
import gc

############################################ Config ############################################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="./training/rirbox_model2_finetune.json", help='Path to configuration file.')
parser.add_argument('--MLP_DEPTH', type=int, default=2, help='MLP depth')
parser.add_argument('--TOA_SYNC', type=bool, default=True, help='Synchronize TOA')
parser.add_argument('--D', type=bool, default=False, help='add D Loss')
parser.add_argument('--RT60', type=bool, default=False, help='add RT60 Loss')
args, _ = parser.parse_known_args()
with open(args.config, 'r') as file: config = load(file)

RIRBOX_MODEL_ARCHITECTURE = config['RIRBOX_MODEL_ARCHITECTURE']
PRETRAINED_MESHNET = config['PRETRAINED_MESHNET']
TRAIN_MESHNET = config['TRAIN_MESHNET']
ISM_MAX_ORDER = config['ISM_MAX_ORDER']
MLP_DEPTH = 2

LEARNING_RATE = config['LEARNING_RATE']
EPOCHS = config['EPOCHS']
MAX_ITERATIONS = config['MAX_ITERATIONS']
BATCH_SIZE = config['BATCH_SIZE']
DEVICE = config['DEVICE']

TOA_SYNC = args.TOA_SYNC
if args.D: D_LOSS_WEIGHT = 1.0
else: D_LOSS_WEIGHT = 0.0
if args.RT60: RT60_LOSS_WEIGHT = 1.0
else: RT60_LOSS_WEIGHT = 0.0

do_wandb = config['do_wandb']

DATALOADER_NUM_WORKERS = 10

print("PARAMETERS:")
print("    > BATCH_SIZE = ", BATCH_SIZE)
print("    > LEARNING_RATE = ", LEARNING_RATE)
print("    > EPOCHS = ", EPOCHS)
# if device is cuda, then check if cuda is available
if DEVICE == 'cuda':
    if not torch.cuda.is_available():
        DEVICE = 'cpu'
        print("    CUDA not available, using CPU")
print("    > DEVICE = ", DEVICE)
print("    > wandb = ",do_wandb, end="\n\n")

############################################  WanDB ############################################

if do_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RIRBox3",
        
        # track hyperparameters and run metadata
        config={
            "training_dataset": "GWA_3DFRONT",
            "architecture": f"Model {RIRBOX_MODEL_ARCHITECTURE}",
            "pretrained_meshnet": PRETRAINED_MESHNET,
            "ism max order": ISM_MAX_ORDER,
            "MLP depth": MLP_DEPTH,

            "learning rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "max iterations": MAX_ITERATIONS,
            "batch size": BATCH_SIZE,
            "device": DEVICE,

            "TOA sync": TOA_SYNC,
            "D loss": D_LOSS_WEIGHT,
            "RT60 loss": RT60_LOSS_WEIGHT
        }
    )
    print("")

############################################ Inits ############################################

# data
dataset=GWA_3DFRONT_Dataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=DATALOADER_NUM_WORKERS, pin_memory=False,
                        collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
print("")

# models
mesh_net = MESH_NET()
if PRETRAINED_MESHNET:
    mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth")
if RIRBOX_MODEL_ARCHITECTURE == 2:
    mesh_to_shoebox = MeshToShoebox_Model_2(meshnet=mesh_net, MLP_depth=MLP_DEPTH).to(DEVICE)
shoebox_to_rir = ShoeboxToRIR(dataset.sample_rate, max_order=ISM_MAX_ORDER, rir_length=3968).to(DEVICE)#.to('cpu') # This doesn't train, it just computes the RIRs
print("")

# losses
edc=EnergyDecay_Loss(frequency_wise=True,
                     synchronize_TOA=TOA_SYNC,
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
# acm=AcousticianMetrics_Loss(sample_rate=dataset.sample_rate,
#                             synchronize_TOA=True, 
#                             crop_to_same_length=True,
#                             normalize_dp=False,
#                             frequency_wise=False,
#                             normalize_total_energy=False,
#                             pad_to_same_length=False,
#                             MeanAroundMedian_pruning=False).to(DEVICE)

loss_edr, loss_mrstft, loss_c80, loss_D, loss_rt60 = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
print("")

# optimizer
if not TRAIN_MESHNET : mesh_to_shoebox.meshnet.requires_grad = False
optimizer = optim.Adam(mesh_to_shoebox.parameters(), lr=LEARNING_RATE)

# utility
timer = LKTimer(print_time=False)
# lkmc = LKMemCheck(print_during_mem_check=False, print_at_last_mem_check=True, always_reset_lists=True)
# lkmc.disable()
iterations = 0
time_load = 0
time_load_remember = 1
time_start_load = time.time()

# Training
for epoch in range(EPOCHS):
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

        with timer.time("Computing RIR losses"):
            loss_edr = edc(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            loss_mrstft = mrstft(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            # loss_c80, loss_D, loss_rt60, _ = acm(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            # loss_c80 = torch.clamp(loss_c80, max=40)
            # loss_rt60 = torch.clamp(loss_rt60, max=1.0)
            del _
            total_loss = loss_edr * config["EDC_LOSS_WEIGHT"]\
                        + loss_mrstft * config["MRSTFT_LOSS_WEIGHT"]\
                        # + loss_c80 * 0\
                        # + loss_D * D_LOSS_WEIGHT\
                        # + loss_rt60 * RT60_LOSS_WEIGHT
        
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

        if do_wandb:
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

        if iterations == 500:
            break

        time_start_load = time.time()

# Save the model to ./models/RIRBOX
torch.save(mesh_to_shoebox.state_dict(), config['SAVE_PATH'])

print("Training completed")

if do_wandb:
    # finish the wandb run
    wandb.finish()