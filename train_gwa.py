import wandb
import torch
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from torch.utils.data import DataLoader
from models.mesh2ir_models import MESH_NET
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR
from models.utility import load_mesh_net
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
import torch.optim as optim
from pyLiam.LKTimer import LKTimer
from tqdm import tqdm
import argparse
from json import load

############################################ Config ############################################

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default="./training/rirbox_model2_finetune.json", help='Path to configuration file.')
args, _ = parser.parse_known_args()
with open(args.config_file, 'r') as file: config = load(file)

RIRBOX_MODEL_ARCHITECTURE = config['RIRBOX_MODEL_ARCHITECTURE']
PRETRAINED_MESHNET = config['PRETRAINED_MESHNET']
TRAIN_MESHNET = config['TRAIN_MESHNET']
LEARNING_RATE = config['LEARNING_RATE']
EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
DEVICE = config['DEVICE']
ISM_MAX_ORDER = config['ISM_MAX_ORDER']
do_wandb = config['do_wandb']

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
        project="RIRBox2",
        
        # track hyperparameters and run metadata
        config={
            "architecture": "Model 2",
            "pretrained_meshnet": PRETRAINED_MESHNET,
            "dataset": "GWA + 3DFRONT",
            "learning rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch size": BATCH_SIZE,
            "device": DEVICE,
            "ism max order": ISM_MAX_ORDER
        }
    )
    print("")

############################################ Inits ############################################

# data
dataset=GWA_3DFRONT_Dataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True,
                        collate_fn=GWA_3DFRONT_Dataset.custom_collate_fn)
print("")

# models
mesh_net = MESH_NET()
if PRETRAINED_MESHNET: mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth")
mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=RIRBOX_MODEL_ARCHITECTURE).to(DEVICE)
shoebox_to_rir = ShoeboxToRIR(dataset.sample_rate, max_order=ISM_MAX_ORDER).to(DEVICE)#.to('cpu') # This doesn't train, it just computes the RIRs
print("")

# losses
edc=EnergyDecay_Loss(frequency_wise=False,
                     synchronize_TOA=True,
                     normalize_dp=False,
                     normalize_decay_curve=True,
                     deemphasize_early_reflections=True,
                     pad_to_same_length=False,
                     crop_to_same_length=True).to(DEVICE)
mrstft=MRSTFT_Loss(sample_rate=16000,
                   device=DEVICE,
                   synchronize_TOA=True,
                   deemphasize_early_reflections=True,
                   normalize_dp=True,
                   pad_to_same_length=False,
                   crop_to_same_length=True).to(DEVICE)
acm=AcousticianMetrics_Loss(sample_rate=16000,
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
if not TRAIN_MESHNET : mesh_to_shoebox.meshnet.requires_grad = False
optimizer = optim.Adam(mesh_to_shoebox.parameters(), lr=LEARNING_RATE)

# utility
timer = LKTimer(print_time=False)

# Training
for epoch in range(EPOCHS):
    for x_batch, edge_index_batch, batch_indexes, label_rir_batch, label_origin_batch, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Epoch "+str(epoch+1)+ " completion"):

        optimizer.zero_grad()

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
            loss_c80, loss_D, loss_rt60, _ = acm(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            del _
            total_loss = loss_edr + loss_mrstft + loss_c80 + loss_D + loss_rt60
        
        # Freeing memory
        del shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch
        losses = [loss_edr, loss_mrstft, loss_c80 , loss_D , loss_rt60]
        for i in range(len(losses)): losses[i] = losses[i].detach().cpu()

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

# Save the model to ./models/RIRBOX
torch.save(mesh_to_shoebox.state_dict(), config['SAVE_PATH'])

print("Training completed")

if do_wandb:
    # finish the wandb run
    wandb.finish()