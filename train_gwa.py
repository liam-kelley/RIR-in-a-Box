import wandb
import torch
from datasets.GWA_3DFRONT.dataset import my_dataset
from torch.utils.data import DataLoader
from models.mesh2ir_meshnet import MESH_NET
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR
from models.utility import load_mesh_net
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
import torch.optim as optim
from pyLiam.LKTimer import LKTimer
from tqdm import tqdm

'''
This code isn't functional yet due to the lack of a proper dataset.py file
'''

############################################ Config ############################################

RIRBOX_MODEL_ARCHITECTURE=2
PRETRAINED_MESHNET=True
TRAIN_MESHNET=False

LEARNING_RATE = 1e-3
EPOCHS = 25
BATCH_SIZE =  16
DEVICE='cuda'

ISM_MAX_ORDER = 10 # 15 is better...

do_wandb=False

print("BATCH_SIZE = ", BATCH_SIZE)
print("LEARNING_RATE = ", LEARNING_RATE)
print("EPOCHS = ", EPOCHS)
print("DEVICE = ", DEVICE)
print("wandb = ",do_wandb)

############################################  WanDB ############################################

if do_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RIRBox1",
        
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

############################################ Inits ############################################

# data
dataset=my_dataset("GWA_3DFRONT/my_dataset.csv") # TODO fix this my_dataset
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_dataset.custom_collate_fn)

# models
mesh_net = MESH_NET()
if PRETRAINED_MESHNET: mesh_net = load_mesh_net(mesh_net, "./models/MESH2IR/mesh_net_epoch_175.pth")
mesh_to_shoebox = MeshToShoebox(meshnet=mesh_net, model=RIRBOX_MODEL_ARCHITECTURE).to(DEVICE)
shoebox_to_rir = ShoeboxToRIR(16000, max_order=ISM_MAX_ORDER).to(DEVICE)#.to('cpu') # This doesn't train, it just computes the RIRs

# losses
edc=EnergyDecay_Loss(frequency_wise=True,
                     synchronize_DP=True,
                     normalize_dp=False,
                     normalize_decay_curve=True,
                     deemphasize_early_reflections=True,
                     pad_to_same_length=False,
                     crop_to_same_length=True)
mrstft=MRSTFT_Loss(sample_rate=16000,
                   device=DEVICE,
                   synchronize_DP=True,
                   deemphasize_early_reflections=True,
                   normalize_dp=True,
                   pad_to_same_length=False,
                   crop_to_same_length=True)
acm=AcousticianMetrics_Loss(sample_rate=16000,
                            synchronize_DP=True, 
                            crop_to_same_length=True,
                            normalize_dp=False,
                            frequency_wise=False,
                            normalize_total_energy=False,
                            pad_to_same_length=False,
                            MeanAroundMedian_pruning=False)
loss_edr, loss_mrstft, loss_c80, loss_D, loss_rt60 = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])


# optimizer
if not TRAIN_MESHNET : mesh_to_shoebox.meshnet.requires_grad = False
optimizer = optim.Adam(mesh_to_shoebox.parameters(), lr=LEARNING_RATE)

# utility
timer = LKTimer(print_time=False)

# Training
for epoch in range(EPOCHS):
    for mesh_data, gt_position_data, label_rir_data in tqdm(dataloader, desc="Epoch "+str(epoch+1)+ " completion"):
        x_batch, edge_index_batch, batch_indexes = mesh_data
        label_rir_batch, label_origin_batch = label_rir_data
        mic_pos_batch, source_pos_batch = gt_position_data
        
        optimizer.zero_grad()

        # TODO : TO.DEVICES

        with timer.time("GNN forward pass"):
            latent_shoebox_batch = mesh_to_shoebox(x_batch, edge_index_batch ,batch_indexes, mic_pos_batch, source_pos_batch)
            # latent_shoebox_batch = latent_shoebox_batch.to("cpu")
        # del x_batch, edge_index_batch, batch_indexes

        with timer.time("Getting pytorch rir"):
            shoebox_rir_batch, shoebox_origin_batch = shoebox_to_rir(latent_shoebox_batch) # shoebox_rir_batch is a list of tensors (batch_size, TODO rir_lengths(i)) , shoebox_origin_batch is a (batch_size) tensor)
        del latent_shoebox_batch

        with timer.time("Computing RIR losses"):
            loss_edr = edc(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            loss_mrstft = mrstft(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            loss_c80, loss_D, loss_rt60, _ = acm(shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch)
            del _
            total_loss = loss_edr + loss_mrstft + loss_c80 + loss_D + loss_rt60
        
        # del shoebox_rir_batch, shoebox_origin_batch, label_rir_batch, label_origin_batch
        # del mic_pos_batch, source_pos_batch
        # del gt_position_data, mesh_data, label_rir_data
        # del x_batch, edge_index_batch, batch_indexes

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

if do_wandb:
    # finish the wandb run
    wandb.finish()