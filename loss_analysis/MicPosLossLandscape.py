import os
import torch
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import griddata
path_to_append = os.path.expanduser('~/pfe/RIR-in-a-Box') # Set up path for imports
sys.path.append(path_to_append)
from losses.shoebox_losses import MicSrcConfigurationLoss
from losses.rir_losses import EnergyDecay_Loss, EnergyBins_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
from loss_analysis.experiment_utility import get_loss_experiment_cmap
from models.rirbox_models import ShoeboxToRIR
from tools.pyLiam.LKTimer import LKTimer

### Experiment parameters
ITERATIONS=1000
BATCH_SIZE=4

### Plotting parameters
INTERPOLATE_DATA=True
INTERPOLATION_RESOLUTION=1000

### RIR Generation Parameters
SAMPLE_RATE=16000
MAX_ORDER=15
DEVICE="cuda"

def main():
    with torch.no_grad():
        # Init encoder
        Box2RIR=ShoeboxToRIR(SAMPLE_RATE, MAX_ORDER).to(DEVICE)

        # Init target z batch
        # room_dimension=torch.rand(3)*torch.tensor([7.5,7.5,7.5])+torch.tensor([1.5,1.5,1.5])
        room_dimension=torch.rand(3)*torch.tensor([6,6,6])+torch.tensor([3,3,3])
        target_mic_pos=torch.rand(3)*torch.tensor([0.8,0.8,0.8])+torch.tensor([0.1,0.1,0.1])
        target_source_pos=torch.rand(3)*torch.tensor([0.8,0.8,0.8])+torch.tensor([0.1,0.1,0.1])
        room_dimension=room_dimension.to(DEVICE)
        target_mic_pos=target_mic_pos.to(DEVICE)
        target_source_pos=target_source_pos.to(DEVICE)
        target_absorption=torch.rand(1,device=DEVICE)*0.8+torch.tensor([0.1],device=DEVICE)
        target_z_batch=torch.cat((room_dimension,target_mic_pos,target_source_pos,target_absorption)).unsqueeze(0)
        # Generate target RIR
        target_rir_batch, target_origin_batch = Box2RIR(target_z_batch)
    
        # Init losses
        msconf=MicSrcConfigurationLoss(return_separate_losses=False,
                                    lambdas={"mic": 1,"src": 1,"mic_src_vector": 1,"src_mic_vector": 1,"mic_src_distance": 1}).to(DEVICE)
        EDLoss = EnergyDecay_Loss().to(DEVICE)
        EBLoss = EnergyBins_Loss().to(DEVICE)
        MRSTFTLoss = MRSTFT_Loss().to(DEVICE)
        AMLoss = AcousticianMetrics_Loss().to(DEVICE)

        # Init lists
        mic_pos_list=[]
        msconf_loss_list=[]
        edl_loss_list=[]
        ebl_loss_list=[]
        mrstftl_loss_list=[]
        c80_loss_list=[]
        d_loss_list=[]
        rt60_loss_list=[]
        betas_loss_list=[]

        for _ in tqdm(range(ITERATIONS), desc="Processing", unit="iteration"):
            # Select batch of mic pos
            mic_pos=torch.rand(BATCH_SIZE,3, device=DEVICE)

            # make the problem 2D
            mic_pos[:,2]=target_mic_pos[2]

            # store positions
            for b in range(BATCH_SIZE): mic_pos_list.append(mic_pos[b].cpu())

            # Format data as a z batch
            proposed_z_batch=torch.cat((room_dimension.unsqueeze(0).expand(BATCH_SIZE, -1),
                                        mic_pos,
                                        target_source_pos.unsqueeze(0).expand(BATCH_SIZE, -1),
                                        target_absorption.unsqueeze(0).expand(BATCH_SIZE, -1)), dim=-1)

            # Generate RIR
            proposed_rir_batch, proposed_origin_batch = Box2RIR(proposed_z_batch)

            # Get losses
            for b in range(BATCH_SIZE):
                # Get losses
                msconf_loss = msconf(proposed_z_batch[b].unsqueeze(0),target_z_batch)
                edl_loss = EDLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                ebl_loss = EBLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                mrstftl_loss = MRSTFTLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                c80_loss, d_loss, rt60_loss, betas_loss= AMLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                
                # Store losses
                msconf_loss_list.append(msconf_loss.cpu())
                edl_loss_list.append(edl_loss.cpu())
                ebl_loss_list.append(ebl_loss.cpu())
                mrstftl_loss_list.append(mrstftl_loss.cpu())
                c80_loss_list.append(c80_loss.cpu())
                d_loss_list.append(d_loss.cpu())
                rt60_loss_list.append(rt60_loss.cpu())
                betas_loss_list.append(betas_loss.cpu()) 

        mic_pos_list=torch.stack(mic_pos_list) * room_dimension.unsqueeze(0).expand(ITERATIONS*BATCH_SIZE,-1).cpu()
        msconf_loss_list=torch.stack(msconf_loss_list).flatten()
        edl_loss_list=torch.stack(edl_loss_list).flatten()
        ebl_loss_list=torch.stack(ebl_loss_list).flatten()
        mrstftl_loss_list=torch.stack(mrstftl_loss_list).flatten()
        c80_loss_list=torch.stack(c80_loss_list).flatten()
        d_loss_list=torch.stack(d_loss_list).flatten()
        rt60_loss_list=torch.stack(rt60_loss_list).flatten()
        betas_loss_list=torch.stack(betas_loss_list).flatten()

    ################################################################################
    ################################### PLOTTING ###################################
    ################################################################################

    loss_dict = {"msconf":msconf_loss_list,
                "edecay":edl_loss_list,
                "ebins":ebl_loss_list,
                "mrstft":mrstftl_loss_list,
                "c80":c80_loss_list,
                # "d":d_loss_list,
                "rt60":rt60_loss_list,
                "rt60 betas": betas_loss_list}

    fig, axs = plt.subplots(2,4,figsize=(19,8))
    cmap=get_loss_experiment_cmap()

    if INTERPOLATE_DATA:
        # Create grid coordinates for interpolation
        xi = np.linspace(0, room_dimension[0].item(), INTERPOLATION_RESOLUTION)
        yi = np.linspace(0, room_dimension[1].item(), INTERPOLATION_RESOLUTION)
        xi, yi = np.meshgrid(xi, yi)

    # Experiment info
    axs[1,3].set_title('Experiment info')
    axs[1,3].axis('off')
    axs[1,3].text(0,0.95, f"N Datapoints : {ITERATIONS*BATCH_SIZE}")
    axs[1,3].text(0,0.9,f"Interpolate data : {INTERPOLATE_DATA}")
    if INTERPOLATE_DATA: axs[1,3].text(0.5,0.9,f"Resolution : {INTERPOLATION_RESOLUTION/10} pts/m")
    axs[1,3].text(0,0.85,f"Sample rate : {SAMPLE_RATE}")
    axs[1,3].text(0,0.8,f"RIR Max order : {MAX_ORDER}")
    axs[1,3].text(0,0.65,f"Wall absorption : {int(target_absorption.item()*100)}%")
    axs[1,3].text(0,0.45,f"(Target room absorption = proposed room absorption)")
    axs[1,3].text(0,0.55,f"(Target mic Z = proposed mic Z = {target_mic_pos[2].cpu().numpy()})")
    axs[1,3].text(0,0.5,f"(Target src Z = proposed src Z = {target_source_pos[2].cpu().numpy()})")
    
    target_mic_pos=target_mic_pos*room_dimension
    target_source_pos=target_source_pos*room_dimension

    i=0
    for key, value in loss_dict.items():
        x=i//(4)
        y=i%(4)

        axs[x,y].set_title('{} LL / mic position'.format(key))
        
        if not INTERPOLATE_DATA:
            scatter=axs[x,y].scatter(mic_pos_list[:,0].numpy(),mic_pos_list[:,1].numpy(), c=value.numpy(), cmap=cmap)
            cbar=plt.colorbar(scatter,ax=axs[x,y])
        else:
            zi = griddata((mic_pos_list[:,0].numpy(), mic_pos_list[:,1].numpy()), value.numpy(), (xi, yi), method='linear')
            img = axs[x,y].imshow(zi, extent=[0, room_dimension[0].item(), 0, room_dimension[1].item()], origin='lower', cmap=cmap)
            cbar=plt.colorbar(img,ax=axs[x,y])

        cbar.set_label('{} loss value'.format(key))

        # target room rectangle
        axs[x,y].add_patch(Rectangle((0,0),room_dimension[0].item(),room_dimension[1].item(),
                edgecolor='black',
                facecolor='none',
                lw=5,
                alpha=1.0,
                label="Target room"))

        axs[x,y].scatter(target_mic_pos[0].item(),target_mic_pos[1].item(), c='red', marker='x', label='Target mic')
        axs[x,y].scatter(target_source_pos[0].item(),target_source_pos[1].item(), c='red', marker='o', label='Target source')

        axs[x,y].set_xlim([-1,9])
        axs[x,y].set_ylim([-1,9])

        axs[x,y].set_xlabel('x (m)')
        axs[x,y].set_ylabel('y (m)')

        # axs[x,y].legend(loc='lower left')

        axs[x,y].grid(True, ls=':', alpha=0.5)
        
        i+=1
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()