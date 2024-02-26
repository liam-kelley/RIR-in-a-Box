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
from losses.shoebox_losses import SBoxRoomDimensionsLoss
from losses.rir_losses import EnergyDecay_Loss, EnergyBins_Loss, MRSTFT_Loss, AcousticianMetrics_Loss
from loss_analysis.experiment_utility import get_loss_experiment_cmap
from models.rirbox_models import ShoeboxToRIR
from tools.pyLiam.LKTimer import LKTimer

### Experiment parameters
ITERATIONS=500
BATCH_SIZE=4
##### Experiment parameters : mic-src configuration selection method
RANDOM_MSCONF                   = True
MULTI_RANDOM_MSCONF             = False
FIXED_RELATIVE_MSCONF           = False
TARGET_ABSOLUTE_MSCONF          = False
TARGET_RELATIVE_MSCONF          = False
TARGET_ABSOLUTE_MS_DISTANCE_RANDOM_MSCONF = False
assert sum([RANDOM_MSCONF, MULTI_RANDOM_MSCONF, FIXED_RELATIVE_MSCONF, TARGET_ABSOLUTE_MSCONF, TARGET_RELATIVE_MSCONF, TARGET_ABSOLUTE_MS_DISTANCE_RANDOM_MSCONF]) == 1, "Only one of the mic-src configuration options should be true"
##### Experiment parameters : in case of multi random mic-src configuration selection method : parameters
MULTI_RANDOM_AMOUNT=4
assert MULTI_RANDOM_AMOUNT > 1, "MULTI_RANDOM_AMOUNT must be greater than 1"

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
        target_room_dimension=torch.rand(3)*torch.tensor([7.5,7.5,7.5])+torch.tensor([1.5,1.5,1.5])
        target_mic_pos=torch.rand(3)*torch.tensor([0.8,0.8,0.8])+torch.tensor([0.1,0.1,0.1])
        target_source_pos=torch.rand(3)*torch.tensor([0.8,0.8,0.8])+torch.tensor([0.1,0.1,0.1])
        target_room_dimension=target_room_dimension.to(DEVICE)
        target_mic_pos=target_mic_pos.to(DEVICE)
        target_source_pos=target_source_pos.to(DEVICE)
        target_absorption=torch.rand(1,device=DEVICE)*0.8+torch.tensor([0.1],device=DEVICE)
        target_z_batch=torch.unsqueeze(torch.cat((target_room_dimension,target_mic_pos,target_source_pos,target_absorption)),dim=0)
        if TARGET_ABSOLUTE_MS_DISTANCE_RANDOM_MSCONF:
            target_abs_ms_distance=torch.linalg.norm(target_mic_pos-target_source_pos).cpu()

        # Generate target RIR
        target_rir_batch, target_origin_batch = Box2RIR(target_z_batch)

        # Init losses
        RDLoss = SBoxRoomDimensionsLoss().to(DEVICE)
        EDLoss = EnergyDecay_Loss().to(DEVICE)
        EBLoss = EnergyBins_Loss().to(DEVICE)
        MRSTFTLoss = MRSTFT_Loss().to(DEVICE)
        AMLoss = AcousticianMetrics_Loss().to(DEVICE)

        # Init lists
        room_dimension_list=[]
        rdl_loss_list=[]
        edl_loss_list=[]
        ebl_loss_list=[]
        mrstftl_loss_list=[]
        c80_loss_list=[]
        d_loss_list=[]
        rt60_loss_list=[]
        betas_loss_list=[]

        # Init configuration
        if FIXED_RELATIVE_MSCONF:
            mic_pos=torch.rand(BATCH_SIZE,3,device=DEVICE)
            src_pos=torch.rand(BATCH_SIZE,3,device=DEVICE)
        if TARGET_RELATIVE_MSCONF:
            mic_pos=target_mic_pos.unsqueeze(0).expand(BATCH_SIZE,-1).clone()
            src_pos=target_source_pos.unsqueeze(0).expand(BATCH_SIZE,-1).clone()

        for _ in tqdm(range(ITERATIONS), desc="Processing", unit="iteration"):
            # Select batch of room dimensions, mic pos, and src pos
            if (not TARGET_ABSOLUTE_MSCONF) and (not TARGET_ABSOLUTE_MS_DISTANCE_RANDOM_MSCONF):
                room_dimension=torch.rand(BATCH_SIZE,3)*torch.tensor([8.5,8.5,8.5]).unsqueeze(0).expand(BATCH_SIZE,-1)+torch.tensor([0.5,0.5,0.5]).unsqueeze(0).expand(BATCH_SIZE,-1)
                room_dimension=room_dimension.to(DEVICE)
            elif TARGET_ABSOLUTE_MSCONF :
                min_room_dimension=target_room_dimension*torch.max(target_mic_pos,target_source_pos)
                max_room_dimension_interval=torch.tensor([9.0,9.0,9.0],device=DEVICE)-min_room_dimension
                room_dimension=torch.rand(BATCH_SIZE,3,device=DEVICE)*max_room_dimension_interval.unsqueeze(0).expand(BATCH_SIZE,-1)+min_room_dimension.unsqueeze(0).expand(BATCH_SIZE,-1)
                mic_pos=(target_room_dimension*target_mic_pos).unsqueeze(0).expand(BATCH_SIZE,-1) / room_dimension
                src_pos=(target_room_dimension*target_source_pos).unsqueeze(0).expand(BATCH_SIZE,-1) / room_dimension
            elif TARGET_ABSOLUTE_MS_DISTANCE_RANDOM_MSCONF:
                min_room_dimension=target_room_dimension.cpu()*target_abs_ms_distance
                max_room_dimension_interval=torch.tensor([9.0,9.0,9.0])-min_room_dimension
                room_dimension=[]
                mic_pos=[]
                src_pos=[]
                for b in range(BATCH_SIZE):
                    room_dimension_temp=torch.rand(3)*max_room_dimension_interval+min_room_dimension
                    mic_pos_temp=torch.rand(3)
                    theta=torch.rand(1)*2*torch.pi
                    src_pos_temp=mic_pos_temp+(torch.cos(theta)*torch.tensor([1,0,0])*target_abs_ms_distance/room_dimension_temp)+\
                                              (torch.sin(theta)*torch.tensor([0,1,0])*target_abs_ms_distance/room_dimension_temp)
                    while (not (0.0 < src_pos_temp[0] < 1.0)) or (not (0.0 < src_pos_temp[1] < 1.0)):
                        mic_pos_temp=torch.rand(3)
                        theta=torch.rand(1)*2*torch.pi
                        src_pos_temp=mic_pos_temp+(torch.cos(theta)*torch.tensor([1,0,0])*target_abs_ms_distance/room_dimension_temp)+\
                                                  (torch.sin(theta)*torch.tensor([0,1,0])*target_abs_ms_distance/room_dimension_temp)
                    room_dimension.append(room_dimension_temp)
                    mic_pos.append(mic_pos_temp)
                    src_pos.append(src_pos_temp)
                room_dimension=torch.stack(room_dimension,dim=0).to(DEVICE)
                mic_pos=torch.stack(mic_pos, dim=0).to(DEVICE)
                src_pos=torch.stack(src_pos, dim=0).to(DEVICE)
                del room_dimension_temp, mic_pos_temp, src_pos_temp
            if RANDOM_MSCONF:
                mic_pos=torch.rand(BATCH_SIZE,3,device=DEVICE)
                src_pos=torch.rand(BATCH_SIZE,3,device=DEVICE)
            if MULTI_RANDOM_MSCONF:
                mic_pos=torch.rand(BATCH_SIZE*MULTI_RANDOM_AMOUNT,3,device=DEVICE)
                src_pos=torch.rand(BATCH_SIZE*MULTI_RANDOM_AMOUNT,3,device=DEVICE)

            # make the problem 2D
            room_dimension[:,2]=target_room_dimension[2]
            mic_pos[:,2]=target_mic_pos[2]
            src_pos[:,2]=target_source_pos[2]

            # store positions
            for b in range(BATCH_SIZE): room_dimension_list.append(room_dimension[b])

            # Format data as a z batch
            if not MULTI_RANDOM_MSCONF:
                proposed_z_batch=torch.cat((room_dimension,mic_pos,src_pos,
                                            target_absorption.unsqueeze(0).expand(BATCH_SIZE,-1)),dim=-1)
            else:
                proposed_z_batch=torch.cat((torch.repeat_interleave(room_dimension, MULTI_RANDOM_AMOUNT, dim=0),mic_pos,src_pos,
                                            target_absorption.unsqueeze(0).expand(BATCH_SIZE*MULTI_RANDOM_AMOUNT,-1)),dim=-1)

            # Generate RIR
            proposed_rir_batch, proposed_origin_batch = Box2RIR(proposed_z_batch)

            # Get losses
            for b in range(BATCH_SIZE):
                if not MULTI_RANDOM_MSCONF:
                    rdl_loss = RDLoss(proposed_z_batch[b].unsqueeze(0),target_z_batch)
                    edl_loss = EDLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                    ebl_loss = EBLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                    mrstftl_loss = MRSTFTLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                    c80_loss, d_loss, rt60_loss, betas_loss= AMLoss(proposed_rir_batch[b].unsqueeze(0), proposed_origin_batch[b].unsqueeze(0), target_rir_batch, target_origin_batch)
                else:
                    mra=MULTI_RANDOM_AMOUNT
                    rdl_loss = RDLoss(proposed_z_batch[b*mra:(b+1)*mra],target_z_batch.expand(mra,-1))
                    edl_loss = EDLoss(proposed_rir_batch[b*mra:(b+1)*mra], proposed_origin_batch[b*mra:(b+1)*mra], target_rir_batch.expand(mra,-1), target_origin_batch.expand(mra))
                    ebl_loss = EBLoss(proposed_rir_batch[b*mra:(b+1)*mra], proposed_origin_batch[b*mra:(b+1)*mra], target_rir_batch.expand(mra,-1), target_origin_batch.expand(mra))
                    mrstftl_loss = MRSTFTLoss(proposed_rir_batch[b*mra:(b+1)*mra], proposed_origin_batch[b*mra:(b+1)*mra], target_rir_batch.expand(mra,-1), target_origin_batch.expand(mra))
                    c80_loss, d_loss, rt60_loss, betas_loss = AMLoss(proposed_rir_batch[b*mra:(b+1)*mra], proposed_origin_batch[b*mra:(b+1)*mra], target_rir_batch.expand(mra,-1), target_origin_batch.expand(mra))   

                # Store losses
                rdl_loss_list.append(rdl_loss)
                edl_loss_list.append(edl_loss)
                ebl_loss_list.append(ebl_loss)
                mrstftl_loss_list.append(mrstftl_loss)
                c80_loss_list.append(c80_loss)
                d_loss_list.append(d_loss)
                rt60_loss_list.append(rt60_loss)
                betas_loss_list.append(betas_loss)

        # Convert to torch tensors
        room_dimension_list=torch.stack(room_dimension_list).cpu()
        rdl_loss_list=torch.stack(rdl_loss_list).flatten().cpu()
        edl_loss_list=torch.stack(edl_loss_list).flatten().cpu()
        ebl_loss_list=torch.stack(ebl_loss_list).flatten().cpu()
        mrstftl_loss_list=torch.stack(mrstftl_loss_list).flatten().cpu()
        c80_loss_list=torch.stack(c80_loss_list).flatten().cpu()
        d_loss_list=torch.stack(d_loss_list).flatten().cpu()
        rt60_loss_list=torch.stack(rt60_loss_list).flatten().cpu()
        betas_loss_list=torch.stack(betas_loss_list).flatten().cpu()

    ################################################################################
    ################################### PLOTTING ###################################
    ################################################################################

    loss_dict = {"rdim":rdl_loss_list,
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
        xi = np.linspace(0.5, 9, INTERPOLATION_RESOLUTION)
        yi = np.linspace(0.5, 9, INTERPOLATION_RESOLUTION)
        xi, yi = np.meshgrid(xi, yi)
    
    axs[1,3].set_title('Experiment info')
    axs[1,3].axis('off')
    axs[1,3].text(0,0.95, f"N Datapoints : {ITERATIONS*BATCH_SIZE}")
    axs[1,3].text(0,0.9,f"Interpolate data : {INTERPOLATE_DATA}")
    if INTERPOLATE_DATA: axs[1,3].text(0.5,0.9,f"Resolution : {INTERPOLATION_RESOLUTION/10} pts/m")
    axs[1,3].text(0,0.85,f"Sample rate : {SAMPLE_RATE}")
    axs[1,3].text(0,0.8,f"RIR Max order : {MAX_ORDER}")

    axs[1,3].text(0,0.7, f"Target room dimensions : {target_room_dimension.cpu().numpy()}")
    axs[1,3].text(0,0.65,f"Wall absorption : {int(target_absorption.item()*100)}%")
    axs[1,3].text(0,0.6,f"(Target room Z = proposed room Z)")
    axs[1,3].text(0,0.55,f"(Target mic Z = proposed mic Z = {target_mic_pos[2].cpu().numpy()})")
    axs[1,3].text(0,0.5,f"(Target src Z = proposed src Z = {target_source_pos[2].cpu().numpy()})")
    axs[1,3].text(0,0.45,f"(Target room absorption = proposed room absorption)")

    if RANDOM_MSCONF:
        axs[1,3].text(0,0.35,f"Random mic-src configuration")
        axs[1,3].text(0,0.3,f"(A random msconf is selected for each sample)")
    elif MULTI_RANDOM_MSCONF:
        axs[1,3].text(0,0.35,f"Multi random mic-src configuration")
        axs[1,3].text(0,0.3,f"(Multiple random msconf for each point. Loss values are averaged.)")
        axs[1,3].text(0,0.25,f"Multi random amount : {MULTI_RANDOM_AMOUNT}")
    elif FIXED_RELATIVE_MSCONF:
        axs[1,3].text(0,0.35,f"Fixed relative mic-src configuration")
        axs[1,3].text(0,0.3,f"(A random [0.0,1.0]^3 msconf is selected. Used for all samples.)")
    elif TARGET_ABSOLUTE_MSCONF:
        axs[1,3].text(0,0.35,f"Target absolute mic-src configuration")
        axs[1,3].text(0,0.3,f"(The target's [0.0,+inf]^3 msconf is selected. Used for all samples.)")
    elif TARGET_RELATIVE_MSCONF:
        axs[1,3].text(0,0.35,f"Target relative mic-src configuration")
        axs[1,3].text(0,0.3,f"(The target's [0.0,1.0]^3 msconf is selected. Used for all samples.)")
    elif TARGET_ABSOLUTE_MS_DISTANCE_RANDOM_MSCONF:
        axs[1,3].text(0,0.35,f"Target absolute mic-src distance, random msconf")
        axs[1,3].text(0,0.3,f"(A random msconf is selected for each sample, where...")
        axs[1,3].text(0,0.25,f"... the mic-src distance = the target's mic-src distance.)")

    i=0
    for key, value in loss_dict.items():
        x=i//(4)
        y=i%(4)

        axs[x,y].set_title('{} LL / room dimensions'.format(key))
        
        if not INTERPOLATE_DATA:
            scatter=axs[x,y].scatter(room_dimension_list[:,0].numpy(),room_dimension_list[:,1].numpy(), c=value.numpy(), cmap=cmap)
            cbar=plt.colorbar(scatter,ax=axs[x,y])
        else:
            zi = griddata((room_dimension_list[:,0].numpy(), room_dimension_list[:,1].numpy()), value.numpy(), (xi, yi), method='linear')
            img = axs[x,y].imshow(zi, extent=[0.5, 9, 0.5, 9], origin='lower', cmap=cmap)
            cbar=plt.colorbar(img,ax=axs[x,y])

        cbar.set_label('{} loss value'.format(key))

        # lowest loss rectangle
        if i!=0:
            min_index = np.argmin(value.numpy())
            axs[x,y].add_patch(Rectangle((0,0),room_dimension_list[min_index,0],room_dimension_list[min_index,1],
                    edgecolor='darkblue',
                    facecolor='none',
                    lw=5,
                    alpha=0.5,
                    label="Lowest loss room"))

        # target room rectangle
        axs[x,y].add_patch(Rectangle((0,0),target_room_dimension[0].cpu(),target_room_dimension[1].cpu(),
                edgecolor='red',
                facecolor='none',
                lw=5,
                alpha=0.5,
                label="Target room"))

        axs[x,y].scatter(target_mic_pos[0].item()*target_room_dimension[0].item(),target_mic_pos[1].item()*target_room_dimension[1].item(), marker="x", c="red", label="Target mic")
        axs[x,y].scatter(target_source_pos[0].item()*target_room_dimension[0].item(),target_source_pos[1].item()*target_room_dimension[1].item(), marker="o", c="red", label="Target src")

        axs[x,y].set_xlim([0,10])
        axs[x,y].set_ylim([0,10])

        axs[x,y].set_xlabel('x (m)')
        axs[x,y].set_ylabel('y (m)')

        # axs[x,y].legend(loc='lower left')

        axs[x,y].grid(True, ls=':', alpha=0.5)
        
        i+=1
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()