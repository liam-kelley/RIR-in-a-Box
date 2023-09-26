from pyLiam.LKLogger import LKLogger
import torch
import numpy as np
from compute_rir_v2 import torch_ism
from torch.nn import MSELoss
from edc_loss import EDC_Loss
from RIRMetricsLoss import RIRMetricsLoss
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from RIRMetricsLoss import pad_to_match_size
import multiprocessing
from math import cos, sin, pi
from random import random

DEVICE='cpu'
MAX_ORDER=15
SAMPLE_RATE=48000
SOUND_SPEED=343
EDC_DEEMPHASIZE_EARLY_REFLECTIONS=False

def _get_grid_of_room_dimensions(room_h, min_size, max_size, step):
    '''Get many different room_dimensions in a grid-like fashion. the room height is fixed'''
    grid_of_room_dimensions=[]
    for x in np.arange(min_size[0],max_size[0],step):
        for y in np.arange(min_size[1],max_size[1],step):
            grid_of_room_dimensions.append([x,y,room_h])
    return grid_of_room_dimensions

def _rotate_point_around_point(origin_point,point_to_rotate, angle=-1.15):
    distance=torch.linalg.norm(origin_point-point_to_rotate).item()
    if origin_point.shape[0]==3:
        new_point_pos=torch.Tensor([cos(angle)*distance,sin(angle)*distance,0]) + origin_point
    elif origin_point.shape[0]==2:
        new_point_pos=torch.Tensor([cos(angle)*distance,sin(angle)*distance]) + origin_point
    else: raise(BaseException("points must be 2 long vector or 3 long vector"))
    return new_point_pos

def two_points_inside_rectangle(distance,rectangle):
    i=0
    while i < 1000:
        point1=torch.rand(2)*rectangle[:2]
        point2_temp=point1+torch.Tensor([distance,0])
        j=0
        while j < 10:
            point2=_rotate_point_around_point(point1,point2_temp, angle=random()*2*pi)
            if point2[0]<rectangle[0] and point2[1]<rectangle[1]:
                return point1,point2
            else:
                j+=1
        i+=1
    raise("Could not find two points inside rectangle. Choose your distance better")
     

def rooms_experiment(grid_step=3,
                 min_room_dimensions=None, max_room_dimensions=None,
                 target_room_dimensions=[5,4,3],
                 target_mic_position=[2,2,1.5],
                 target_source_position=[3,2,1.5],
                 print_stuff=True, logger=None):
    '''
    EXPERIMENT
    this method is used to try and show how convex the different metrics are and wether they can be used to do regression or not.
    We try many different room_dimensions in a grid-like fashion and compute their loss against the target room.
    We will then have a different loss value for every room size.
    We then plot these values in a 2D space with colored points for the values of the losses giving us sort of a heat map.
    There is no regression, we're just testing the convexity of the losses.

    mic/source positions = target mic/source positions.
    room height = Target room height.
    
    args:
        grid_step: the step between each room dimension in the grid. the smaller the step, the more points we will have and the more precise the heat map will be.
        min_room_dimensions: the minimum room dimensions to try. if None, it will be set to [max(target_mic_position[i],target_source_position[i]) + 0.2 for i in range(3)]
        max_room_dimensions: the maximum room dimensions to try. if None, it will be set to [i *1.8 for i in self.target_room_dimensions]
        plot: if True, will plot the heat map.
        print_stuff: if True, will print the losses for each room dimension.
        logger: if None, will create a new logger. if not None, will use the given logger.

    returns:
        None
    '''
    # Log inits
    which_losses=['shoebox_mse',
                  "edc_no_early_reflections","edc_early_reflections", 
                  "edr_no_early_reflections","edr_early_reflections", 
                  "d","c80","rt60",
                #   "center_time",
                  "mrstft","mrstft_care_about_origin",
                  "ms_env","ms_env_care_about_origin"]
    if logger==None:
        logger=LKLogger(filename='experiment_room.csv', columns_for_a_new_log_file= [ "max_order",
                                                                                    "room_dimension_x","room_dimension_y","room_dimension_z",
                                                                                    "target_room_dimensions_x","target_room_dimensions_y","target_room_dimensions_z",
                                                                                    "source_position_x","source_position_y","source_position_z",
                                                                                    "mic_position_x","mic_position_y","mic_position_z",
                                                                                    ] + which_losses)

    # Manage default max and min room dimensions (mic/source positions = target mic/source positions.)
    max_room_dimensions=max_room_dimensions if max_room_dimensions!=None else [i *1.8 for i in target_room_dimensions[:2]]
    min_room_dimensions=min_room_dimensions if min_room_dimensions!=None else [max(target_mic_position[i],target_source_position[i]) + 0.2 for i in range(2)]
    
    # init mic and src positions
    mic_position=torch.tensor(target_mic_position, dtype=torch.float32, requires_grad=False, device=DEVICE)
    source_position=torch.tensor(target_source_position, dtype=torch.float32, requires_grad=False, device=DEVICE)
    
    # Get label rir and origin
    og_label_rir = torch_ism(torch.Tensor(target_room_dimensions),mic_position,source_position,SAMPLE_RATE, max_order=MAX_ORDER)
    label_distance = torch.linalg.norm(mic_position-source_position)
    label_origin = 40 + (SAMPLE_RATE*label_distance/SOUND_SPEED)
    label_origin=torch.unsqueeze(label_origin,0)

    # init losses
    shoebox_mse=MSELoss()
    edc_no_early_reflections=EDC_Loss(deemphasize_early_reflections=False,plot=False)
    edc_early_reflections=EDC_Loss(deemphasize_early_reflections=True,plot=False)
    edr_no_early_reflections=EDC_Loss(deemphasize_early_reflections=False,plot=False, edr=True)
    edr_early_reflections=EDC_Loss(deemphasize_early_reflections=True,plot=False, edr=True)
    d=RIRMetricsLoss(lambda_param={'d': 1}, sample_rate=SAMPLE_RATE, print_info=False)
    c80=RIRMetricsLoss(lambda_param={'c80': 1}, sample_rate=SAMPLE_RATE, print_info=False)
    rt60=RIRMetricsLoss(lambda_param={'rt60': 1}, sample_rate=SAMPLE_RATE, print_info=False)
    # center_time=RIRMetricsLoss(lambda_param={'center_time': 1}, sample_rate=SAMPLE_RATE, print_info=False)
    mrstft=RIRMetricsLoss(lambda_param={'mrstft': 1}, sample_rate=SAMPLE_RATE, mrstft_care_about_origin=False, print_info=False)
    mrstft_care_about_origin=RIRMetricsLoss(lambda_param={'mrstft': 1}, sample_rate=SAMPLE_RATE, mrstft_care_about_origin=True, print_info=False)
    multi_filtered_env=RIRMetricsLoss(lambda_param={'ms_env': 1}, sample_rate=SAMPLE_RATE, ms_env_care_about_origin=False, print_info=False)
    multi_filtered_env_care_about_origin=RIRMetricsLoss(lambda_param={'ms_env': 1}, sample_rate=SAMPLE_RATE, ms_env_care_about_origin=True, print_info=False)

    # Print relevant info
    if print_stuff:
        print("target room dimensions ", target_room_dimensions)
        print("mic position ", target_mic_position)
        print("source_position ", target_source_position)
        print("room grid step ", grid_step)
        print('min_room_dimensions', min_room_dimensions)
        print('max_room_dimensions', max_room_dimensions)

        print("\n##################### ROOM LOSSES START #####################\n\n")

    grid_of_room_dimensions=_get_grid_of_room_dimensions(target_room_dimensions[2], min_room_dimensions, max_room_dimensions, grid_step)
    
    for room_dimension_as_list in grid_of_room_dimensions:
        # current subtask hyperparameters
        log_row={"max_order": MAX_ORDER,
            "room_dimension_x": room_dimension_as_list[0],"room_dimension_y": room_dimension_as_list[1],"room_dimension_z": room_dimension_as_list[2],
            "target_room_dimensions_x": target_room_dimensions[0],"target_room_dimensions_y": target_room_dimensions[1], "target_room_dimensions_z": target_room_dimensions[2],
            "source_position_x": target_source_position[0],"source_position_y": target_source_position[1], "source_position_z": target_source_position[2],
            "mic_position_x": target_mic_position[0],"mic_position_y": target_mic_position[1], "mic_position_z": target_mic_position[2]
            }
        # Format parameters to be logged
        for key , value in log_row.items():
            log_row[key]="{:.3f}".format(value)
        
        # Check if this task is already logged
        if not logger.check_if_line_in_log(log_row):

            room_dimensions=torch.tensor(room_dimension_as_list, dtype=torch.float32, requires_grad=False, device=DEVICE)

            # compute shoebox rir with these room dimensions
            # torch_rir = torch_ism(room_dimensions,mic_position,source_position,SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 2 - mic and src = Targets
            # torch_rir = torch_ism(room_dimensions,source_position,mic_position,SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 3 - mic and src switched
            # torch_rir = torch_ism(room_dimensions,mic_position-torch.Tensor([0.5,0.5,0.5]),source_position-torch.Tensor([0.5,0.5,0.5]),SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 4.1 - Translated, same mic-src distance
            # torch_rir = torch_ism(room_dimensions,mic_position,_rotate_point_around_point(mic_position,source_position),SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 4.2 - Rotated (angle -1.15 radians), same mic-src distance
            # torch_rir = torch_ism(room_dimensions,mic_position-torch.Tensor([1,0.5,0]),source_position-torch.Tensor([0.25,0.5,0]),SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 5.1 - Translated , different mic-src distance
            # torch_rir = torch_ism(room_dimensions,mic_position-torch.Tensor([1,0.5,0]),source_position-torch.Tensor([0.25,0.25,0]),SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 5.2 - Translated + rotated , different mic-src distance
            # torch_rir = torch_ism(room_dimensions,torch.rand(3)*room_dimensions,torch.rand(3)*room_dimensions,SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 6.1 - Random Positions
            new_mic_pos, new_src_pos = two_points_inside_rectangle(torch.linalg.norm(mic_position-source_position),room_dimensions) # Experiment 6.2 - Random Positions, correct distance
            torch_rir = torch_ism(room_dimensions,torch.cat((new_mic_pos,torch.unsqueeze(mic_position[2], dim=-1))),torch.cat((new_src_pos,torch.unsqueeze(source_position[2], dim=-1))),SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 6.2 - Random Positions, correct distance

            
            torch_distance = torch.linalg.norm(mic_position-source_position)
            torch_origin = 40 + (SAMPLE_RATE*torch_distance/SOUND_SPEED)
            torch_origin=torch.unsqueeze(torch_origin,0)

            # Get losses
            shoebox_loss=shoebox_mse(room_dimensions, torch.Tensor(target_room_dimensions))
            edc_loss_no_early_reflections = edc_no_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)
            edc_loss_early_reflections = edc_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)
            edr_loss_no_early_reflections = edr_no_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)
            edr_loss_early_reflections = edr_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)

            torch_rir=torch.unsqueeze(torch_rir,0)
            label_rir=torch.unsqueeze(og_label_rir,0)

            d_loss=d(torch_rir,torch_origin,label_rir,label_origin)
            c80_loss=c80(torch_rir,torch_origin,label_rir,label_origin)
            rt60_loss=rt60(torch_rir,torch_origin,label_rir,label_origin)
            # center_time_loss=center_time(torch_rir,torch_origin,label_rir,label_origin)
            
            torch_rir,label_rir = pad_to_match_size(torch_rir,label_rir) # should be implemented directly into the loss as a safeguard

            mrstft_loss=mrstft(torch_rir,torch_origin,label_rir,label_origin)
            mrstft_care_about_origin_loss=mrstft_care_about_origin(torch_rir,torch_origin,label_rir,label_origin)
            multi_filtered_env_loss=multi_filtered_env(torch_rir,torch_origin,label_rir,label_origin)
            multi_filtered_env_care_about_origin_loss=multi_filtered_env_care_about_origin(torch_rir,torch_origin,label_rir,label_origin)

            # Log losses
            log_row["shoebox_mse"]=shoebox_loss.item()
            log_row["edc_no_early_reflections"]=edc_loss_no_early_reflections.item()
            log_row["edc_early_reflections"]=edc_loss_early_reflections.item()
            log_row["edr_no_early_reflections"]=edr_loss_no_early_reflections.item()
            log_row["edr_early_reflections"]=edr_loss_early_reflections.item()
            log_row["d"]=d_loss.item()
            log_row["c80"]=c80_loss.item()
            log_row["rt60"]=rt60_loss.item()
            # log_row["center_time"]=center_time_loss.item()
            log_row["mrstft"]=mrstft_loss.item()
            log_row["mrstft_care_about_origin"]=mrstft_care_about_origin_loss.item()
            log_row["ms_env"]=multi_filtered_env_loss.item()
            log_row["ms_env_care_about_origin"]=multi_filtered_env_care_about_origin_loss.item()

            # Format losses to be logged with not too many decimals
            for loss in which_losses:
                log_row[loss]="{:.6f}".format(log_row[loss])
            
            # print losses and current Room dimension
            if print_stuff:
                print("Target Room dimensions ", log_row['target_room_dimensions_x'],log_row['target_room_dimensions_y'],log_row['target_room_dimensions_z'])
                print("Room dimension : ", log_row['room_dimension_x'],log_row['room_dimension_y'],log_row['room_dimension_z'])
                print("Losses :", end=" ")
                for loss in which_losses:
                    print(loss, log_row[loss], end=" , ")
                print("\n")

            # log task parameters
            logger.add_line_to_log(log_row)
        else:
            print("Already logged this task:", log_row, "skipping")
        
        torch.cuda.empty_cache()

def plot_rooms_experiment(logger=None, losses_to_plot=None):
    if logger==None:
        logger=LKLogger(filename='experiment_room.csv')
    df=logger.get_df()

    # get parameters of last experiment. we assume that the last experiment is the one we want to plot.
    last_row=df.iloc[-1]
    target_room_dimensions=[last_row['target_room_dimensions_x'],last_row['target_room_dimensions_y'],last_row['target_room_dimensions_z']]
    target_mic_position=[last_row['mic_position_x'],last_row['mic_position_y'],last_row['mic_position_z']]
    target_source_position=[last_row['source_position_x'],last_row['source_position_y'],last_row['source_position_z']]

    if losses_to_plot==None:
        losses_to_plot=['shoebox_mse',
                        "edc_no_early_reflections","edc_early_reflections", 
                        "edr_no_early_reflections","edr_early_reflections", 
                        "d","c80","rt60",
                        #   "center_time",
                        "mrstft","mrstft_care_about_origin",
                        "ms_env","ms_env_care_about_origin"]

    # Filter df to only keep the losses we want to plot and to only keep the experiment we are interested in
    df=df[df['target_room_dimensions_x']==target_room_dimensions[0]]
    df=df[df['target_room_dimensions_y']==target_room_dimensions[1]]
    df=df[df['target_room_dimensions_z']==target_room_dimensions[2]]
    df=df[df['mic_position_x']==target_mic_position[0]]
    df=df[df['mic_position_y']==target_mic_position[1]]
    df=df[df['mic_position_z']==target_mic_position[2]]
    df=df[df['source_position_x']==target_source_position[0]]
    df=df[df['source_position_y']==target_source_position[1]]
    df=df[df['source_position_z']==target_source_position[2]]
    df=df[['room_dimension_x','room_dimension_y','room_dimension_z']+losses_to_plot]

    # BEGIN PLOTTING
    number_of_plots_x=3
    number_of_plots_y=4
    fig_rooms, axs_rooms = plt.subplots(number_of_plots_x,number_of_plots_y, figsize=(24, 10))
    # fig_rooms.suptitle("Rooms losses")

    i=0
    for loss in losses_to_plot:
        x=i//(number_of_plots_y)
        y=i%(number_of_plots_y)
        
        axs_rooms[x,y].set_title(loss +' loss based on room size')
        axs_rooms[x,y].set_xlabel('x (m)')
        axs_rooms[x,y].set_ylabel('y (m)')
        axs_rooms[x,y].grid(True, ls=':', alpha=0.5)

        # scatter plot of losses
        scatter=axs_rooms[x,y].scatter(df['room_dimension_x'],df['room_dimension_y'], c=df[loss], alpha=1)
        # lowest loss rectangle
        min_index = df[loss].idxmin()
        min_row = df.loc[min_index]
        axs_rooms[x,y].add_patch(Rectangle((0,0),min_row['room_dimension_x'],min_row['room_dimension_y'],
                edgecolor='blue',
                facecolor='none',
                lw=5,
                alpha=0.5,
                label="Lowest loss room"))
        # target room rectangle
        axs_rooms[x,y].add_patch(Rectangle((0,0),target_room_dimensions[0],target_room_dimensions[1],
                edgecolor='red',
                facecolor='none',
                lw=5,
                alpha=0.5,
                label="Target room"))
        plt.colorbar(scatter,ax=axs_rooms[x,y])
        # Add mic and source positions for reference
        axs_rooms[x,y].scatter(target_source_position[0], target_source_position[1], c="red", marker='x')#, label="Target Source")
        axs_rooms[x,y].scatter(target_mic_position[0], target_mic_position[1], c="red", marker='D')#, label="Target Microphone")
        # rotated_point=_rotate_point_around_point(torch.Tensor(target_mic_position),torch.Tensor(target_source_position)).cpu().tolist() # Use for experiment 4.2
        # axs_rooms[x,y].scatter(rotated_point[0], rotated_point[1], c="cyan", marker='x', label="Source") # Use for experiment 4.2
        # axs_rooms[x,y].scatter(target_source_position[0]-0.25, target_source_position[1]-0.25, c="cyan", marker='x', label="Source") # Use for experiments 4.1, 5.1, 5.2
        # axs_rooms[x,y].scatter(target_mic_position[0], target_mic_position[1], c="cyan", marker='D', label="Microphone") # Use for experiments 4.1, 4.2, 5.1, 5.2
        # axs_rooms[x,y].legend()
        i+=1
    plt.tight_layout()
    plt.show()

def main():
    rooms_experiment(grid_step=0.165,
                 min_room_dimensions=None, max_room_dimensions=None,
                #  min_room_dimensions=[0.5,0.5], max_room_dimensions=None, # Experiment 6.1
                 target_room_dimensions=[5.03,4.02,3.01],
                 target_mic_position=[2,2,1.5],
                 target_source_position=[3,2,1.5],
                 print_stuff=True, logger=None)
    # plot_rooms_experiment(logger=LKLogger(filename='experiment_room_archive_3.csv'), losses_to_plot=None)
    plot_rooms_experiment(logger=None, losses_to_plot=None)


if __name__ == "__main__":
    main()