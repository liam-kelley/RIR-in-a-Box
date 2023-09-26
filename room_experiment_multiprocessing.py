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
import torch.multiprocessing as multiprocessing
from pyLiam.LKTimer import LKTimer

from room_experiment import _get_grid_of_room_dimensions, two_points_inside_rectangle

def plot_rooms_experiment(logger=None, losses_to_plot=None):
    if logger==None:
        logger=LKLogger(filename='experiment_room.csv')
    df=logger.get_df()

    # get parameters of last experiment. we assume that the last experiment is the one we want to plot.
    last_row=df.iloc[-1]
    TARGET_ROOM_DIMENSIONS=[last_row['target_room_dimensions_x'],last_row['target_room_dimensions_y'],last_row['target_room_dimensions_z']]
    TARGET_MIC_POSITION=[last_row['mic_position_x'],last_row['mic_position_y'],last_row['mic_position_z']]
    TARGET_SOURCE_POSITION=[last_row['source_position_x'],last_row['source_position_y'],last_row['source_position_z']]

    if losses_to_plot==None:
        losses_to_plot=["shoebox_mse","edc_no_early_reflections","edc_early_reflections",
                  "d","c80","rt60","center_time","mrstft","mrstft_care_about_origin",
                  "ms_env","ms_env_care_about_origin"]

    # Filter df to only keep the losses we want to plot and to only keep the experiment we are interested in
    df=df[df['target_room_dimensions_x']==TARGET_ROOM_DIMENSIONS[0]]
    df=df[df['target_room_dimensions_y']==TARGET_ROOM_DIMENSIONS[1]]
    df=df[df['target_room_dimensions_z']==TARGET_ROOM_DIMENSIONS[2]]
    df=df[df['mic_position_x']==TARGET_MIC_POSITION[0]]
    df=df[df['mic_position_y']==TARGET_MIC_POSITION[1]]
    df=df[df['mic_position_z']==TARGET_MIC_POSITION[2]]
    df=df[df['source_position_x']==TARGET_SOURCE_POSITION[0]]
    df=df[df['source_position_y']==TARGET_SOURCE_POSITION[1]]
    df=df[df['source_position_z']==TARGET_SOURCE_POSITION[2]]
    df=df[['room_dimension_x','room_dimension_y','room_dimension_z']+losses_to_plot]

    # BEGIN PLOTTING
    number_of_plots_x=4
    number_of_plots_y=6
    fig_rooms, axs_rooms = plt.subplots(number_of_plots_x,number_of_plots_y, figsize=(28, 10))
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
        axs_rooms[x,y].add_patch(Rectangle((0,0),TARGET_ROOM_DIMENSIONS[0],TARGET_ROOM_DIMENSIONS[1],
                edgecolor='red',
                facecolor='none',
                lw=5,
                alpha=0.5,
                label="Target room"))
        plt.colorbar(scatter,ax=axs_rooms[x,y])
        # Add mic and source positions for reference
        axs_rooms[x,y].scatter(TARGET_SOURCE_POSITION[0], TARGET_SOURCE_POSITION[1], c="red", marker='x', label="Target Source")
        axs_rooms[x,y].scatter(TARGET_MIC_POSITION[0], TARGET_MIC_POSITION[1], c="red", marker='D', label="Target Microphone")
        # axs_rooms[x,y].scatter(TARGET_SOURCE_POSITION[0]-0.5, TARGET_SOURCE_POSITION[1]-0.5, c="cyan", marker='x', label="Source")
        # axs_rooms[x,y].scatter(TARGET_MIC_POSITION[0]-0.5, TARGET_MIC_POSITION[1]-0.5, c="cyan", marker='D', label="Microphone")
        # axs_rooms[x,y].legend()
        i+=1
    plt.tight_layout()
    plt.show()

def _multiprocessed_loss_computations(args):
    room_dimensions, og_label_rir, label_origin , loss_list= args # Experiment 7.1 Random positions
    # room_dimensions, og_label_rir, label_origin , loss_list, distance = args # Experiment 7.2 Random positions, correct distance

    edc_no_early_reflections,edc_early_reflections,d,c80,rt60, mrstft, mrstft_care_about_origin, multi_filtered_env, multi_filtered_env_care_about_origin, edr_no_early_reflections, edr_early_reflections = loss_list

    torch_rir = torch_ism(room_dimensions, room_dimensions * torch.Tensor(np.random.uniform(size=3)), room_dimensions * torch.Tensor(np.random.uniform(size=3)), SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 7.1 Random positions
    # new_mic_pos, new_src_pos = two_points_inside_rectangle(distance,room_dimensions) # Experiment 7.2 - Random Positions, correct distance
    # torch_rir = torch_ism(room_dimensions,torch.cat((new_mic_pos,torch.unsqueeze(mic_position[2], dim=-1))),torch.cat((new_src_pos,torch.unsqueeze(source_position[2], dim=-1))),SAMPLE_RATE, max_order=MAX_ORDER) # Experiment 7.2 - Random Positions, correct distance
    
    torch_distance = torch.linalg.norm(mic_position - source_position)
    torch_origin = 40 + (SAMPLE_RATE * torch_distance / SOUND_SPEED)
    torch_origin = torch.unsqueeze(torch_origin, 0)

    edc_early_reflections_loss = edc_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)
    edc_no_early_reflections_loss = edc_no_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)
    edr_early_reflections_loss = edr_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)
    edr_no_early_reflections_loss = edr_no_early_reflections([torch_rir], torch_origin, [og_label_rir], label_origin)

    torch_rir = torch.unsqueeze(torch_rir, 0)
    label_rir = torch.unsqueeze(og_label_rir, 0)

    d_loss = d(torch_rir, torch_origin, label_rir, label_origin)
    c80_loss = c80(torch_rir, torch_origin, label_rir, label_origin)
    rt60_loss = rt60(torch_rir, torch_origin, label_rir, label_origin)

    torch_rir,label_rir = pad_to_match_size(torch_rir,label_rir) # should be implemented directly into the loss as a safeguard with a warning

    mrstft_loss=mrstft(torch_rir,torch_origin,label_rir,label_origin)
    mrstft_care_about_origin_loss=mrstft_care_about_origin(torch_rir,torch_origin,label_rir,label_origin)
    multi_filtered_env_loss=multi_filtered_env(torch_rir,torch_origin,label_rir,label_origin)
    multi_filtered_env_care_about_origin_loss=multi_filtered_env_care_about_origin(torch_rir,torch_origin,label_rir,label_origin)

    return edc_early_reflections_loss.item(), edc_no_early_reflections_loss.item(), d_loss.item(), c80_loss.item(), rt60_loss.item(),\
            mrstft_loss.item(), mrstft_care_about_origin_loss.item(), multi_filtered_env_loss.item(), multi_filtered_env_care_about_origin_loss.item(),\
            edr_early_reflections_loss.item(), edr_no_early_reflections_loss.item()

if __name__ == "__main__":
    '''EXPERIMENT
    this method is used to try and show how convex the different metrics are and wether they can be used to do regression or not.
    We try many different room_dimensions in a grid-like fashion and compute their loss against the target room.
    We will then have a different loss value for every room size.
    We then plot these values in a 2D space with colored points for the values of the losses giving us sort of a heat map.
    There is no regression, we're just testing the convexity of the losses.

    mic/source positions = target mic/source positions.
    room height = Target room height.'''

    # losses_to_plot=['shoebox_mse',
    #               "edc_no_early_reflections_mean","edc_no_early_reflections_min",
    #               "edc_early_reflections_mean","edc_early_reflections_min",
    #               "d_mean","d_min","c80_mean","c80_min","rt60_mean","rt60_min",
    #               # "center_time",
    #               "mrstft_mean", "mrstft_min",
    #               "mrstft_care_about_origin_mean", "mrstft_care_about_origin_min",
    #               "ms_env_mean", "ms_env_min",
    #               "ms_env_care_about_origin_mean", "ms_env_care_about_origin_min",
    #               "edr_no_early_reflections_mean","edr_no_early_reflections_min",
    #               "edr_early_reflections_mean","edr_early_reflections_min"]
    # plot_rooms_experiment(losses_to_plot=losses_to_plot)

    # Experiment Hyperparameters
    DEVICE='cpu'  # As of right now, multiprocessing does not work with cuda AT ALL literraly had to restart my computer. Oops!
    MAX_ORDER=15
    SAMPLE_RATE=48000
    SOUND_SPEED=343
    EDC_DEEMPHASIZE_EARLY_REFLECTIONS=False
    GRID_STEP=0.163*2
    MIN_ROOM_DIMENSIONS=[0.5, 0.5]
    MAX_ROOM_DIMENSIONS=None
    TARGET_ROOM_DIMENSIONS=[5.03,4.02,3.01]
    TARGET_MIC_POSITION=[2,2,1.5]
    TARGET_SOURCE_POSITION=[3,2,1.5]
    PRINT_STUFF=True

    # Manage default max and min room dimensions (mic/source positions = target mic/source positions.)
    MAX_ROOM_DIMENSIONS=MAX_ROOM_DIMENSIONS if MAX_ROOM_DIMENSIONS!=None else [i *1.8 for i in TARGET_ROOM_DIMENSIONS[:2]]
    MIN_ROOM_DIMENSIONS=MIN_ROOM_DIMENSIONS if MIN_ROOM_DIMENSIONS!=None else [max(TARGET_MIC_POSITION[i],TARGET_SOURCE_POSITION[i]) + 0.2 for i in range(2)]

    # Init stuff
    torch.set_num_threads(1)
    # torch.multiprocessing.set_start_method('spawn') # This should be used for cuda but it doesn't work so don't use it
    timer=LKTimer(print_time=True)
    which_losses=['shoebox_mse',
                  "edc_no_early_reflections_mean","edc_no_early_reflections_min",
                  "edc_early_reflections_mean","edc_early_reflections_min",
                  "d_mean","d_min","c80_mean","c80_min","rt60_mean","rt60_min",
                  # "center_time",
                  "mrstft_mean", "mrstft_min",
                  "mrstft_care_about_origin_mean", "mrstft_care_about_origin_min",
                  "ms_env_mean", "ms_env_min",
                  "ms_env_care_about_origin_mean", "ms_env_care_about_origin_min",
                  "edr_no_early_reflections_mean","edr_no_early_reflections_min",
                  "edr_early_reflections_mean","edr_early_reflections_min"]
    logger=LKLogger(filename='experiment_room.csv', columns_for_a_new_log_file= [ "max_order",
                                                                                "room_dimension_x","room_dimension_y","room_dimension_z",
                                                                                "target_room_dimensions_x","target_room_dimensions_y","target_room_dimensions_z",
                                                                                "source_position_x","source_position_y","source_position_z",
                                                                                "mic_position_x","mic_position_y","mic_position_z",
                                                                                ] + which_losses)

    # init losses
    shoebox_mse=MSELoss().to(DEVICE)
    edc_no_early_reflections=EDC_Loss(deemphasize_early_reflections=False,plot=False).to(DEVICE)
    edc_early_reflections=EDC_Loss(deemphasize_early_reflections=True,plot=False).to(DEVICE)
    edr_no_early_reflections=EDC_Loss(deemphasize_early_reflections=False,plot=False, edr=True)
    edr_early_reflections=EDC_Loss(deemphasize_early_reflections=True,plot=False, edr=True)
    d=RIRMetricsLoss(lambda_param={'d': 1}, sample_rate=SAMPLE_RATE, print_info=False).to(DEVICE)
    c80=RIRMetricsLoss(lambda_param={'c80': 1}, sample_rate=SAMPLE_RATE, print_info=False).to(DEVICE)
    rt60=RIRMetricsLoss(lambda_param={'rt60': 1}, sample_rate=SAMPLE_RATE, print_info=False).to(DEVICE)
        # center_time=RIRMetricsLoss(lambda_param={'center_time': 1}, sample_rate=SAMPLE_RATE, print_info=False).to(DEVICE)
    mrstft=RIRMetricsLoss(lambda_param={'mrstft': 1}, sample_rate=SAMPLE_RATE, mrstft_care_about_origin=False, print_info=False).to(DEVICE)
    mrstft_care_about_origin=RIRMetricsLoss(lambda_param={'mrstft': 1}, sample_rate=SAMPLE_RATE, mrstft_care_about_origin=True, print_info=False).to(DEVICE)
    multi_filtered_env=RIRMetricsLoss(lambda_param={'ms_env': 1}, sample_rate=SAMPLE_RATE, ms_env_care_about_origin=False, print_info=False).to(DEVICE)
    multi_filtered_env_care_about_origin=RIRMetricsLoss(lambda_param={'ms_env': 1}, sample_rate=SAMPLE_RATE, ms_env_care_about_origin=True, print_info=False).to(DEVICE)

    loss_list=(edc_no_early_reflections,edc_early_reflections,d,c80,rt60, mrstft, mrstft_care_about_origin, multi_filtered_env, multi_filtered_env_care_about_origin, edr_no_early_reflections, edr_early_reflections)

    # Print relevant info
    if PRINT_STUFF:
        print("target room dimensions ", TARGET_ROOM_DIMENSIONS)
        print("mic position ", TARGET_MIC_POSITION)
        print("source_position ", TARGET_SOURCE_POSITION)
        print("room grid step ", GRID_STEP)
        print('MIN_ROOM_DIMENSIONS', MIN_ROOM_DIMENSIONS)
        print('MAX_ROOM_DIMENSIONS', MAX_ROOM_DIMENSIONS)

        print("\n##################### ROOM LOSSES START #####################\n\n")

    # init mic and src positions
    mic_position=torch.tensor(TARGET_MIC_POSITION, dtype=torch.float32, requires_grad=False, device=DEVICE)
    source_position=torch.tensor(TARGET_SOURCE_POSITION, dtype=torch.float32, requires_grad=False, device=DEVICE)

    # Get label rir and origin
    og_label_rir = torch_ism(torch.Tensor(TARGET_ROOM_DIMENSIONS).to(DEVICE),mic_position,source_position,SAMPLE_RATE, max_order=MAX_ORDER)
    label_distance = torch.linalg.norm(mic_position-source_position)
    label_origin = 40 + (SAMPLE_RATE*label_distance/SOUND_SPEED)
    label_origin=torch.unsqueeze(label_origin,0)

    # Get all rooms dimensions to test
    grid_of_room_dimensions=_get_grid_of_room_dimensions(TARGET_ROOM_DIMENSIONS[2], MIN_ROOM_DIMENSIONS, MAX_ROOM_DIMENSIONS, GRID_STEP)

    for room_dimension_as_list in grid_of_room_dimensions:
        # current subtask hyperparameters
        log_row={"max_order": MAX_ORDER,
            "room_dimension_x": room_dimension_as_list[0],"room_dimension_y": room_dimension_as_list[1],"room_dimension_z": room_dimension_as_list[2],
            "target_room_dimensions_x": TARGET_ROOM_DIMENSIONS[0],"target_room_dimensions_y": TARGET_ROOM_DIMENSIONS[1], "target_room_dimensions_z": TARGET_ROOM_DIMENSIONS[2],
            "source_position_x": TARGET_SOURCE_POSITION[0],"source_position_y": TARGET_SOURCE_POSITION[1], "source_position_z": TARGET_SOURCE_POSITION[2],
            "mic_position_x": TARGET_MIC_POSITION[0],"mic_position_y": TARGET_MIC_POSITION[1], "mic_position_z": TARGET_MIC_POSITION[2]
            }
        
        # Format parameters to be logged
        for key , value in log_row.items():
            log_row[key]="{:.3f}".format(value)
        
        # Check if this task is already logged
        if not logger.check_if_line_in_log(log_row):

            room_dimensions=torch.tensor(room_dimension_as_list, dtype=torch.float32, requires_grad=False, device=DEVICE)

            args_list = [(room_dimensions, og_label_rir, label_origin, loss_list) for _ in range(32)] # Experiment 7.1 Random positions
            # args_list = [(room_dimensions, og_label_rir, label_origin, loss_list, label_distance) for _ in range(16)] # Experiment 7.2 Random positions, correct distance

            # Using multiprocessing.Pool
            with timer.time("multiprocessing.Pool()"):
                with multiprocessing.Pool() as pool:
                    results = pool.map(_multiprocessed_loss_computations, args_list)

            edc_early_reflections_loss_list, edc_no_early_reflections_loss_list, d_loss_list , c80_loss_list , rt60_loss_list = [], [], [], [], []
            mrstft_loss_list, mrstft_care_about_origin_loss_list, multi_filtered_env_loss_list, multi_filtered_env_care_about_origin_loss_list = [], [], [], []
            edr_early_reflections_loss_list, edr_no_early_reflections_loss_list = [], []

            for edc_early_reflections_loss, edc_no_early_reflections_loss, d_loss, c80_loss, rt60_loss,\
                mrstft_loss, mrstft_care_about_origin_loss, multi_filtered_env_loss, multi_filtered_env_care_about_origin_loss,\
                edr_early_reflections_loss, edr_no_early_reflections_loss in results:

                edc_early_reflections_loss_list.append(edc_early_reflections_loss)
                edc_no_early_reflections_loss_list.append(edc_no_early_reflections_loss)
                d_loss_list.append(d_loss)
                c80_loss_list.append(c80_loss)
                rt60_loss_list.append(rt60_loss)
                mrstft_loss_list.append(mrstft_loss)
                mrstft_care_about_origin_loss_list.append(mrstft_care_about_origin_loss)
                multi_filtered_env_loss_list.append(multi_filtered_env_loss)
                multi_filtered_env_care_about_origin_loss_list.append(multi_filtered_env_care_about_origin_loss)
                edr_early_reflections_loss_list.append(edr_early_reflections_loss)
                edr_no_early_reflections_loss_list.append(edr_no_early_reflections_loss)

            # Get losses
            shoebox_loss=shoebox_mse(room_dimensions, torch.Tensor(TARGET_ROOM_DIMENSIONS))

            edc_no_early_reflections_loss_mean=torch.tensor(edc_no_early_reflections_loss_list).mean()
            edc_early_reflections_loss_mean=torch.tensor(edc_early_reflections_loss_list).mean()
            
            edc_no_early_reflections_loss_min=torch.tensor(edc_no_early_reflections_loss_list).min()
            edc_early_reflections_loss_min=torch.tensor(edc_early_reflections_loss_list).min()

            edr_no_early_reflections_loss_mean=torch.tensor(edr_no_early_reflections_loss_list).mean()
            edr_early_reflections_loss_mean=torch.tensor(edr_early_reflections_loss_list).mean()
            
            edr_no_early_reflections_loss_min=torch.tensor(edr_no_early_reflections_loss_list).min()
            edr_early_reflections_loss_min=torch.tensor(edr_early_reflections_loss_list).min()

            d_loss_mean=torch.tensor(d_loss_list).mean()
            c80_loss_mean=torch.tensor(c80_loss_list).mean()
            rt60_loss_mean=torch.tensor(rt60_loss_list).mean()
            
            d_loss_min=torch.tensor(d_loss_list).min()
            c80_loss_min=torch.tensor(c80_loss_list).min()
            rt60_loss_min=torch.tensor(rt60_loss_list).min()
            
                # center_time_loss=center_time(torch_rir,torch_origin,label_rir,label_origin)

            mrstft_loss_mean=torch.Tensor(mrstft_loss_list).mean()
            mrstft_care_about_origin_loss_mean=torch.Tensor(mrstft_care_about_origin_loss_list).mean()
            multi_filtered_env_loss_mean=torch.Tensor(multi_filtered_env_loss_list).mean()
            multi_filtered_env_care_about_origin_loss_mean=torch.Tensor(multi_filtered_env_care_about_origin_loss_list).mean()
           
            mrstft_loss_min=torch.Tensor(mrstft_loss_list).min()
            mrstft_care_about_origin_loss_min=torch.Tensor(mrstft_care_about_origin_loss_list).min()
            multi_filtered_env_loss_min=torch.Tensor(multi_filtered_env_loss_list).min()
            multi_filtered_env_care_about_origin_loss_min=torch.Tensor(multi_filtered_env_care_about_origin_loss_list).min()

            # Log losses
            log_row["shoebox_mse"]=shoebox_loss.item()

            log_row["edc_no_early_reflections_mean"]=edc_no_early_reflections_loss_mean.item()
            log_row["edc_early_reflections_mean"]=edc_early_reflections_loss_mean.item()

            log_row["edc_no_early_reflections_min"]=edc_no_early_reflections_loss_min.item()
            log_row["edc_early_reflections_min"]=edc_early_reflections_loss_min.item()

            log_row["edr_no_early_reflections_mean"]=edr_no_early_reflections_loss_mean.item()
            log_row["edr_early_reflections_mean"]=edr_early_reflections_loss_mean.item()

            log_row["edr_no_early_reflections_min"]=edr_no_early_reflections_loss_min.item()
            log_row["edr_early_reflections_min"]=edr_early_reflections_loss_min.item()

            log_row["d_mean"]=d_loss_mean.item()
            log_row["c80_mean"]=c80_loss_mean.item()
            log_row["rt60_mean"]=rt60_loss_mean.item()

            log_row["d_min"]=d_loss_min.item()
            log_row["c80_min"]=c80_loss_min.item()
            log_row["rt60_min"]=rt60_loss_min.item()
                
                # log_row["center_time"]=center_time_loss.item()
           
            log_row["mrstft_mean"]=mrstft_loss_mean.item()
            log_row["mrstft_care_about_origin_mean"]=mrstft_care_about_origin_loss_mean.item()
            log_row["ms_env_mean"]=multi_filtered_env_loss_mean.item()
            log_row["ms_env_care_about_origin_mean"]=multi_filtered_env_care_about_origin_loss_mean.item()

            log_row["mrstft_min"]=mrstft_loss_min.item()
            log_row["mrstft_care_about_origin_min"]=mrstft_care_about_origin_loss_min.item()
            log_row["ms_env_min"]=multi_filtered_env_loss_min.item()
            log_row["ms_env_care_about_origin_min"]=multi_filtered_env_care_about_origin_loss_min.item()


            # Format losses to be logged with not too many decimals
            for loss in which_losses:
                log_row[loss]="{:.6f}".format(log_row[loss])
            
            # print losses and current Room dimension
            if PRINT_STUFF:
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