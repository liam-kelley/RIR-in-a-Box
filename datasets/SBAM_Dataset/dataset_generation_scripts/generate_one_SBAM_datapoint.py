from dataset_generation_scripts.config_and_csv_utility import check_config_dict, get_empty_csv_row

from get_sbox_room_configuration import get_sbox_room_configuration
from get_shoebox_mesh import get_shoebox_mesh, save_shoebox_mesh
from get_rir import get_rir, save_rir

import matplotlib.pyplot as plt
import numpy as np

def generate_one_SBAM_datapoint(config : dict, args):
    check_config_dict(config)

    # Get data
    sbox_dim, materials, mic_pos, src_pos = get_sbox_room_configuration(config)
    rir = get_rir(config, sbox_dim, materials, mic_pos, src_pos)
    mesh = get_shoebox_mesh(config, sbox_dim)

    # Plot data
    if args.plot:
        plt.plot(rir)
        plt.ylabel("Amplitude")
        plt.xlabel(f"Time in samples ({config["rir_sample_rate"]}Hz)")
        plt.show()
        mesh.plot()

    # Save data on disk
    if not args.dontsave:
        rir_file_name = save_rir(config, rir)
        mesh_file_name = save_shoebox_mesh(mesh)
    else:
        rir_file_name=""
        mesh_file_name=""

    # Log data
    log_row = get_empty_csv_row()
    log_row['mesh_file_name'] = mesh_file_name
    log_row['rir_file_name'] = rir_file_name

    log_row['rir_initial_toa'] = np.linalg.norm(mic_pos-src_pos) * config["rir_sample_rate"] + 40 # 40 is pra window length (81) //2
    log_row['rir_sample_rate'] = config['rir_sample_rate']
    log_row['rir_max_order'] = config['rir_max_order']

    log_row['rd_x'] = sbox_dim[0]
    log_row['rd_y'] = sbox_dim[1]
    log_row['rd_z'] = sbox_dim[2]
    log_row['mic_x'] = mic_pos[0]
    log_row['mic_y'] = mic_pos[1]
    log_row['mic_z'] = mic_pos[2]
    log_row['src_x'] = src_pos[0]
    log_row['src_y'] = src_pos[1]
    log_row['src_z'] = src_pos[2]

    log_row['absorption_floor'] = materials["floor"].energy_absorption
    log_row['absorption_ceiling'] = materials["ceiling"].energy_absorption
    log_row['absorption_walls'] = materials["north"].energy_absorption

    log_row['scattering_floor'] = materials["floor"].scattering
    log_row['scattering_ceiling'] = materials["ceiling"].scattering
    log_row['scattering_walls'] = materials["north"].scattering

    return (log_row)
    