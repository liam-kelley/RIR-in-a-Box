import random
import pyroomacoustics as pra
import numpy as np
import random

def check_min_sbox_dims(config : dict):
    if (1.66*config['sbox_dim_x_min'] < config['src_mic_distance_min'] + config['src_mic_wall_distance_min']*2 ) \
        or (1.66*config['sbox_dim_y_min'] < config['src_mic_distance_min'] + config['src_mic_wall_distance_min']*2 ) \
        or (1.66*config['sbox_dim_z_min'] < config['src_mic_distance_min'] + config['src_mic_wall_distance_min']*2 ) : 
        raise ValueError( "Room dimensions are too small for src_mic_distance_min or src_mic_wall_distance_min")

def make_random_sbox(config : dict):
    sbox_dim_x = random.uniform(config['sbox_dim_x_min'],config['sbox_dim_x_max'])
    sbox_dim_y = random.uniform(config['sbox_dim_y_min'],config['sbox_dim_y_max'])
    sbox_dim_z = random.uniform(config['sbox_dim_z_min'],config['sbox_dim_z_max'])

    counter=0
    while sbox_dim_x*sbox_dim_y*sbox_dim_z < config['sbox_volume_min'] or sbox_dim_x*sbox_dim_y < config['sbox_floor_area_min']:
        sbox_dim_x = random.uniform(config['sbox_dim_x_min'],config['sbox_dim_x_max'])
        sbox_dim_y = random.uniform(config['sbox_dim_y_min'],config['sbox_dim_y_max'])
        sbox_dim_z = random.uniform(config['sbox_dim_z_min'],config['sbox_dim_z_max'])
        
        counter+=1
        if counter>1000: raise ValueError( "couldn't find a room with enough volume or floor area")
        
    return np.array([sbox_dim_x, sbox_dim_y, sbox_dim_z])

def get_materials():
    # Create a shoebox pra room
    wall_material = pra.Material(energy_absorption=random.uniform(0.01,0.81), scattering=random.uniform(0.05,0.8))
    floor_material = pra.Material(energy_absorption=random.uniform(0.01,0.81), scattering=random.uniform(0.05,0.5))
    ceiling_material = pra.Material(energy_absorption=random.uniform(0.01,0.81), scattering=random.uniform(0.05,0.7))
    # Define the materials for each surface of the room
    materials = {
        'east': wall_material,   # Right wall
        'west': wall_material,   # Left wall
        'north': wall_material,  # Front wall
        'south': wall_material,  # Back wall
        'floor': floor_material,
        'ceiling': ceiling_material
    }
    return materials

def get_mic_and_src_within_sbox(config, sbox_dim):
    mic_pos = np.ones(3)*config["src_mic_wall_distance_min"] + np.random.random(3)*(sbox_dim - np.ones(3)*config["src_mic_wall_distance_min"]*2)
    src_pos = np.ones(3)*config["src_mic_wall_distance_min"] + np.random.random(3)*(sbox_dim - np.ones(3)*config["src_mic_wall_distance_min"]*2)

    counter = 0
    while np.linalg.norm(src_pos - mic_pos) < config['src_mic_distance_min']:
        mic_pos = np.ones(3)*config["src_mic_wall_distance_min"] + np.random.random(3)*(sbox_dim - np.ones(3)*config["src_mic_wall_distance_min"]*2)
        src_pos = np.ones(3)*config["src_mic_wall_distance_min"] + np.random.random(3)*(sbox_dim - np.ones(3)*config["src_mic_wall_distance_min"]*2)

        counter+=1
        if counter>1000: raise ValueError( "couldn't find a mic-src configuration that fits min mic-src distance and min mic-src-wall distance")

    return mic_pos, src_pos


def get_sbox_room_configuration(config):
    check_min_sbox_dims(config)

    # Generate a sbox room configuration
    sbox_dim = make_random_sbox(config)
    materials = get_materials()
    mic_pos, src_pos = get_mic_and_src_within_sbox(config, sbox_dim)

    return sbox_dim, materials, mic_pos, src_pos