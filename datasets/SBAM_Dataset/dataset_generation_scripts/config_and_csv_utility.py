def get_empty_csv_row():
    empty_csv_row={
        'mesh_file_name' : None,
        'rir_file_name' : None,

        'rir_initial_toa': None, # Redundant, but it's nice to have immediately
        'rir_sample_rate' : None,
        'rir_max_order' : None,

        'rd_x': None,
        'rd_y': None,
        'rd_z': None,
        'mic_x': None,
        'mic_y': None,
        'mic_z': None,
        'src_x': None,
        'src_y': None,
        'src_z': None,

        'absorption_floor': None,
        'absorption_ceiling': None,
        'absorption_walls': None,
        'scattering_floor': None,
        'scattering_ceiling': None,
        'scattering_walls': None,
    }
    return empty_csv_row

def check_config_dict(config : dict):
    assert "sbox_dim_x_min" in config.keys()
    assert "sbox_dim_x_max" in config.keys()
    assert "sbox_dim_y_min" in config.keys()
    assert "sbox_dim_y_max" in config.keys()
    assert "sbox_dim_z_min" in config.keys()
    assert "sbox_dim_z_max" in config.keys()
    assert "sbox_floor_area_min" in config.keys()
    assert "sbox_volume_min" in config.keys()
    assert "src_mic_distance_min" in config.keys()
    assert "src_mic_wall_distance_min" in config.keys()
    assert "mesh_nodes_per_m2_min" in config.keys()
    assert "mesh_nodes_per_m2_max" in config.keys()
    assert "mesh_node_max_normal_random_offset_min" in config.keys()
    assert "mesh_node_max_normal_random_offset_max" in config.keys()
    assert "rir_sample_rate" in config.keys()
    assert "rir_max_order" in config.keys()
