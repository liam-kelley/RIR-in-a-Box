from torch import load

def load_mesh_net(mesh_net_obj,mesh_net_path):
    '''
    load MESH2IR mesh_net.
    Renames sate dict variable names for torch 12.1 compatibility. 
    '''
    state_dict = load(mesh_net_path,
                    map_location=lambda storage, loc: storage)
    change_dict = {"pool1.weight":"pool1.select.weight",
                    "pool2.weight":"pool2.select.weight",
                    "pool3.weight":"pool3.select.weight"}
    state_dict_torch121 = {}
    for k,v in state_dict.items():
        if k in change_dict.keys(): state_dict_torch121[change_dict[k]] = v
        else: state_dict_torch121[k] = v
    mesh_net_obj.load_state_dict(state_dict_torch121)
    print('Pretrained mesh_net loaded from: ', mesh_net_path)
    return mesh_net_obj

def load_GAN(GAN_obj,GAN_path):
    '''
    load MESH2IR GAN.
    '''
    state_dict = load(GAN_path,
                    map_location=lambda storage, loc: storage)
    GAN_obj.load_state_dict(state_dict)
    print('Pretrained GAN loaded from: ', GAN_path)
    return GAN_obj

def load_mesh_to_shoebox(mesh_to_shoebox_obj,mesh_to_shoebox_path):
    '''
    load RIRBox mesh_to_shoebox.
    '''
    state_dict = load(mesh_to_shoebox_path,
                    map_location=lambda storage, loc: storage)
    mesh_to_shoebox_obj.load_state_dict(state_dict)
    return mesh_to_shoebox_obj
