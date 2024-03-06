from torch import load
from torch.cuda import is_available
from torch.nn import Module
from models.mesh2ir_models import MESH2IR_FULL, MESH_NET, STAGE1_G
from models.rirbox_models import MeshToShoebox, ShoeboxToRIR, RIRBox_FULL, RIRBox_MESH2IR_Hybrid
from training.utility import filter_rir_like_rirbox
import torch
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from json import load as json_load

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

def load_all_models_for_inference(model_config : str, START_FROM_IR_ONSET=True, ISM_MAX_ORDER=None):
    with open(model_config, 'r') as file: config = json_load(file)
    if ISM_MAX_ORDER is not None: config['ISM_MAX_ORDER'] = ISM_MAX_ORDER
    DEVICE = 'cuda' if is_available() else 'cpu'
    if config['SAVE_PATH'] == "": config['SAVE_PATH'] = "./models/RIRBOX/"+ model_config.split("/")[-2] + "/"+ model_config.split("/")[-1].split(".")[0] + ".pth"
    print("PARAMETERS:")
    for key, value in config.items():
        print(f"    > {key} = {value}")
    print("")

    # Init baseline
    mesh_net = load_mesh_net(MESH_NET(), "./models/MESH2IR/mesh_net_epoch_175.pth")
    net_G = load_GAN(STAGE1_G(), "./models/MESH2IR/netG_epoch_175.pth")
    mesh2ir = MESH2IR_FULL(mesh_net, net_G).eval().to(DEVICE)
    print("")

    # Init Rirbox
    mesh_to_shoebox = load_mesh_to_shoebox(MeshToShoebox(meshnet=mesh_net,
                                                        model=config['RIRBOX_MODEL_ARCHITECTURE'],
                                                        MLP_Depth=config['MLP_DEPTH'],
                                                        hidden_size=config['HIDDEN_LAYER_SIZE'],
                                                        dropout_p=False,
                                                        random_noise=False,
                                                        distance_in_latent_vector=config["DIST_IN_LATENT_VECTOR"]),
                                            config['SAVE_PATH'])
    shoebox_to_rir = ShoeboxToRIR(sample_rate=16000,
                                max_order=config['ISM_MAX_ORDER'],
                                rir_length=3968,
                                start_from_ir_onset=START_FROM_IR_ONSET,
                                normalized_distance=False)
    rirbox = RIRBox_FULL(mesh_to_shoebox, shoebox_to_rir, return_sbox=True).eval().to(DEVICE)
    print("")

    # Init Hybrid Model
    hybrid = RIRBox_MESH2IR_Hybrid(mesh_to_shoebox, shoebox_to_rir, return_sbox=False).eval().to(DEVICE)
    print("")

    return mesh2ir, rirbox, hybrid, config, DEVICE

def print_model_params(model : Module):
    # get the total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

def inference_on_all_models(x_batch : torch.Tensor, edge_index_batch : torch.Tensor, batch_indexes : torch.Tensor,
                            mic_pos_batch : torch.Tensor, src_pos_batch : torch.Tensor, label_origin_batch : torch.Tensor,
                            mesh2ir : MESH2IR_FULL, rirbox : RIRBox_FULL, hybrid : RIRBox_MESH2IR_Hybrid, DEVICE : str,
                            SCALE_MESH2IR_BY_ITS_ESTIMATED_STD : bool = False,
                            SCALE_MESH2IR_GWA_SCALING_COMPENSATION : bool = False,
                            MESH2IR_USES_LABEL_ORIGIN : bool = False,
                            RESPATIALIZE_RIRBOX : bool = False,
                            FILTER_MESH2IR_IN_HYBRID : bool = False):
    # Moving data to device
    x_batch = x_batch.to(DEVICE)
    edge_index_batch = edge_index_batch.to(DEVICE)
    batch_indexes = batch_indexes.to(DEVICE)
    mic_pos_batch = mic_pos_batch.to(DEVICE)
    src_pos_batch = src_pos_batch.to(DEVICE)

    # Find Ground Truth theoretical direct path onset
    distance = torch.linalg.norm(mic_pos_batch[0]-src_pos_batch[0])
    dp_onset_in_samples = int(distance*16000/343)
    
    # MESH2IR
    rir_mesh2ir = mesh2ir(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
    assert(rir_mesh2ir.shape[1] == 4096)
    if SCALE_MESH2IR_BY_ITS_ESTIMATED_STD: rir_mesh2ir = rir_mesh2ir*torch.mean(rir_mesh2ir[:,-64:], dim=1).unsqueeze(1).expand(-1, 4096)
    if SCALE_MESH2IR_GWA_SCALING_COMPENSATION: rir_mesh2ir = rir_mesh2ir / 0.0625029951333999
    
    rir_mesh2ir = rir_mesh2ir[:3968]
    if MESH2IR_USES_LABEL_ORIGIN: origin_mesh2ir = label_origin_batch.to(DEVICE)
    else : origin_mesh2ir = torch.tensor([GWA_3DFRONT_Dataset._estimate_origin(rir_mesh2ir.cpu().numpy())]).to(DEVICE)

    # RIRBOX
    rir_rirbox, origin_rirbox, latent_vector = rirbox(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch)
    virtual_shoebox = ShoeboxToRIR.extract_shoebox_from_latent_representation(latent_vector)
    if RESPATIALIZE_RIRBOX: rir_rirbox, origin_rirbox = ShoeboxToRIR.respatialize_rirbox(rir_rirbox, dp_onset_in_samples)
    
    # Hybrid model
    if FILTER_MESH2IR_IN_HYBRID :
        rir_mesh2ir_filtered = filter_rir_like_rirbox(rir_mesh2ir)
        hybrid_rir, origin_hybrid = hybrid(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch, rir_mesh2ir_filtered, origin_mesh2ir)
    else:
        hybrid_rir, origin_hybrid = hybrid(x_batch, edge_index_batch, batch_indexes, mic_pos_batch, src_pos_batch, rir_mesh2ir, origin_mesh2ir)
    if RESPATIALIZE_RIRBOX:
        hybrid_rir, origin_hybrid = ShoeboxToRIR.respatialize_rirbox(hybrid_rir, dp_onset_in_samples)

    return rir_mesh2ir, rir_rirbox, hybrid_rir, origin_mesh2ir, origin_rirbox, origin_hybrid, virtual_shoebox

