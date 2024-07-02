import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch

from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset

class GWA_3DFRONT_plus_SBAM_Dataset(GWA_3DFRONT_Dataset):
    '''
    This torch...Dataset class loads a GWA_3DFRONT datapoint AND a SBAM datapoint for each __get_item__ call.
    '''
    def __init__(self, gwa3D_csv_file="./datasets/GWA_3DFRONT/subsets/gwa_3Dfront.csv",
                 rir_length = 3968, sample_rate=16000,
                 rir_std_normalization=False, gwa_scaling_compensation=False, dont_load_rirs=False, dont_load_meshes=False,
                 sbam_csv_file="./datasets/SBAM_Dataset/subsets/sbam.csv"):
        super().__init__(gwa3D_csv_file, rir_length, sample_rate, rir_std_normalization, gwa_scaling_compensation, dont_load_rirs, dont_load_meshes)

        self.sbam_csv_file = sbam_csv_file
        self.sbam_data = pd.read_csv(sbam_csv_file)
        print('SBAM csv loaded at ', sbam_csv_file)

    def __getitem__(self, index):
        # get GWA_3DFRONT data
        gwa_data_list, gwa_label_rir_tensor, gwa_label_origin_tensor, gwa_mic_pos_tensor, gwa_src_pos_tensor = super().__getitem__(index)

        if torch.is_tensor(index):
            index = index.tolist()
        df = self.sbam_data.iloc[index]
        sbam_mesh_path = df['mesh_file_name']
        sbam_label_rir = df['rir_file_name']

        # get all the data
        if not self.dont_load_meshes: x, edge_index = GWA_3DFRONT_Dataset._load_mesh(sbam_mesh_path)
        else:
            if self.a_single_mesh == None:
                self.a_single_mesh = GWA_3DFRONT_Dataset._load_mesh(sbam_mesh_path)
            x, edge_index = self.a_single_mesh
        if not self.dont_load_rirs:
            label_rir, _ = self._load_rir(sbam_label_rir)
            label_origin = df['rir_initial_toa']
        else:
            label_rir, label_origin = np.random.rand(self.rir_length).astype('float32'), 0

        # convert to tensors
        sbam_x = torch.tensor(x, dtype=torch.float32)
        sbam_edge_index = torch.tensor(edge_index, dtype=torch.int).T
        sbam_label_rir_tensor = torch.tensor(label_rir, dtype=torch.float32).squeeze()
        sbam_label_origin_tensor = torch.tensor(label_origin, dtype=torch.float32)
        sbam_room_dim_tensor = torch.tensor([df['rd_x'],df['rd_y'],df['rd_z']], dtype=torch.float32)
        sbam_mic_pos_tensor = torch.tensor([df['mic_x'],df['mic_y'],df['mic_z']], dtype=torch.float32)
        sbam_source_pos_tensor = torch.tensor([df['src_x'],df['src_y'],df['src_z']], dtype=torch.float32)
        sbam_absorption_tensor = torch.tensor([df['absorption_walls'],df['absorption_floor'],df['absorption_ceiling']], dtype=torch.float32)
        # Our decoder doesn't take into account scattering
        # sbam_scattering_tensor = torch.tensor([df['scattering_walls'],df['scattering_floor'],df['scattering_ceiling']], dtype=torch.float32)

        return (gwa_data_list, 
                gwa_label_rir_tensor, 
                gwa_label_origin_tensor,
                gwa_mic_pos_tensor, 
                gwa_src_pos_tensor,
                Data(x=sbam_x, edge_index=sbam_edge_index.long().contiguous()), 
                sbam_label_rir_tensor, 
                sbam_label_origin_tensor,
                sbam_room_dim_tensor,
                sbam_mic_pos_tensor, 
                sbam_source_pos_tensor,
                sbam_absorption_tensor)

    @staticmethod
    def custom_collate_fn(list_of_tuples):
        gwa_data_list, gwa_label_rir_tensors, gwa_label_origin_tensors, gwa_mic_pos_tensors, gwa_src_pos_tensors, \
            sbam_data_list, sbam_label_rir_tensors, sbam_label_origin_tensors, sbam_room_dim_tensors, sbam_mic_pos_tensors, sbam_src_pos_tensors, sbam_absorption_tensors\
            = zip(*list_of_tuples)

        # create a batch vector for the graph data.        
        gwa_graph_batch=Batch.from_data_list(gwa_data_list)
        sbam_graph_batch=Batch.from_data_list(sbam_data_list)

        return (gwa_graph_batch.x, gwa_graph_batch.edge_index, gwa_graph_batch.batch,
                torch.stack(gwa_label_rir_tensors,dim=0), torch.stack(gwa_label_origin_tensors, dim=0),
                torch.stack(gwa_mic_pos_tensors, dim=0), torch.stack(gwa_src_pos_tensors, dim=0),
                sbam_graph_batch.x, sbam_graph_batch.edge_index, sbam_graph_batch.batch,
                torch.stack(sbam_label_rir_tensors,dim=0), torch.stack(sbam_label_origin_tensors, dim=0),
                torch.stack(sbam_room_dim_tensors,dim=0),
                torch.stack(sbam_mic_pos_tensors, dim=0), torch.stack(sbam_src_pos_tensors, dim=0),
                torch.stack(sbam_absorption_tensors, dim=0) )