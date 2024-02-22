from torch.utils.data import Dataset
# import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import os
import torch
import pymeshlab as ml

from torch_geometric.data import Data
from torch_geometric.data import Batch

def string_to_array(s):
    '''
    Useful for what's in that dataset csv file
    '''
    # Remove square brackets and split the string
    elements = s.strip("[]").split()
    # Convert each element to float and create a numpy array
    return np.array([float(e) for e in elements])

from datasets.GWA_3DFRONT.preprocessing.rir_preprocessing import mesh2ir_rir_preprocessing

class my_dataset(Dataset):
    def __init__(self, csv_file="./datasets/GWA_3DFRONT/gwa_3Dfront.csv", ):
        self.csv_file=csv_file
        self.sample_rate=None
        self.data = pd.read_csv(csv_file)
        print('GWA_3DFRONT csv loaded at ', csv_file)

        self.meshes_folder = "./datasets/GWA_3DFRONT/preprocessed_obj_meshes"
        if not os.path.exists(self.meshes_folder):
            raise Exception("Mesh folder not found: ", self.meshes_folder, "\nDidn't you run the preprocess_3Dfront.py script first?")
        
        self.label_rir_folder = "./datasets/GWA_3DFRONT/GWA_Dataset_small"
        if not os.path.exists(self.label_rir_folder):
            raise Exception("Label RIR folder not found: ", self.label_rir_folder,
                            "\nPlease download the GWA_Dataset_small and place it in the datasets/GWA_3DFRONT folder.")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        df = self.data.iloc[index]

        mesh_path = os.path.join(self.meshes_folder, df['mesh_name'])
        label_rir_path = os.path.join(self.label_rir_folder, df['rir_name'])

        ####### LOAD MESH #######

        # Load the mesh
        ms = ml.MeshSet()
        ms.load_new_mesh(mesh_path)

        # get x and edge_index
        x = ms.current_mesh.vertices
        edge_index = ms.current_mesh.edges

        ####### LOAD RIR #######

        label_rir, fs = librosa.load(label_rir_path)
        label_rir = librosa.resample(label_rir,orig_sr=fs, target_sr=16000)
        label_rir = mesh2ir_rir_preprocessing(label_rir)
        label_rir = np.array([label_rir]).astype('float32')

        # find origin of RIR
        # peak_indexes, _ = find_peaks(label_rir/np.max(label_rir),height=0.3)
        # label_origin = peak_indexes[0]
        label_origin = 41 # TODO: find a way to get the origin of the RIR

        ####### GET Ground Truth Mic and Src positions #######

        src_pos = string_to_array(df["Source_Pos"]).astype('float32')
        mic_pos = string_to_array(df["Receiver_Pos"]).astype('float32')

        ######### CREATE TENSORS #########

        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.int)
        label_rir_tensor = torch.tensor(label_rir, dtype=torch.float32)
        label_origin_tensor = torch.tensor(label_origin, dtype=torch.float32)
        mic_pos_tensor = torch.tensor(mic_pos, dtype=torch.float32)
        source_pos_tensor = torch.tensor(src_pos, dtype=torch.float32)

        return (Data(x=x, edge_index=edge_index.int().contiguous()), 
                label_rir_tensor, 
                label_origin_tensor,
                mic_pos_tensor, 
                source_pos_tensor)

    def custom_collate_fn(list_of_tuples):
        data_list, label_rir_tensors, label_origin_tensors, mic_pos_tensors, src_pos_tensors = zip(*list_of_tuples)

        # create a batch vector for the graph data.        
        graph_batch=Batch.from_data_list(data_list)

        return (graph_batch.x, graph_batch.edge_index, graph_batch.batch,
                torch.stack(label_rir_tensors,dim=0), torch.stack(label_origin_tensors, dim=0),
                torch.stack(mic_pos_tensors, dim=0), torch.stack(src_pos_tensors, dim=0) )