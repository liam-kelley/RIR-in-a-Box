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

from scipy.signal import find_peaks

from datasets.GWA_3DFRONT.preprocessing.rir_preprocessing import mesh2ir_rir_preprocessing

def string_to_array(s):
    '''
    Useful for what's in that dataset csv file
    '''
    # Remove square brackets and split the string
    elements = s.strip("[]").split()
    # Convert each element to float and create a numpy array
    return np.array([float(e) for e in elements])

def edge_matrix_from_face_matrix(face_matrix):
    '''
    TODO Need to check if this is OK.
    '''
    # Iterate over face_matrix to extract edges
    edges = []
    for f in face_matrix:
        face_edges = [(f[i], f[(i+1) % len(f)]) for i in range(len(f))]
        for e in face_edges:
            if e not in edges and (e[1], e[0]) not in edges:  # Avoid duplicates
                edges.append(e)
    # Convert edges list to a numpy array
    edge_matrix = np.array(edges)
    return edge_matrix

class GWA_3DFRONT_Dataset(Dataset):
    def __init__(self, csv_file="./datasets/GWA_3DFRONT/gwa_3Dfront.csv", rir_std_normalization=False, gwa_scaling_compensation=False):
        self.csv_file=csv_file
        self.rir_std_normalization = rir_std_normalization
        self.gwa_scaling_compensation = gwa_scaling_compensation
        self.sample_rate=16000
        self.data = pd.read_csv(csv_file)
        print('GWA_3DFRONT csv loaded at ', csv_file)

        self.meshes_folder = "./datasets/GWA_3DFRONT/preprocessed_obj_meshes"
        if not os.path.exists(self.meshes_folder):
            raise Exception("Mesh folder not found: ", self.meshes_folder, "\nDidn't you run the preprocess_3Dfront.py script first?")
        
        self.label_rir_folder = "./datasets/GWA_3DFRONT/GWA_Dataset_small"
        if not os.path.exists(self.label_rir_folder):
            raise Exception("Label RIR folder not found: ", self.label_rir_folder,
                            "\nPlease download the GWA_Dataset_small and place it in the datasets/GWA_3DFRONT folder.")

        print('GWA_3DFRONT dataset loaded at ', self.csv_file)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _load_mesh(mesh_path):
        # Load your mesh
        ms = ml.MeshSet()
        ms.load_new_mesh(mesh_path)
        # Get the current mesh
        m = ms.current_mesh()
        x = m.vertex_matrix()
        edge_matrix = edge_matrix_from_face_matrix(m.face_matrix())
        return x.astype('float32'), edge_matrix.astype('long')
    
    @staticmethod
    def _load_rir(label_rir_path, rir_std_normalization=True, gwa_scaling_compensation=False):
        # Load RIR
        label_rir, fs = librosa.load(label_rir_path)
        
        # Resample to 16kHz
        label_rir = librosa.resample(label_rir,orig_sr=fs, target_sr=16000)

        # crop or pad all rirs to same length
        crop_length = 3968
        length = label_rir.size
        if(length<crop_length):
            zeros = np.zeros(crop_length-length)
            label_rir = np.concatenate([label_rir,zeros])
        else: label_rir = label_rir[0:crop_length]

        # Preprocess RIR (standardization by std)
        if rir_std_normalization :
            label_rir = mesh2ir_rir_preprocessing(label_rir)

        if gwa_scaling_compensation:
            label_rir = label_rir / 0.0625029951333999

        label_rir = np.array([label_rir]).astype('float32')

        # find origin of RIR
        label_origin = GWA_3DFRONT_Dataset._estimate_origin(label_rir)

        return label_rir, label_origin

    @staticmethod
    def _estimate_origin(label_rir):
        peak_indexes, _ = find_peaks(label_rir[0],height=0.05*np.max(label_rir), distance=40)
        # peak_indexes, _ = find_peaks(label_rir[0],height=0.8*np.max(label_rir), distance=40) # Only activate this for find_mystery_scaling.py
        try:
            label_origin = peak_indexes[0]
        except IndexError:
            label_origin = 0
        return label_origin

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        df = self.data.iloc[index]

        # path names
        mesh_path = os.path.join(self.meshes_folder, df['mesh_name'])
        label_rir_path = os.path.join(self.label_rir_folder, df['rir_name'])

        # get all the data
        x, edge_index = GWA_3DFRONT_Dataset._load_mesh(mesh_path)
        label_rir, label_origin = GWA_3DFRONT_Dataset._load_rir(label_rir_path, self.rir_std_normalization, self.gwa_scaling_compensation)
        src_pos = string_to_array(df["Source_Pos"]).astype('float32')
        mic_pos = string_to_array(df["Receiver_Pos"]).astype('float32')

        # convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.int).T
        label_rir_tensor = torch.tensor(label_rir, dtype=torch.float32).squeeze()
        label_origin_tensor = torch.tensor(label_origin, dtype=torch.float32)
        mic_pos_tensor = torch.tensor(mic_pos, dtype=torch.float32)
        source_pos_tensor = torch.tensor(src_pos, dtype=torch.float32)

        return (Data(x=x, edge_index=edge_index.long().contiguous()), 
                label_rir_tensor, 
                label_origin_tensor,
                mic_pos_tensor, 
                source_pos_tensor)

    @staticmethod
    def custom_collate_fn(list_of_tuples):
        data_list, label_rir_tensors, label_origin_tensors, mic_pos_tensors, src_pos_tensors = zip(*list_of_tuples)

        # create a batch vector for the graph data.        
        graph_batch=Batch.from_data_list(data_list)

        return (graph_batch.x, graph_batch.edge_index, graph_batch.batch,
                torch.stack(label_rir_tensors,dim=0), torch.stack(label_origin_tensors, dim=0),
                torch.stack(mic_pos_tensors, dim=0), torch.stack(src_pos_tensors, dim=0) )