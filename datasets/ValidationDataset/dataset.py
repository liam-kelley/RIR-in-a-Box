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

class HL2_Dataset(Dataset):
    def __init__(self, csv_file="datasets/ValidationDataset/subsets/realval_dataset.csv", rir_length = 3968, sample_rate=16000,
                 dont_load_rirs=False, dont_load_meshes=False):
        self.csv_file=csv_file
        self.sample_rate=sample_rate
        self.rir_length=rir_length
        self.dont_load_rirs = dont_load_rirs
        self.dont_load_meshes = dont_load_meshes
        if self.dont_load_meshes:
            self.a_single_mesh = None
        self.data = pd.read_csv(csv_file)
        print('GWA_3DFRONT csv loaded at ', csv_file)

        self.meshes_folder = "./datasets/ValidationDataset/fixed_meshes"
        if not os.path.exists(self.meshes_folder):
            raise Exception("Mesh folder not found: ", self.meshes_folder)
        
        self.label_rir_folder = "datasets/ValidationDataset/estimated_rirs"
        if not os.path.exists(self.label_rir_folder):
            raise Exception("Label RIR folder not found: ", self.label_rir_folder)

        print('Real Validation dataset loaded at ', self.csv_file)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _load_mesh(mesh_path):
        # Load your mesh
        ms = ml.MeshSet()
        ms.load_new_mesh(mesh_path)

        # do poisson mesh reconstruction

        # ms.apply_filter('surface_reconstruction_screened_poisson', preclean=True)
        # m = ms.current_mesh()
        # print("Poisson reconstructed mesh has" , m.face_number(), "faces and ", m.vertex_number(), "vertices.")

        # preprocess like mesh2ir
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=2000, preservenormal=True)
        m = ms.current_mesh()
        print("Decimated to", m.face_number(), "faces and ", m.vertex_number(), "vertices.")

        # Get the current mesh
        m = ms.current_mesh()

        x = m.vertex_matrix()
        edge_matrix = m.edge_matrix() # edge_matrix = edge_matrix_from_face_matrix(m.face_matrix())

        return x.astype('float32'), edge_matrix.astype('long')
    
    def _load_rir(self, label_rir_path):
        # Load RIR
        label_rir, _ = librosa.load(label_rir_path, sr=self.sample_rate, duration=self.rir_length/self.sample_rate)

        # crop or pad all rirs to same length
        length = label_rir.size
        if(length<self.rir_length):
            zeros = np.zeros(self.rir_length-length)
            label_rir = np.concatenate([label_rir,zeros])
        else: label_rir = label_rir[0:self.rir_length]

        label_rir = np.array(label_rir).astype('float32')

        # find origin of RIR
        ir_onset = HL2_Dataset._estimate_origin(label_rir)

        return label_rir, ir_onset

    @staticmethod
    def _estimate_origin(label_rir):
        peak_indexes, _ = find_peaks(label_rir,height=0.05*np.max(label_rir), distance=40)
        try:
            label_origin = peak_indexes[0]
        except IndexError:
            print("No peak found in loaded RIR. Returning 0 as origin.")
            label_origin = 0
        return label_origin

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        df = self.data.iloc[index]

        # path names
        mesh_path = os.path.join(self.meshes_folder, df['mesh_name'])
        label_rir_path = os.path.join(self.label_rir_folder, df['audio_name'])

        # get all the data
        if not self.dont_load_meshes: x, edge_index = HL2_Dataset._load_mesh(mesh_path)
        else:
            if self.a_single_mesh == None:
                self.a_single_mesh = HL2_Dataset._load_mesh(mesh_path)
            x, edge_index = self.a_single_mesh
        if not self.dont_load_rirs: label_rir, observed_origin = self._load_rir(label_rir_path)
        else: label_rir, observed_origin = np.random.rand(self.rir_length).astype('float32'), 0
        
        src_pos = np.array([df["SrcX"],df["SrcY"],df["SrcZ"]]).astype('float32')
        mic_pos = np.array([df["MicX"],df["MicY"],df["MicZ"]]).astype('float32')

        # estimate origin
        distance = np.linalg.norm(src_pos - mic_pos)
        predicted_Ir_onset = (distance / 343 ) * 16000
        # print(label_rir.shape, predicted_Ir_onset, observed_origin)

        # Respatialize RIR
        pad = int(predicted_Ir_onset - observed_origin)
        if pad < 0:
            label_rir = label_rir[abs(pad):]
            # pad ending of rir
            label_rir = np.concatenate([label_rir, np.zeros(abs(pad))])
        else:
            label_rir = np.concatenate([np.zeros(pad), label_rir])
            label_rir = label_rir[:-pad]

        # Rescale RIR
        label_rir = label_rir * (1 / distance)

        # import matplotlib.pyplot as plt
        # plt.plot(label_rir)
        # plt.axvline(x=observed_origin, color='r',label="Observed Origin")
        # plt.axvline(x=predicted_Ir_onset, color='g', label="Predicted Origin")
        # plt.show()

        # convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.int).T
        label_rir_tensor = torch.tensor(label_rir, dtype=torch.float32).squeeze()
        label_origin_tensor = torch.tensor(predicted_Ir_onset, dtype=torch.float32)
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