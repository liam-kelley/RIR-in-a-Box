import pymeshlab
import os
import pickle
import glob
import numpy as np
import pandas as pd

from mesh_dataset import load_wall_features_and_wall_edge_index, edge_index_to_triangle_list, triangle_list_to_edge_index, plot_mesh_from_edge_index

path ="meshdataset/meshes"
simplified_path = "meshdataset/simplified_meshes"

TARGET_FACES=100
PLOT=False

if not os.path.exists(simplified_path):
    os.mkdir(simplified_path)

# mesh_paths=glob.glob(os.path.join(path, '*.meshpkl'))
data = pd.read_csv("meshdataset/shoebox_mesh_dataset.csv")
data['simplified_mesh_file_name'] = ""

for idx, row in data.iterrows():
    mesh_path = row['mesh_file_name']
    formatted_idx = "{:06}".format(idx)
    new_mesh_path = os.path.join(simplified_path, "mesh_" + formatted_idx + ".meshpkl") # yeah, i save them as mesh pickles

    with open(mesh_path, 'rb') as f:
        # load mesh.
        wall_features, wall_edge_index = pickle.load(f)
        x, edge_index = load_wall_features_and_wall_edge_index(mesh_path)
        
        if PLOT : plot_mesh_from_edge_index(x , edge_index, show=False)
        
        # Filter The loaded mesh
        og_vertices = x[:, :3]
        og_triangle_list = edge_index_to_triangle_list(edge_index)
        mesh = pymeshlab.Mesh(og_vertices, og_triangle_list)
        print('input mesh has', mesh.vertex_number(), 'vertex and', mesh.face_number(), 'faces')
        meshset = pymeshlab.MeshSet()
        meshset.add_mesh(mesh)
        mesh = meshset.current_mesh()
        meshset.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=TARGET_FACES, preservenormal=True)
        mesh = meshset.current_mesh()
        new_vertices = mesh.vertex_matrix()
        new_triangle_list = mesh.face_matrix()
        print('output mesh has', mesh.vertex_number(), 'vertex and', mesh.face_number(), 'faces. Target faces:', TARGET_FACES)

        # Find nearest node for each new filtered nodes, and use the label features of that node for the new node
        og_vertices=x[:,:3]
        new_x=[]
        for vertex in new_vertices:
            min_dist = np.inf
            for i, og_vertex in enumerate(og_vertices):
                dist = np.linalg.norm(vertex-og_vertex)
                if dist < min_dist:
                    min_dist = dist
                    min_point_idx = i
            new_x.append(np.concatenate((vertex,x[min_point_idx,3:])))
        new_x = np.asarray(new_x)

        new_edge_index = triangle_list_to_edge_index(new_triangle_list)
        
        if PLOT : plot_mesh_from_edge_index(new_x , new_edge_index, show=True)

    with open(new_mesh_path, 'wb') as file:
            pickle.dump((new_x,new_edge_index),file)

    # Update the dataset descriptor with the new path
    data.at[idx, 'simplified_mesh_file_name'] = new_mesh_path
    print("saved",new_mesh_path )

# Save the updated metadata dataframe back to CSV
data.to_csv("meshdataset/shoebox_mesh_dataset_w_simple.csv", index=False)