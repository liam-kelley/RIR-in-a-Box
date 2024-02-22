import pymeshlab as ml
import os
import numpy as np
import pymeshfix

def obj_preprocessing(temp_file_path, preproc_obj_folder, target_faces=5000, random_reduction_factor=0):
    '''
    This preprocesses .obj meshes according to the following steps:
        1. Load the mesh
        2. Optionally, randomly remove a percentage of faces to simulate 'destroying' parts of the mesh
        3. Optionally, Mesh reconstruction on the sparse meshes to obtain realistic looking rooms using Poisson Reconstruction.
        4. Decimate the mesh to the target number of faces
        5. Optionally, Pymeshfix to fix the mesh
        6. Save the preprocessed mesh to the preproc_obj_folder
        7. Return the path to the preprocessed mesh
    '''
    # init pymeshlab
    ms = ml.MeshSet()

    # load mesh
    ms.load_new_mesh(temp_file_path)
    m = ms.current_mesh()
    print('Loaded obj mesh has' , m.face_number(), 'faces and', m.vertex_number(), 'vertices.')

    # # Fix the mesh using pymeshfix # CANT USE PYMESHFIX ON GWA BECAUSE ALL THE FURNITURE IS JUST ADDED AS EXTRAS.
    # v, f = m.vertex_matrix(), m.face_matrix()
    # vclean, fclean = pymeshfix.clean_from_arrays(v, f)
    # # Create a new MeshSet for the fixed mesh
    # m = ml.Mesh(vclean, fclean)
    # ms = ml.MeshSet()
    # ms.add_mesh(m, "fixed_mesh")
    # m = ms.current_mesh()
    # print('Pymeshfixed mesh has', m.face_number(), 'faces and', m.vertex_number(), 'vertices.')

    # Optionally, randomly remove a percentage of faces to simulate 'destroying' parts of the mesh
    # Not tested yet
    if random_reduction_factor > 0:
        total_faces = m.face_number()
        faces_to_remove = int(total_faces * random_reduction_factor)
        all_faces_indices = np.arange(total_faces)
        # Fully random shuffle
        np.random.shuffle(all_faces_indices) 
        faces_to_remove_indices = all_faces_indices[:faces_to_remove]  # Select faces to remove
        v, f = m.vertex_matrix(), m.face_matrix()
        f = np.delete(f, faces_to_remove_indices, axis=0)  # Remove selected faces
        m = ml.Mesh(v, f)
        ms = ml.MeshSet()
        ms.add_mesh(m, "randomly_reduced_mesh")
        # Compact the mesh to remove unused vertices
        ms.apply_filter('meshlab_filter_clean', delete_unreferenced_vertices=True)
        m = ms.current_mesh()
        print('After random reduction, mesh now has', m.vertex_number(), 'vertices and', m.face_number(), 'faces')

    # # Mesh reconstruction on the sparse meshes to obtain realistic looking rooms using Poisson Reconstruction.
    # ms.apply_filter('surface_reconstruction_screened_poisson', preclean=True)
    # m = ms.current_mesh()
    # print("Poisson reconstructed mesh has" , m.face_number(), "faces and ", m.vertex_number(), "vertices.")

    # quadric edge collapse decimation
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target_faces, preservenormal=True)
    m = ms.current_mesh()
    print("Decimated to", m.face_number(), "faces and ", m.vertex_number(), "vertices.")

    # end, and save mesh
    m = ms.current_mesh()
    print('preprocessed mesh has', m.face_number(), 'faces and', m.vertex_number(), 'vertices.')
    simple_obj_path = os.path.join(preproc_obj_folder, temp_file_path.split('/')[-2], temp_file_path.split('/')[-1])
    # make directory if it doesn't exist
    if not os.path.exists(os.path.dirname(simple_obj_path)):
        os.makedirs(os.path.dirname(simple_obj_path))
    ms.save_current_mesh(simple_obj_path)
    print('Saved preprocessed mesh to', simple_obj_path)

    return simple_obj_path