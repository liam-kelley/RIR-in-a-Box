import numpy as np
from typing import Tuple
import trimesh

'''
Procedure to download labelled mesh data from HoloLens:

1. Link the HL2 to the Windows Device Portal.
2. Download the 3D model (OBJ file).
3. Move the obj into the HLobj folder.
'''

def deserialize_mesh(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deserialize mesh data to vertices and triangles.
    
    Returns:
        vertices: Nx3 numpy array of vertex coordinates.
        triangles: Mx3 numpy array of triangle indices.
    """
    with memoryview(data) as view:
        # Read vertices
        vertex_count = int.from_bytes(view[0:4], 'little')
        offset = 4
        vertex_data = []
        for _ in range(vertex_count):
            x = np.frombuffer(view[offset:offset+4], dtype=np.float32)[0]
            offset += 4
            y = np.frombuffer(view[offset:offset+4], dtype=np.float32)[0]
            offset += 4
            z = np.frombuffer(view[offset:offset+4], dtype=np.float32)[0]
            offset += 4
            vertex_data.append([x, y, z])
        vertices = np.array(vertex_data, dtype=np.float32)

        # Read triangles
        triangle_count = int.from_bytes(view[offset:offset+4], 'little') // 3
        offset += 4
        triangle_data = []
        for _ in range(triangle_count):
            triangle = np.frombuffer(view[offset:offset+12], dtype=np.int32)
            triangle_data.append(triangle)
            offset += 12
        triangles = np.array(triangle_data, dtype=np.int32)

    return vertices, triangles

def visualize_meshes(mesh_data: dict):
    """
    Load and visualize multiple meshes.

    Args:
        mesh_data: Dictionary of mesh name to serialized data.
    """
    mesh_colors = {
        "background": [255, 0, 0, 255],  # Red
        "ceiling": [0, 255, 0, 255],    # Green
        "Floor": [0, 0, 255, 255],      # Blue
        "Wall": [255, 255, 0, 255]      # Yellow
    }
    scene = trimesh.Scene()

    for mesh_name, data in mesh_data.items():
        vertices, triangles = deserialize_mesh(data)
        print(mesh_name, vertices.shape, triangles.shape)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=mesh_colors[mesh_name])
        scene.add_geometry(mesh)

    scene.show()

def load_data_from_file(file_path: str) -> bytes:
    """Load serialized mesh data from a file."""
    with open(file_path, 'rb') as file:
        return file.read()

if __name__ == "__main__":
    # File paths for each mesh type:
    background_path = "HLobj/Labelled/Attempt2/Background_0004.dat"
    ceiling_path = "HLobj/Labelled/Attempt2/Ceiling_0004.dat"
    floor_path = "HLobj/Labelled/Attempt2/Floor_0004.dat"
    wall_path = "HLobj/Labelled/Attempt2/Wall_0004.dat"

    # Load serialized data from the files:
    background_data = load_data_from_file(background_path)
    ceiling_data = load_data_from_file(ceiling_path)
    floor_data = load_data_from_file(floor_path)
    wall_data = load_data_from_file(wall_path)

    mesh_data = {
        "background": background_data,
        "ceiling": ceiling_data,
        "Floor": floor_data,
        "Wall": wall_data
    }
    
    visualize_meshes(mesh_data)
