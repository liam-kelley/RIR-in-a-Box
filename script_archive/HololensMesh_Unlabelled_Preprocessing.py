'''
Procedure to download mesh data from HoloLens:
1. Link the HL2 to the Windows Device Portal.
2. Download the 3D model (OBJ file).
3. Move the obj into the HLobj folder.
'''

import pymeshlab
import pymeshfix
import trimesh

filenames = ['./HLobj/AfterReset/SpatialMapping0.obj', './HLobj/AfterReset/SpatialMapping1.obj',
             './HLobj/AfterReset/SpatialMapping2.obj']

ms = pymeshlab.MeshSet()
ms.load_new_mesh(filenames[0])
mesh = ms.current_mesh()
vertices = mesh.vertex_matrix()
triangle_list = mesh.face_matrix()

# ORIGINAL RAW MESH
print("_______ORIGINAL RAW MESH_______")
print("vertices",vertices.shape,vertices)
print("triangle_list",triangle_list.shape,triangle_list)

# POISSON RECONSTRUCTION
mesh = pymeshlab.Mesh(vertices, triangle_list)
ms.add_mesh(mesh)
ms.apply_filter('generate_surface_reconstruction_screened_poisson', preclean=True)
mesh = ms.current_mesh()
vertices = mesh.vertex_matrix()
triangle_list = mesh.face_matrix()
print("_______SURFACE RECONSTRUCTED WITH POISSON_______")
print("vertices",vertices.shape,vertices)
print("triangle_list",triangle_list.shape,triangle_list)

# MESHFIX
vertices, triangle_list = pymeshfix.clean_from_arrays(vertices, triangle_list)
print("_______CLEANED VERTICES AND TRIANGLES_______")
print("vertices",vertices.shape,vertices)
print("triangle_list",triangle_list.shape,triangle_list)

# QUADRIC EDGE COLLAPSE DECIMATION
mesh=pymeshlab.Mesh(vertices, triangle_list)
ms = pymeshlab.MeshSet()
ms.add_mesh(mesh)
ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=2000, preservenormal=True)
mesh = ms.current_mesh()
vertices = mesh.vertex_matrix()
triangle_list = mesh.face_matrix()
print("_______FILTERED WITH QUADRATIC EDGE COLLAPSE_______")
print("vertices",vertices.shape,vertices)
print("triangle_list",triangle_list.shape,triangle_list)

# VISUALIZE
scene = trimesh.Scene()
mesh = trimesh.Trimesh(vertices=vertices, faces=triangle_list)
scene.add_geometry(mesh)
scene.show()

