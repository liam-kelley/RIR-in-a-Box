import numpy as np
import random
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
import pyvista as pv
from math import sqrt
import glob
import os

'''
This code could eventually be upgraded for rooms with more complex shapes than just shoeboxes.
'''

def check_rectangle(corners):
    polygon = Polygon(corners)
    if not (polygon.is_valid and polygon.is_rectangle):
        raise TypeError, "corners don't make a valid rectangle"

def rectangle_area(corners):
    polygon = Polygon(corners)
    return polygon.area

def generate_random_point_on_rectangle(corners : list, normal_max_offset : float):
    # Extract three corners to define the rectangle's plane
    p1, p2, p3 = corners[:3]

    # Compute edge vectors
    edge1 = np.array(p2) - np.array(p1)
    edge2 = np.array(p3) - np.array(p1)
    
    # Compute the normal vector as the cross product of edge1 and edge2
    normal_vector = np.cross(edge1, edge2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector

    # Compute area vectors
    width_vector = edge1
    height_vector = edge2
    
    # Generate random coefficients for width and height within the rectangle
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)

    # Calculate a random point within the bounds of the rectangle
    random_point = np.array(p1) + u * width_vector + v * height_vector

    # Add a small random offset along the normal vector
    random_point += normal_vector * np.random.uniform(-normal_max_offset, normal_max_offset)

    return random_point

def generate_a_wall_pointcloud(corners, normal_max_offset, mesh_nodes_per_m2):
    check_rectangle(corners)
    number_of_points_on_wall = rectangle_area(corners) * mesh_nodes_per_m2
    point_cloud = []
    for i in range(number_of_points_on_wall): # TODO parallelize this
        point_cloud.append(generate_random_point_on_rectangle(corners,normal_max_offset))
    return point_cloud

def get_sbox_corners(sbox_dim):
    length, width, height = sbox_dim
    # Define the corners of the shoebox
    floor_corners = [
        np.array([0, 0, 0]),
        np.array([length, 0, 0]),
        np.array([0, width, 0]),
        np.array([length, width, 0])
    ]
    
    ceiling_corners = [
        np.array([0, 0, height]),
        np.array([length, 0, height]),
        np.array([0, width, height]),
        np.array([length, width, height])
    ]

    left_wall_corners = [
        np.array([0, 0, 0]),
        np.array([0, 0, height]),
        np.array([0, width, 0]),
        np.array([0, width, height])
    ]

    right_wall_corners = [
        np.array([length, 0, 0]),
        np.array([length, 0, height]),
        np.array([length, width, 0]),
        np.array([length, width, height])
    ]

    back_wall_corners = [
        np.array([0, 0, 0]),
        np.array([0, 0, height]),
        np.array([length, 0, 0]),
        np.array([length, 0, height])
    ]

    front_wall_corners = [
        np.array([0, width, 0]),
        np.array([0, width, height]),
        np.array([length, width, 0]),
        np.array([length, width, height])
    ]

    # Collect all corners into a dictionary
    wall_corners = {
        'floor': floor_corners,
        'ceiling': ceiling_corners,
        'left_wall': left_wall_corners,
        'right_wall': right_wall_corners,
        'back_wall': back_wall_corners,
        'front_wall': front_wall_corners
    }

    return wall_corners


def get_shoebox_mesh(config : dict, sbox_dim) -> pv.PolyData:
    normal_max_offset = random.uniform(config['mesh_node_max_normal_random_offset_min'],config['mesh_node_max_normal_random_offset_max'])
    mesh_nodes_per_m2 = random.uniform(config['mesh_nodes_per_m2_min'],config['mesh_nodes_per_m2_max'])

    # Get the pointcloud for each wall
    wall_names=['floor','ceiling','left_wall','right_wall','back_wall','front_wall']
    wall_corners = get_sbox_corners(sbox_dim)
    wall_pointclouds = {}
    for name in wall_names:
        wall_pointclouds[name] = generate_a_wall_pointcloud(wall_corners[name], normal_max_offset, mesh_nodes_per_m2)
    
    # combine the pointclouds
    complete_pointcloud=[]
    for name in wall_names:
        complete_pointcloud.extend(wall_pointclouds[name])
    complete_pointcloud = np.stack(complete_pointcloud)
    
    # Reconstruct the whole surface
    pv_pointcloud = pv.PolyData(complete_pointcloud)
    surface = pv_pointcloud.reconstruct_surface(sample_spacing=sqrt(mesh_nodes_per_m2)*2)
    
    return surface # , normal_max_offset , mesh_nodes_per_m2


def save_shoebox_mesh(mesh : pv.PolyData):
    mesh_names=glob.glob("datasets/SBAM_Dataset/meshes/mesh_*.wav")
    mesh_names.sort()

    if mesh_names!=[]: # if there are already some meshs saved, write at the next index
        index=int(mesh_names[-1].split("_")[-1].split(".")[0]) + 1
        mesh_file_name = "datasets/SBAM_Dataset/meshes/mesh_{:06}".format(index)+".obj"
    else: # initialize
        index=0
        mesh_file_name = "datasets/SBAM_Dataset/meshes/mesh_000000.obj"
    
    if not os.path.exists(os.path.dirname(mesh_file_name)):
        os.makedirs(os.path.dirname(mesh_file_name))

    mesh.save(mesh_file_name)
    print("Mesh saved as " + mesh_file_name)
