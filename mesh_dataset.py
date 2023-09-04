import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import pickle
from torch.utils.data import Dataset
import torch
import pandas as pd
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from LKLogger import LKLogger

from shapely.geometry import Polygon
import pyroomacoustics as pra

from complex_room_dataset import get_an_inside_point, get_an_inside_array, check_points_polygon_distances,\
                                    check_direct_line_of_sight, convert_lists_to_str
import random
import glob
from scipy.io.wavfile import write
from torch_geometric.data import Data
from torch_geometric.data import Batch

def generate_random_point_on_rectangle(center_of_rectangle_coords, normal_vector, normal_max_offset):
    random_point = center_of_rectangle_coords + np.random.uniform(-center_of_rectangle_coords, center_of_rectangle_coords, size=3)
    random_point -= normal_vector * np.dot(normal_vector, random_point - center_of_rectangle_coords)
    random_point += normal_vector * np.random.uniform(-normal_max_offset, normal_max_offset)
    return random_point

def generate_random_point_on_a_segment(origin, vector):
    random_point = origin + vector*np.random.uniform(0,1)
    return random_point

def create_shoebox_mesh(max_offset, num_points_per_wall, num_points_per_edge, room_size=np.array([1.1, 1.1, 1.1]), by_m2=True):
    # idk why i did it with 2s but now to fix that bug i need to do
    room_size = room_size / 2

    wall_names=['floor','ceiling','left_wall','right_wall','back_wall','front_wall']
    
    walls = {
        'floor': (np.array([1, 1, 0]), np.array([0, 0, 1])),
        'ceiling': (np.array([1, 1, 2]), np.array([0, 0, -1])),
        'left_wall': (np.array([0, 1, 1]), np.array([1, 0, 0])),
        'right_wall': (np.array([2, 1, 1]), np.array([-1, 0, 0])),
        'back_wall': (np.array([1, 2, 1]), np.array([0, -1, 0])),
        'front_wall': (np.array([1, 0, 1]), np.array([0, 1, 0]))
    }
    
    origin000 = np.array([0, 0, 0])*room_size
    origin200 = np.array([2, 0, 0])*room_size
    origin020= np.array([0, 2, 0])*room_size
    origin002 = np.array([0, 0, 2])*room_size
    origin222 = np.array([2, 2, 2])*room_size

    vector200 = np.array([2, 0, 0])*room_size
    vector020 = np.array([0, 2, 0])*room_size
    vector002 = np.array([0, 0, 2])*room_size

    edge_vertice_list=[[generate_random_point_on_a_segment(origin000,vector200) for _ in range(num_points_per_edge)] + [origin000,origin000+vector200],
                       [generate_random_point_on_a_segment(origin000,vector020) for _ in range(num_points_per_edge)] + [origin000,origin000+vector020],
                       [generate_random_point_on_a_segment(origin000,vector002) for _ in range(num_points_per_edge)] + [origin000,origin000+vector002],

                       [generate_random_point_on_a_segment(origin200,vector020) for _ in range(num_points_per_edge)] + [origin200,origin200+vector020],
                       [generate_random_point_on_a_segment(origin200,vector002) for _ in range(num_points_per_edge)] + [origin200,origin200+vector002],

                       [generate_random_point_on_a_segment(origin020,vector200) for _ in range(num_points_per_edge)] + [origin020,origin020+vector200],
                       [generate_random_point_on_a_segment(origin020,vector002) for _ in range(num_points_per_edge)] + [origin020,origin020+vector002],
                       
                       [generate_random_point_on_a_segment(origin002,vector200) for _ in range(num_points_per_edge)] + [origin002,origin002+vector200],
                       [generate_random_point_on_a_segment(origin002,vector020) for _ in range(num_points_per_edge)] + [origin002,origin002+vector020],
                       
                       [generate_random_point_on_a_segment(origin222,-vector200) for _ in range(num_points_per_edge)] + [origin222,origin222-vector200],
                       [generate_random_point_on_a_segment(origin222,-vector020) for _ in range(num_points_per_edge)] + [origin222,origin222-vector020],
                       [generate_random_point_on_a_segment(origin222,-vector002) for _ in range(num_points_per_edge)] + [origin222,origin222-vector002]]
    
    wall_vertices = { # please draw it out cause i can't explain it
        'floor': edge_vertice_list[0] + edge_vertice_list[1] + edge_vertice_list[3] + edge_vertice_list[5],
        'ceiling': edge_vertice_list[7] + edge_vertice_list[8] + edge_vertice_list[9] + edge_vertice_list[10],
        'left_wall': edge_vertice_list[1] + edge_vertice_list[2] + edge_vertice_list[8] + edge_vertice_list[6],
        'right_wall': edge_vertice_list[3] + edge_vertice_list[4] + edge_vertice_list[10] + edge_vertice_list[11],
        'back_wall': edge_vertice_list[5] + edge_vertice_list[6] + edge_vertice_list[9] + edge_vertice_list[11],
        'front_wall': edge_vertice_list[0] + edge_vertice_list[2] + edge_vertice_list[7] + edge_vertice_list[4],
    }

    wall_tri = {
        'floor': None,
        'ceiling': None,
        'left_wall': None,
        'right_wall': None,
        'back_wall': None,
        'front_wall': None,
    }

    # make the proper amount of points per wall
    if type(num_points_per_wall) in [int, list]:
        if type(num_points_per_wall) == int and by_m2==True: # this option does points per m2 rather than points overall on a wall, and it should be preferred!
            num_points_per_wall={
                'floor': int(room_size[0]*room_size[1]*num_points_per_wall),
                'ceiling': int(room_size[0]*room_size[1]*num_points_per_wall),
                'left_wall': int(room_size[1]*room_size[2]*num_points_per_wall),
                'right_wall': int(room_size[1]*room_size[2]*num_points_per_wall),
                'back_wall': int(room_size[0]*room_size[2]*num_points_per_wall),
                'front_wall': int(room_size[0]*room_size[2]*num_points_per_wall)
            }
        else:
            num_points_per_wall = dict.fromkeys(wall_names, num_points_per_wall)
    else:
        assert(type(num_points_per_wall) == dict)
        assert(num_points_per_wall.keys()==wall_names)

    for wall in wall_names:
        center_of_wall_coords, normal_vector = walls[wall]
        # sample some points on the wall
        for _ in range(num_points_per_wall[wall]):
            wall_vertices[wall].append(generate_random_point_on_rectangle(np.multiply(center_of_wall_coords,room_size), normal_vector, max_offset))
        wall_vertices[wall]=np.asarray(wall_vertices[wall])

        # Compute delaunay triangulation on wall
        if wall == 'floor' or  wall == 'ceiling':
            wall_tri[wall] = Delaunay(wall_vertices[wall][:,:2]) #tri is calculated with only first two dimensions because fck you
        elif wall == 'left_wall' or  wall == 'right_wall':
            wall_tri[wall] = Delaunay(wall_vertices[wall][:,1:3])
        elif wall == 'front_wall' or  wall == 'back_wall':
            wall_tri[wall] = Delaunay(wall_vertices[wall][:,0:3:2])

    return wall_vertices , wall_tri

def triangle_list_to_edge_index(triangle_list):
    """
    Convert a triangle list to an edge index format.

    Args:
    - triangle_list (n_triangles, 3 vertex indices): Each inner list contains 3 vertex indices representing a triangle.

    Returns:
    - edge_index (2 * n_edges): Two lists, one for source nodes and one for target nodes.
    """
    
    edge_set = set()  # Use a set to avoid duplicate edges
    
    for triangle in triangle_list:
        # Ensure each edge is stored in a consistent order (smaller index first)
        edges = [
            (min(triangle[0], triangle[1]), max(triangle[0], triangle[1])), #edge 0-1
            (min(triangle[1], triangle[2]), max(triangle[1], triangle[2])), #edge 1-2
            (min(triangle[0], triangle[2]), max(triangle[0], triangle[2]))  #edge 0-2
        ]
        
        for edge in edges:
            edge_set.add(edge)
    
    # Convert the set of edges to the edge index format
    source_nodes, target_nodes = zip(*list(edge_set))
    
    return np.array([list(source_nodes), list(target_nodes)])   

def plot_mesh_from_triangle_list(wall_vertices , wall_tri):
    '''*from edge index is slightly more up to date'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for wall in ['floor', 'left_wall']: # wall_vertices.keys():
        ax.scatter([v[0] for v in wall_vertices[wall]], [v[1] for v in wall_vertices[wall]], [v[2] for v in wall_vertices[wall]], label=wall)
        for triangle in wall_tri[wall].simplices:
            triangle = np.append(triangle, triangle[0]) # Here we cycle back to the first coordinate for a full triangle plot
            ax.plot([wall_vertices[wall][vertex_index][0] for vertex_index in triangle],
                    [wall_vertices[wall][vertex_index][1] for vertex_index in triangle],
                    [wall_vertices[wall][vertex_index][2] for vertex_index in triangle],
                    color='black')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def plot_mesh_from_edge_index(wall_vertices , wall_edge_index, show=True):
    '''
    Manages both having a dicts of different meshes and having a single tensor mesh.
    '''
    print("plotting mesh from edge index")
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    
    if type(wall_vertices) == dict and type(wall_edge_index) == dict:
        for wall in wall_vertices.keys(): #['floor', 'left_wall']:
            ax.scatter([v[0] for v in wall_vertices[wall]], [v[1] for v in wall_vertices[wall]], [v[2] for v in wall_vertices[wall]], label=wall)
            for edge in np.transpose(wall_edge_index[wall]):
                ax.plot([wall_vertices[wall][vertex_index][0] for vertex_index in edge],
                        [wall_vertices[wall][vertex_index][1] for vertex_index in edge],
                        [wall_vertices[wall][vertex_index][2] for vertex_index in edge],
                        color='black')
    else:
        assert(type(wall_vertices) == torch.Tensor and type(wall_edge_index) == torch.Tensor)
        wall_vertices = wall_vertices.detach().numpy()
        wall_edge_index = wall_edge_index.detach().numpy().astype(int)
        ax.scatter([v[0] for v in wall_vertices], [v[1] for v in wall_vertices], [v[2] for v in wall_vertices], label='vertices')
        for edge in np.transpose(wall_edge_index):
            ax.plot([wall_vertices[vertex_index][0] for vertex_index in edge],
                    [wall_vertices[vertex_index][1] for vertex_index in edge],
                    [wall_vertices[vertex_index][2] for vertex_index in edge],
                    color='black')
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Mesh used as input for network')
    
    if show : plt.show()

def plot_mesh_from_edge_index_batch(x_batch , edge_index_batch, batch_indexes, show=True):
    '''
    graph batches are somwhat annoying, this unbatches a graph batch and to plot it.
    '''
    batch_indexes_0 = [elem for elem in batch_indexes.detach().numpy() if elem == 0]
    x_batch_0 = x_batch.detach()[:len(batch_indexes_0)]
    edge_index_batch_0 = [torch.unsqueeze(elem, dim=0) for elem in torch.transpose(edge_index_batch.detach(),0,1) if elem[0] < len(x_batch_0) and elem[1] < len(x_batch_0)]
    edge_index_batch_0 = torch.cat(edge_index_batch_0,dim=0)
    edge_index_batch_0 = torch.transpose(edge_index_batch_0,0,1)
    assert(len(x_batch_0) == torch.max(edge_index_batch_0)+1)
    plot_mesh_from_edge_index(x_batch_0, edge_index_batch_0, show=show)

def shoebox_mesh_dataset_generate_rir_and_mesh(options, log_row):
    '''
    Create a new random shoebox room.
    Obtains useful info from it.
    Computes RIR and saves it.
    Generate its mesh and save it.
    Prepare all pertinent data to be saved in a pandas dataframe/LKLogger.
    '''
    # Constant
    NUMBER_OF_MICS = 1
    
    # make sure that src/mic min distances are compatible with room dimensions.
    for i in range(3):
        if 1.66*options['room_shoebox_dimensions_range'][0][i] < options['min_source_mic_distance'] + options['min_source_wall_distance']*2:
            options['min_source_mic_distance'] = 0.3*options['room_shoebox_dimensions_range'][0][i]
            options['min_source_wall_distance'] = 0.15*options['room_shoebox_dimensions_range'][0][i]
    
    # Make a random shoebox within room shoebox dimensions range
    room_dim = np.random.uniform(options['room_shoebox_dimensions_range'][0],options['room_shoebox_dimensions_range'][1], size=3)
    counter=0
    while room_dim[0]*room_dim[1]*room_dim[2] < options['room_min_volume'] or room_dim[0]*room_dim[1] < options['room_min_floor_area']:
        room_dim = np.random.uniform(options['room_shoebox_dimensions_range'][0],options['room_shoebox_dimensions_range'][1], size=3)
        counter+=1
        if counter>100:
            print("couldn't find a room with enough volume or floor area")
            return log_row
    
    room_polygon=Polygon([[0,0], [room_dim[0],0], [room_dim[0],room_dim[1]], [0, room_dim[1]] ] )

    # Extract the vertices that define the perimeter of the polygon
    xx, yy = room_polygon.exterior.coords.xy
    x=xx.tolist()
    y=yy.tolist()
    vertex_arr=np.array([x[:-1],y[:-1]])

    # We invert Sabine's formula to obtain the parameters for the ISM simulator # we could actually do this to get an rt60
    # e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create a shoebox pra room
    wall_material = pra.Material(energy_absorption=options['rir_energy_absorption'], scattering=options['rir_scattering'])
    room = pra.ShoeBox(
        room_dim,
        materials=wall_material,
        fs=options['rir_sample_rate'],
        max_order=options['rir_max_order'],
    )
    # room.extrude(room_dim[2]) #shoebox is already 3D

    # put source and mic inside the room once
    source_pos = get_an_inside_point(room, shoebox=True)
    mic_array = get_an_inside_array(room, n=NUMBER_OF_MICS, shoebox=True)
    source_height=options['min_source_wall_distance'] + (room_dim[2]-2*options['min_source_wall_distance'])*random.random() 
    mic_height=options['min_source_wall_distance'] + (room_dim[2]-2*options['min_source_wall_distance'])*random.random() 
    source_pos += [source_height]
    mic_array=np.transpose(mic_array)
    mic_array=np.transpose(np.asarray([mic_array[0].tolist()+[mic_height]]))

    # Ensure they're far away enough from each other and far enough from walls
    while np.linalg.norm(source_pos - mic_array) < options['min_source_mic_distance'] or \
        (not check_points_polygon_distances(room_polygon, [source_pos[:2],mic_array[:2]], options['min_source_wall_distance'])) :
        # put source and mic inside the room again until prerequisites are met
        source_pos = get_an_inside_point(room, shoebox=True)
        mic_array = get_an_inside_array(room, n=NUMBER_OF_MICS, shoebox=True)
        source_height=options['min_source_wall_distance'] + (room_dim[2]-2*options['min_source_wall_distance'])*random.random() 
        mic_height=options['min_source_wall_distance'] + (room_dim[2]-2*options['min_source_wall_distance'])*random.random() 
        source_pos += [source_height]
        mic_array=np.transpose(mic_array)
        mic_array=np.transpose(np.asarray([mic_array[0].tolist()+[mic_height]]))
    
    # Plot the room geometry
    if options['plot']:
        room2D = pra.room.Room.from_corners(vertex_arr)
        room2D.add_source(source_pos[:2])
        room2D.add_microphone_array(mic_array[:2])
        fig, ax = room2D.plot()
        # ax.set_xlim(-6, 6)
        # ax.set_ylim(-6, 6)
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        ax.grid(True, ls=':', alpha=0.5)
        ax.text(1, 9, 'room height = '+str(room_dim[2]), bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Room used for pyroom label calculation")

    # Compute RIR    
    room.add_source(source_pos)
    room.add_microphone_array(mic_array)
    room.compute_rir()
    rir=room.rir[0][0]

    # Save RIR
    if options['save']:
        rir_names=glob.glob("meshdataset/rirs/rir_**.wav")
        rir_names.sort()
        if rir_names!=[]: # if there are already some rirs saved, write at the next index
            index=int(rir_names[-1][rir_names[-1].index('_')+1:rir_names[-1][1:].index('.')+1]) # get the last index, converted to int
            rir_file_name = f"meshdataset/rirs/rir_"+'{:06}'.format(index+1)+".wav"
        else:
            index=0
            rir_file_name = "meshdataset/rirs/rir_000000.wav"
        write(rir_file_name, options['rir_sample_rate'], rir)
        print("RIR saved as " + rir_file_name)
    else:
        rir_file_name=""
    
    # plot the RIR between mic 0 and source 0
    if options['plot']:
        plt.figure()
        plt.title("RIR : " + rir_file_name)
        plt.plot(np.arange(len(rir))/options['rir_sample_rate'], rir, label="rir")
        plt.legend()
        plt.grid(True, ls=':', alpha=0.5)
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")

    # Convert lists to strings for saving the vertices in dataframe later
    vertex_arr_str, mic_array_str, source_pos_str = convert_lists_to_str(vertex_arr, mic_array, source_pos)
    log_row['room_vertex_arr_str']=vertex_arr_str
    log_row['rir_mic_array_str']=mic_array_str
    log_row['rir_source_pos_str']=source_pos_str

    # Create mesh
    wall_vertices , wall_tri = create_shoebox_mesh(options['mesh_max_offset'], options['mesh_points_per_m2'], options['mesh_points_per_edge'], room_dim) # room dim is a numpy array of float 64s
    
    # Convert triangle lists to edge_indexes
    wall_edge_index = {}
    for wall in wall_tri.keys():
        wall_edge_index[wall] = triangle_list_to_edge_index(wall_tri[wall].simplices)
    
    # plot the mesh
    if options['plot']:
        # plot_mesh_from_triangle_list(wall_vertices , wall_tri)
        plot_mesh_from_edge_index(wall_vertices , wall_edge_index)

    # Save mesh
    if options['save']:
        mesh_names=glob.glob("meshdataset/meshes/mesh_**.meshpkl")
        mesh_names.sort()
        if mesh_names!=[]: # if there are already some rirs saved, write at the next index
            index=int(mesh_names[-1][mesh_names[-1].index('_')+1:mesh_names[-1][1:].index('.')+1]) # get the last index, converted to int
            mesh_file_name = f"meshdataset/meshes/mesh_"+'{:06}'.format(index+1)+".meshpkl"
        else:
            index=0
            mesh_file_name = "meshdataset/meshes/mesh_000000.meshpkl"
        with open(mesh_file_name, 'wb') as file:
            pickle.dump((wall_vertices,wall_edge_index),file)
        print("mesh saved as " + mesh_file_name)

    # log_rows
    log_row['mesh_file_name'] = mesh_file_name
    log_row['mesh_points_per_m2'] = options['mesh_points_per_m2']
    log_row['mesh_max_offset']= options['mesh_max_offset']
    log_row["mesh_points_per_edge"]= options['mesh_points_per_edge']
    log_row['room_n_walls']=4
    log_row['room_height']=room_dim[2]
    log_row['room_shoebox']=True
    log_row['room_area']=room_polygon.area
    log_row['rir_energy_absorption']=options['rir_energy_absorption']
    log_row['rir_scattering']=options['rir_scattering']
    log_row['rir_sample_rate']=options['rir_sample_rate']
    log_row['rir_max_order']=options['rir_max_order']
    log_row['rir_line_of_sight']=check_direct_line_of_sight(room_polygon,source_pos,mic_array)
    log_row['rir_file_name']=rir_file_name

    return (log_row)

def shoebox_mesh_dataset_generation(dataset_name="meshdataset/shoebox_mesh_dataset.csv", iterations=10, options={}):
    '''
    Use this function to generate more data points for the dataset

    1. Generate a random shoebox mesh and compute its shoebox rir
    2. save the meshes in pkl files and save their rir's in wavs
    3. log the generation information in a log file using my custom logger class.

    As far as weight goes, PER datapoint you're looking at about
    a 150-300kb mesh when mesh_points_per_m2=200
    and a 50 - 115kb rir
    '''
    empty_log_row={
            'mesh_file_name' : None,
            'mesh_points_per_m2' : None,
            'mesh_max_offset' : None,
            'mesh_points_per_edge' : None,
            'room_n_walls' : None,
            'room_vertex_arr_str' : None,
            'room_height' : None,
            'room_shoebox' : None,
            'room_area' : None,
            'rir_energy_absorption' : None,
            'rir_scattering' : None,
            'rir_sample_rate' : None,
            'rir_max_order' : None,
            'rir_mic_array_str' : None,
            'rir_source_pos_str' : None,
            'rir_line_of_sight' : None,
            'rir_file_name' : None,
    }

    shoebox_mesh_generation_default_options={
        'mesh_points_per_m2' :            10,   # reality should be 100-400 ?
        'mesh_max_offset' :               0.1,  # could do more
        'mesh_points_per_edge' :          10,
        'room_shoebox_dimensions_range' : [[1.5, 1.5, 1],[7.5, 7.5, 3.5]],
        'room_min_floor_area' :           4,
        'room_min_volume' :               10,
        'rir_energy_absorption' :         0.1,
        'rir_scattering' :                0.1,
        'rir_sample_rate' :               48000,
        'rir_max_order' :                 15,
        'min_source_mic_distance' :       1,
        'min_source_wall_distance' :      1,
        'plot' :                          False,
        'save' :                          True,
        }

    # set defaults where needs be
    for key in list(set(shoebox_mesh_generation_default_options.keys()).difference(set(options.keys()))):
        options[key]=shoebox_mesh_generation_default_options[key]

    # init logger
    logger=LKLogger(filename=dataset_name, columns_for_a_new_log_file = empty_log_row.keys())

    # Iterate generation
    for _ in range(iterations):
        log_row=shoebox_mesh_dataset_generate_rir_and_mesh(options, empty_log_row)
        logger.add_line_to_log(log_row)

class GraphDataset(Dataset):
    '''
    An example raw datapoint from the csv file :
    
    mesh_file_name                         meshdataset/meshes/mesh_000.pkl (contains (wall_x, wall_edge_index))
    mesh_points_per_m2                                                 200
    mesh_max_offset                                                    0.1
    mesh_points_per_edge                                                10 (should be per meter)
    room_n_walls                                                         4 (doesn't include floor and ceiling)
    room_vertex_arr_str  [[-0.10601230167115427 1.9224084403891974 -1.9...
    room_height                                                        2.2
    room_shoebox                                                      True
    room_area                                                    30.580212
    rir_energy_absorption                                              0.1
    rir_scattering                                                     0.1
    rir_sample_rate                                                  48000
    rir_max_order                                                       15
    rir_mic_array_str    [[-0.7507980346516723 ][-4.886256689504428 ][1...
    rir_source_pos_str      [-0.34120049727709434 -1.940370924558073 1.1 ]
    rir_line_of_sight                                                 True
    rir_file_name                             meshdataset/rirs/rir_000.wav
    '''
    def __init__(self, csv_file="meshdataset/shoebox_mesh_dataset.csv", filter={}, pad_in_collate=True):
        self.pad_in_collate=pad_in_collate #This option should be on. You pad when loading in the dataloader, not on preprocessing
        self.csv_file=csv_file
        self.sample_rate=None
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print('csv file "', csv_file ,'" not found, generate small default dataset? (y/n)')
            if input()=='y':
                print("generating dataset...")
                shoebox_mesh_dataset_generation(iterations=10, dataset_name=csv_file)
                print("dataset generated, loading csv...")
                self.data = pd.read_csv(csv_file)
            else:
                print("exiting...")
                exit()
        print('Graph Dataset', csv_file ,'loaded')
        self.max_length_rir = 0
        self.preprocess(filter)
        print("Graph Dataset preprocessing done.")

    def preprocess(self, filter):
        # preprocess Booleans
        self.data['rir_line_of_sight'] = self.data['rir_line_of_sight'].apply(lambda x: 1 if x else 0)
        self.data['room_shoebox'] = self.data['room_shoebox'].apply(lambda x: 1 if x else 0)

        # Filtering
        for key, value in filter.items():
            if type(value) == list:
                if len(value) == 2: self.data=self.data[value[0] <= self.data[key] <= value[1]]
                else: raise ValueError('filter value list must be of length 2')
            self.data=self.data[self.data[key]==value]
        self.data.reset_index(drop=True, inplace=True)

        # Strip and split the string representations of coordinates
        self.data['vertex_arr'] = self.data['room_vertex_arr_str'].apply(lambda x: x.strip('[] ').replace("]","").replace("[","").split(' '))
        self.data['mic_array'] = self.data['rir_mic_array_str'].apply(lambda x: x.strip('[] ').replace("]","").replace("[","").split(' '))
        self.data['source_pos'] = self.data['rir_source_pos_str'].apply(lambda x: x.strip('[] ').replace("]","").replace("[","").split(' '))

        # Convert the strings to floats
        self.data['vertex_arr'] = self.data['vertex_arr'].apply(lambda x: [float(i) for i in x])
        self.data['mic_array'] = self.data['mic_array'].apply(lambda x: [float(i) for i in x])
        self.data['source_pos'] = self.data['source_pos'].apply(lambda x: [float(i) for i in x])

        # Further split the 'vertex_arr' into 2D array based on 'room_n_walls'
        for i in range(len(self.data)):
            assert(len(self.data.at[i, 'vertex_arr']) == 2*self.data.at[i, 'room_n_walls'])
            self.data.at[i, 'vertex_arr'] = [self.data.at[i, 'vertex_arr'][ : self.data.at[i, 'room_n_walls']],
                                             self.data.at[i, 'vertex_arr'][self.data.at[i, 'room_n_walls'] : ]]
        
        # These columns are no longer needed
        self.data = self.data.drop(["room_vertex_arr_str","rir_mic_array_str","rir_source_pos_str"], axis=1)

        # Drop columns that won't be used
        drop_columns=['mesh_points_per_m2','mesh_max_offset','mesh_points_per_edge',
                      'room_n_walls','room_area', # 'room_height', 'room_shoebox'
                      'rir_sample_rate','rir_max_order','rir_line_of_sight']
        # drop_columns+=['rir_energy_absorption','rir_scattering']
        self.data = self.data.drop(drop_columns, axis=1)

        # Initialize max_length_rir (this takes a long time and is not necessary if you don't pad the RIRs but...)
        if not self.pad_in_collate : self.update_max_length_of_label_rirs()
        else : # still need to update sample_rate
            self.sample_rate, _ = read(self.data['rir_file_name'].iloc[-1])

        print("length of dataset",len(self.data))

    def update_max_length_of_label_rirs(self):
        '''
        Initializes self.max_length_rir to the length of the longest RIR in the dataset.
        This is useful for padding the RIRs to the same length for the dataloader.
        Re-run this if you add to the dataset.
        '''
        for rir_file_name in self.data['rir_file_name']:
            self.sample_rate, label_rir = read(rir_file_name)
            if len(label_rir) > self.max_length_rir:
                self.max_length_rir = len(label_rir)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        '''
        Calling GraphDataset[index] returns a tuple of tensors:
        (x, edge_index, label_rir_tensor, label_origin, mic_pos_tensor, source_pos_tensor,
                absorption_tensor ,scattering_tensor)

        - x is the feature tensor of shape (n_nodes, n_features)
        - edge_index is the edge index tensor of shape (2, n_edges)
        - The label_rir_tensor is a tensor of shape (rir_length, 1)
        It can be used as an intermediate label for the encoder.
        - The label_origin is an integer representing the index of the peak of the RIR.
        - mic_tensor is a tensor of shape (3) representing the position of the microphone used for label RIR computation.
        - source_tensor is a tensor of shape (3) representing the position of the source used for label RIR computation.
        - absorption_tensor is a tensor of shape (1)
        - scattering_tensor is a tensor of shape (1)
        '''
        if torch.is_tensor(index):
            index = index.tolist()
        df = self.data.iloc[index]

        ################ Get NODE FEATURES (X) and EDGE_INDEX ####################################
        with open(df['mesh_file_name'], 'rb') as file:
            wall_features, wall_edge_index = pickle.load(file) # these will be numpy arrays

        wall_names=wall_features.keys()

        n_nodes={}
        for wall, features in wall_features.items():
            n_nodes[wall] = features.shape[0]

        # add label 0 for floor
        i=0
        label_ones = np.ones((n_nodes['floor'], 1))
        wall_features['floor'] = np.concatenate((wall_features['floor'], i*label_ones), axis=1)
        index_translation = n_nodes['floor'] # when concatenated, indexes will be higher!

        # add label 1 for ceiling
        i+=1
        label_ones = np.ones((n_nodes['ceiling'], 1))
        wall_features['ceiling'] = np.concatenate((wall_features['ceiling'], i*label_ones), axis=1)
        # properly translate the indexes in preparation for concatenation
        wall_edge_index['ceiling'] = wall_edge_index['ceiling'] + index_translation*np.ones(wall_edge_index['ceiling'].shape)
        index_translation += n_nodes['ceiling']

        # similarly, add a different label for all other walls/platforms
        # and properly translate the indexes in preparation for concatenation
        for wall in wall_names:
            if wall=='floor' or wall=='ceiling':
                continue
            i+=1
            label_ones = np.ones((n_nodes[wall], 1))
            wall_features[wall] = np.concatenate((wall_features[wall], i*label_ones), axis=1)
            wall_edge_index[wall] = wall_edge_index[wall] + index_translation*np.ones(wall_edge_index[wall].shape)
            index_translation += n_nodes[wall]

        # concatenate all the walls/platforms into two big tensors
        x = wall_features['floor']
        edge_index = wall_edge_index['floor']
        for wall in wall_names:
            if wall=='floor':
                continue
            x = np.concatenate((x, wall_features[wall]),axis=0)
            edge_index = np.concatenate((edge_index,wall_edge_index[wall]), axis=1)

        # sanity checks
        assert(x.shape[0] == sum(n_nodes.values()))
        assert(x.shape[1] == 4)
        assert(edge_index.shape[0]==2)
        assert(edge_index.shape[1]== sum([wall_edge_index[wall].shape[1] for wall in wall_names]))

        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.float32)

        ################ Get LABEL RIR and LABEL RIR ORIGIN ####################################

        self.sample_rate, label_rir = read(df['rir_file_name'])
        label_rir_tensor = torch.tensor(label_rir, dtype=torch.float32)
        # find origin of RIR
        peak_indexes, _ = find_peaks(label_rir/np.max(label_rir),height=0.3)
        label_origin = peak_indexes[0]
        # pad tensor to max_length_rir for dataloader
        if not self.pad_in_collate : label_rir_tensor = torch.nn.functional.pad(label_rir_tensor, (0, self.max_length_rir - len(label_rir_tensor)))

        ################ Get LABEL ROOM_DIMENSIONS ####################################

        if df['room_shoebox']:
            room_dim = [max(df['vertex_arr'][0]),max(df['vertex_arr'][1]), df['room_height']]


        ################ Get MIC POS, SRC POS, ENERGY ABS, SCATTERING ####################################

        label_origin_tensor = torch.tensor(label_origin, dtype=torch.float32)
        if df['room_shoebox']: room_dim_tensor = torch.tensor(room_dim, dtype=torch.float32)
        else : room_dim_tensor = torch.tensor([0,0,0], dtype=torch.float32)
        mic_pos_tensor = torch.tensor(df['mic_array'], dtype=torch.float32)
        source_pos_tensor = torch.tensor(df['source_pos'], dtype=torch.float32)
        absorption_tensor = torch.tensor(df['rir_energy_absorption'], dtype=torch.float32)
        scattering_tensor = torch.tensor(df['rir_scattering'], dtype=torch.float32)

        # I need to create a batch vector, but the torch geometric dataloader doesn't work for my use so
        # I return a tuple of elements that can be used to create a batch vectors with this dataset custom collate_fn
        return (Data(x=x, edge_index=edge_index.long().contiguous()), 
                label_rir_tensor, 
                label_origin_tensor,
                room_dim_tensor, 
                mic_pos_tensor, 
                source_pos_tensor,
                absorption_tensor,
                scattering_tensor)    

    @staticmethod
    def custom_collate_fn(list_of_tuples):
        '''
        Use this in a normal torch.utils.data.DataLoader to get a batch.
        returns (list_of_Data, rir_tensor_batch, list_of_origin, mic_tensor_batch, source_tensor_batch, absorption_tensor_batch, scattering_tensor_batch)
        '''
        data_list, label_rir_tensors, origin_tensors, room_dim_tensors, mic_pos_tensors, source_pos_tensors, absorption_tensors, scattering_tensors = zip(*list_of_tuples)

        # create a batch vector for the graph data.        
        graph_batch=Batch.from_data_list(data_list)

        # if theres any tensor which is not the same length
        label_rir_tensors = list(label_rir_tensors)
        if any([len(label_rir_tensors[0]) != len(label_rir_tensors[i]) for i in range(1,len(label_rir_tensors))]):
            # pad tensors to max length before stacking
            max_rir_length=max([len(label_rir_tensors[i]) for i in range(len(label_rir_tensors))])
            for i in range(len(label_rir_tensors)):
                label_rir_tensors[i] = torch.nn.functional.pad(label_rir_tensors[i], (0, max_rir_length - len(label_rir_tensors[i])))

        return (graph_batch.x, graph_batch.edge_index, graph_batch.batch,
                torch.stack(label_rir_tensors, dim=0), torch.stack(origin_tensors, dim=0),
                torch.stack(room_dim_tensors, dim=0), torch.stack(mic_pos_tensors, dim=0), torch.stack(source_pos_tensors, dim=0),
                torch.stack(absorption_tensors, dim=0), torch.stack(scattering_tensors, dim=0))

def main():
    '''
    Example use of the GraphDataset class and shoebox_mesh_dataset_generation function.

    shoebox_mesh_dataset_generation default options dictionnary. Feel free to change them with the options={} argument
        mesh_points_per_m2 :            10       # reality should be 100-400
        mesh_max_offset :               0.1      # could do more
        mesh_points_per_edge :          10
        room_shoebox_dimensions_range : [[1.5, 1.5, 1],[7.5, 7.5, 3.5]]    # [min, max]
        room_min_floor_area :           4
        room_min_volume :               10
        rir_energy_absorption :         0.1
        rir_scattering :                0.1
        rir_sample_rate :               48000
        rir_max_order :                 15
        min_source_mic_distance :       1
        min_source_wall_distance :      1
        plot :                          False
        save :                          True
    
    elements of the *dataset.csv that can be filtered_upon GraphDataset creation using the filter={} argument
        mesh_points_per_m2                                  int, or [int, int]
        mesh_max_offset                                                    0.1
        mesh_points_per_edge                                int, or [int, int] (should be per meter)
        room_n_walls                                        int, or [int, int] (doesn't include floor and ceiling)
        room_height                                             [float, float]
        room_shoebox                                                    1 or 0
        room_area                                               [float, float]
        rir_energy_absorption                         float, or [float, float] (should be between 0.0 and 1.0)
        rir_scattering                                float, or [float, float] (should be between 0.0 and 1.0)
        rir_sample_rate                                     int, or [int, int]
        rir_max_order                                       int, or [int, int]
        rir_line_of_sight                                               1 or 0 
    '''
    for i in range(180):
        offset=(random.random()/3)+0.05
        mesh_points_per_edge = random.randint(5, 20)
        mesh_points_per_m2 = random.randint(10, 200)
        rir_energy_absorption = random.choice([0.1,0.2,0.3,0.4,0.5])
        rir_scattering = random.choice([0.1,0.2,0.3,0.4,0.5])
        shoebox_mesh_dataset_generation(iterations=5, options={'plot':False,
                                                                'mesh_points_per_m2' : mesh_points_per_m2, #200
                                                                'mesh_max_offset' : offset,
                                                                'mesh_points_per_edge' : mesh_points_per_edge,
                                                                'rir_energy_absorption' : rir_energy_absorption,
                                                                'rir_scattering' : rir_scattering,})
    
    # dataset=GraphDataset("meshdataset/shoebox_mesh_dataset.csv", filter={})
    
    # graph_data, _ , _ ,_ ,_, _, _= dataset[-3]
    # plot_mesh_from_edge_index(graph_data['x'], graph_data['edge_index'])


if __name__ == '__main__':
    main()