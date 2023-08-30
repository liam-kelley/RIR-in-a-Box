from shapely.geometry import Polygon, LineString, Point
import math
import random
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile
import glob
from LKLogger import LKLogger
from torch.utils.data import Dataset
import torch
from scipy.signal import find_peaks

# Utility functions

def generate_random_room_polygon(min_sides, max_sides, min_length, max_length, min_area=3, vertex_line_distance_threshold=1):
    '''
    Generates and returns a random room polygon according to the given constraints.
    It has a minimum and maximum number of sides, a minimum and maximum total width and length.
    A minimim area is also given. And mixnimum distance between vertices and lines.
    '''
    num_sides = random.randint(min_sides, max_sides)
    vertices = []
    while not Polygon(vertices).is_valid \
                or vertices == [] \
                or Polygon(vertices).area < min_area \
                or not check_vertex_line_distances(Polygon(vertices), vertex_line_distance_threshold):
        
        vertices = quick_build_not_intersecting_vertices(num_sides, min_length, max_length)
        
    # print(check_vertex_line_distances(Polygon(vertices), vertex_line_distance_threshold))
    return Polygon(vertices)

def quick_build_not_intersecting_vertices(num_sides, min_length, max_length):
    '''
    There might be issues with this method with the final line, which might be intersecting.
    This is why it is a "quick" build. Additionnal constraints are needed to ensure that the
    final line is not intersecting and that the polygon is correct for our use.
    '''
    vertices = []
    for _ in range(num_sides):
        # Create new lines until a non-intersecting one is found
        valid_newline = False
        while not valid_newline:
            length = random.uniform(min_length, max_length)
            angle = random.uniform(0, 2 * math.pi)
            x = length * math.cos(angle)
            y = length * math.sin(angle)
            vertices.append((x, y))

            if len(vertices) > 2:
                # Check if the newly added line segment intersects with any of the previous line segments
                newline = LineString(vertices[-2:])
                for i in range(len(vertices) - 3):
                    prev_line = LineString(vertices[i:i + 2])
                    if newline.intersects(prev_line):
                        # If there is an intersection, remove the last vertex and try again
                        vertices.pop()
                        break
                else: valid_newline = True
            else: valid_newline = True
    return vertices

def check_vertex_line_distances(polygon, threshold):
    '''
    Checks that every vertex is at least a certain distance away from every other line segment.
    '''
    vertices = polygon.exterior.coords
    # for every line segment
    for i in range(len(vertices)):
        current_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % len(vertices)]
        line_segment = LineString([current_vertex, next_vertex])

        # Check if they're too close to any vertex
        for j in range(len(vertices)):
            checked_vertex = vertices[j]
            if checked_vertex == current_vertex or checked_vertex == next_vertex:
                continue  # Skip the current vertex

            distance = line_segment.distance(Point(checked_vertex))
            # print("line", line_segment)
            # print("point", Point(checked_vertex))
            # print("distance", distance)
            if distance < threshold:
                return False  # Found a vertex that is too close to a line segment
    return True  # All vertices are far enough from all line segments

def check_points_polygon_distances(polygon, points, threshold):
    '''
    Checks that a point is at least a certain distance away from every other line segment.
    '''
    vertices = polygon.exterior.coords
    for point in points:
        # for every line segment
        for i in range(len(vertices)):
            current_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % len(vertices)]
            line_segment = LineString([current_vertex, next_vertex])

            # Check if point is too close to any line segment
            distance = line_segment.distance(Point(point))
            if distance < threshold:
                return False  # Found a line that is too close to the point
    return True  # All lines are far enough from the point

def center_polygon(polygon):
    # Find the centroid of the polygon
    centroid = polygon.centroid
    centroid_x, centroid_y = centroid.x, centroid.y
    # Translate the polygon to center it
    translated_coords = [(x - centroid_x, y - centroid_y) for x, y in polygon.exterior.coords]
    centered_polygon = Polygon(translated_coords)
    return centered_polygon

def draw_polygon(polygon):
    fig, ax = plt.subplots()
    x, y = polygon.exterior.xy
    ax.fill(x, y, alpha=0.5, edgecolor='black', linewidth=2)
    ax.set_aspect('equal', 'box')
    plt.show()

def get_an_inside_point(room):
    p = [0.0, 0.0]
    bbox=room.get_bbox()
    p[0] = random.uniform(bbox[0][0], bbox[0][1])
    p[1] = random.uniform(bbox[1][0], bbox[1][1])
    while (not room.is_inside(p, include_borders=False)):
        p[0] = random.uniform(bbox[0][0], bbox[0][1])
        p[1] = random.uniform(bbox[1][0], bbox[1][1])
    return p

def is_array_inside_room(array, room):
    for point in array:
        if not room.is_inside(point, include_borders=False):
            return False
    return True

def get_an_inside_array(room, n=4, distance_between_mics=0.1):
    #init array
    array = []
    for i in range(n): array.append([0.0,0.0])
    bbox=room.get_bbox()

    #Generate array inside bbox
    array[0][0] = random.uniform(bbox[0][0], bbox[0][1])
    array[0][1] = random.uniform(bbox[1][0], bbox[1][1])
    for i in range(1,n):
        array[i][0] = array[0][0]+distance_between_mics*i
        array[i][1] = array[0][1]

    while not is_array_inside_room(array, room):
        #Generate array inside bbox
        array[0][0] = random.uniform(bbox[0][0], bbox[0][1])
        array[0][1] = random.uniform(bbox[1][0], bbox[1][1])
        for i in range(1,n):
            array[i][0] = array[0][0]+distance_between_mics*i
            array[i][1] = array[0][1]

    return np.transpose(np.array(array))

def check_direct_line_of_sight(room_polygon,source_pos,array):
    # print("array", array)
    for mic in np.transpose(array):
        # print(np.concatenate(np.array([[source_pos[0],source_pos[1]]]),mic))
        line = LineString([source_pos, mic])
        if room_polygon.contains(line):
            return True
        else:
            return False

def convert_lists_to_str(vertex_arr, mic_array, source_pos):
    # Convert lists to strings for saving the vertices in dataframe later
    vertex_arr_str="["
    for xy in vertex_arr:
        vertex_arr_str+="["
        for vertex in xy:
            vertex_arr_str+=str(vertex)+" "
        vertex_arr_str+="]"
    vertex_arr_str+="]"

    mic_array_str="["
    for xy in mic_array:
        mic_array_str+="["
        for mic in xy:
            mic_array_str+=str(mic)+" "
        mic_array_str+="]"
    mic_array_str+="]"

    source_pos_str="["
    for xy in source_pos:
        source_pos_str+=str(xy)+" "
    source_pos_str+="]"

    return vertex_arr_str, mic_array_str, source_pos_str

def generate_a_room_and_its_rir_and_save_it(rooms_min_sides=8, rooms_max_sides=8, rooms_min_length=2, rooms_max_length=5,\
                                  rooms_min_area=3, vertex_line_distance_threshold=1, fs=44100, energy_absorption=0.1,\
                                  scattering=0.1, number_of_mics=1, distance_between_mics=0.1, max_order=10, plot=False, save_rir=True,\
                                  vertex_arr=None, room_3d=True, min_source_mic_distance=1, min_source_wall_distance=1):
    '''
    Creates a new random room polygon or uses a specific one.
    Obtains useful info from it. Computes RIR and saves it.
    Saves all pertinent data to pandas dataframe.
    '''
    # Room polygon generation, saves the vertices
    if vertex_arr==None:
        #Create a polygon
        room_polygon = generate_random_room_polygon(rooms_min_sides, rooms_max_sides, rooms_min_length, rooms_max_length,\
                                                    min_area=rooms_min_area, vertex_line_distance_threshold=vertex_line_distance_threshold)
        room_polygon = center_polygon(room_polygon)
        # draw_polygon(room_polygon)

        # Extract the vertices that define the perimeter of the polygon
        xx, yy = room_polygon.exterior.coords.xy
        x=xx.tolist()
        y=yy.tolist()
        vertex_arr=np.array([x[:-1],y[:-1]])
    else:
        room_polygon=Polygon(np.transpose(vertex_arr))
    
    # get dimension, number of vertices, area
    (dim, n_vertex) = np.shape(vertex_arr)
    room_area=room_polygon.area

    # Convert polygon to Pyroomacoustics Room object
    wall_material = pra.Material(energy_absorption=energy_absorption, scattering=scattering)
    room = pra.room.Room.from_corners(vertex_arr, materials=wall_material, fs=fs, max_order=max_order)

    # put source and mic inside the room
    source_pos = get_an_inside_point(room)
    mic_array = get_an_inside_array(room, n=number_of_mics, distance_between_mics=distance_between_mics)
    # Ensure they're far away enough from each other and
    while np.linalg.norm(source_pos - mic_array) < min_source_mic_distance or \
        (not check_points_polygon_distances(room_polygon, [source_pos,mic_array], min_source_wall_distance)) :
        source_pos = get_an_inside_point(room)
        mic_array = get_an_inside_array(room, n=number_of_mics, distance_between_mics=distance_between_mics)
    
    if plot:
        # Plot the room geometry
        fig, ax = room.plot()
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        # ax.set_xlim(0, 10)
        # ax.set_ylim(0, 10)
        plt.show()

    line_of_sight=check_direct_line_of_sight(room_polygon,source_pos,mic_array)

    # Manage if room is 3d
    if room_3d:
        room.extrude(2.5)
        dim=3
        source_pos += [1.1]
        mic_array=np.transpose(mic_array)
        mic_array=np.transpose(np.asarray([mic_array[0].tolist()+[1.1]]))

    # Compute RIR    
    room.add_source(source_pos)
    room.add_microphone_array(mic_array)
    room.compute_rir()
    rir=room.rir[0][0]

    # Save RIR
    if save_rir:
        rir_names=glob.glob("rirsdatasetSimple/rir_**.wav")
        rir_names.sort()
        if rir_names!=[]: # if there are already some rirs saved, write at the next index
            index=int(rir_names[-1][rir_names[-1].index('_')+1:rir_names[-1][1:].index('.')+1]) # get the last index, converted to int
            rir_file_name = f"rirsdatasetSimple/rir_"+'{:06}'.format(index+1)+".wav"
        else:
            index=0
            rir_file_name = "rirsdatasetSimple/rir_000000.wav"
        scipy.io.wavfile.write(rir_file_name, fs, rir)
        print("RIR saved as " + rir_file_name)
    else:
        rir_file_name=""
    
    # plot the RIR between mic 0 and source 0
    if plot:
        plt.figure()
        plt.title("RIR : " + rir_file_name)
        plt.plot(rir)
        plt.show()

    # Convert lists to strings for saving the vertices in dataframe later
    vertex_arr_str, mic_array_str, source_pos_str = convert_lists_to_str(vertex_arr, mic_array, source_pos)

    return {"dim":dim, "n_vertex":n_vertex, "room_area":room_area, "line_of_sight":line_of_sight,
            "energy_absorption":energy_absorption, "scattering":scattering, "rir_file_name":rir_file_name,
            "vertex_arr_str":vertex_arr_str, "mic_array_str":mic_array_str, "source_pos_str":source_pos_str}

# Use this function to generate more data points for the dataset

def dataset_generation( iterations=10, rooms_min_sides=4, rooms_max_sides=10, rooms_min_length=2, rooms_max_length=7, rooms_min_area=3,
                        vertex_line_distance_threshold=1, #how far must points be from other lines?
                        fs=44100, energy_absorption=0.1, scattering=0.1, number_of_mics=1, distance_between_mics=0.1, max_order=10, plot=False, save_rir=True,
                        dataset_name="complex_room_dataset.csv", vertex_arr=None, room_3d=True, min_source_mic_distance=1, min_source_wall_distance=1):
    '''
    Generate random rooms and compute their rir's
    save their rir's in wavs
    create a bigass pandas dataframe with all the relevant data:
    dimension, number of vertexes, room area, vertexes (list), wall energy absorption,
    wall energy scattering, source position, mic position, rir_file_name, Is there a line_of_sight(true or false)
    Ordered list of nearest distance to walls, ordered list of nearest distance to walls' angles
    Ordered list of shortest paths (1 reflection) between source and mic + their incident angle
    '''
    logger=LKLogger(filename=dataset_name,
                    columns_for_a_new_log_file= ["dim", "n_vertex", "room_area", "line_of_sight",
                                                "energy_absorption", "scattering", "rir_file_name",
                                                "vertex_arr_str", "mic_array_str", "source_pos_str"])
    for _ in range(iterations):
        try:
            info=generate_a_room_and_its_rir_and_save_it(rooms_min_sides=rooms_min_sides,\
                                                        rooms_max_sides=rooms_max_sides,\
                                                        rooms_min_length=rooms_min_length,\
                                                        rooms_max_length=rooms_max_length,\
                                                        rooms_min_area=rooms_min_area,\
                                                        vertex_line_distance_threshold=vertex_line_distance_threshold,\
                                                        fs=fs,\
                                                        energy_absorption=energy_absorption,\
                                                        scattering=scattering,\
                                                        number_of_mics=number_of_mics,\
                                                        distance_between_mics=distance_between_mics,\
                                                        max_order=max_order,\
                                                        plot=plot,\
                                                        save_rir=save_rir,\
                                                        vertex_arr=vertex_arr,\
                                                        room_3d=room_3d,\
                                                        min_source_mic_distance=min_source_mic_distance,\
                                                        min_source_wall_distance=min_source_wall_distance)
            # print(info)
            logger.add_line_to_log(info)
        except Exception as e:
            # write error into a simple log file
            with open("dataset_gen_error_log.txt", "a") as f:
                f.write(str(e)+"\n")
            continue

class RoomDataset(Dataset):
    '''
    Calling RoomDataset[index] returns a tuple of tensors:
    (input_tensor, label_rir_tensor, label_origin)
    - input_tensor is a tensor of shape (3 + 3 + n_vertices*2, 1)
    It is composed of the values of the vertices, the mic position, and the source position.
    - The label_rir_tensor is a tensor of shape (rir_length, 1)
    It can be used as an intermediate label for the encoder.
    - The label_origin is an int?

    An example raw datapoint from the csv file :
    dim                                                                  3
    n_vertex                                                             7
    room_area                                                    30.580212
    line_of_sight                                                     True
    energy_absorption                                                  0.1
    scattering                                                         0.1
    rir_file_name                                  rirsdatasetSimple/rir_000.wav
    vertex_arr_str       [[-0.10601230167115427 1.9224084403891974 -1.9...
    mic_array_str        [[-0.7507980346516723 ][-4.886256689504428 ][1...
    source_pos_str          [-0.34120049727709434 -1.940370924558073 1.1 ]
    '''
    def __init__(self, csv_file, pre_filtering={'dim':3,'n_vertex':8}):
        self.csv_file=csv_file
        self.sample_rate=None
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print('csv file "', csv_file ,'" not found, generate small default dataset? (y/n)')
            if input()=='y':
                print("generating dataset...")
                dataset_generation(iterations=10, dataset_name=csv_file)
                print("dataset generated, loading csv...")
                self.data = pd.read_csv(csv_file)
            else:
                print("exiting...")
                exit()
        print('complex room dataset', csv_file ,'loaded')
        self.max_length_rir = 0
        self.preprocess(pre_filtering)
        print("complex room dataset preprocessing done.")
        # print(self.data.iloc[0])

    def preprocess(self, pre_filtering={'dim':3,'n_vertex':8}):
        # Filtering
        for key, value in pre_filtering.items():
            self.data=self.data[self.data[key]==value]
        self.data.reset_index(drop=True, inplace=True)
        # Drop columns that I don't know what to do with for now
        self.data = self.data.drop(["dim", 'energy_absorption','scattering'], axis=1)

        # Strip and split the string representations of coordinates
        self.data['vertex_arr'] = self.data['vertex_arr_str'].apply(lambda x: x.strip('[] ').replace("]","").replace("[","").split(' '))
        self.data['mic_array'] = self.data['mic_array_str'].apply(lambda x: x.strip('[] ').replace("]","").replace("[","").split(' '))
        self.data['source_pos'] = self.data['source_pos_str'].apply(lambda x: x.strip('[] ').replace("]","").replace("[","").split(' '))

        # Convert the strings to floats
        self.data['vertex_arr'] = self.data['vertex_arr'].apply(lambda x: [float(i) for i in x])
        self.data['mic_array'] = self.data['mic_array'].apply(lambda x: [float(i) for i in x])
        self.data['source_pos'] = self.data['source_pos'].apply(lambda x: [float(i) for i in x])

        # Further split the 'vertex_arr' into 2D array based on 'n_vertex'
        for i in range(len(self.data)):
            assert(len(self.data.at[i, 'vertex_arr']) == 2*self.data.at[i, 'n_vertex'])
            self.data.at[i, 'vertex_arr'] = [self.data.at[i, 'vertex_arr'][ : self.data.at[i, 'n_vertex']],
                                             self.data.at[i, 'vertex_arr'][self.data.at[i, 'n_vertex'] : ]]
        
        # These columns are no longer needed
        self.data = self.data.drop(["vertex_arr_str","mic_array_str","source_pos_str", 'n_vertex'], axis=1)

        # preprocess Boolean line_of_sight
        self.data['line_of_sight'] = self.data['line_of_sight'].apply(lambda x: 1 if x else 0)

        # Initialize max_length_rir
        self.update_max_length_of_label_rirs()

        print("length of dataset",len(self.data))

    def update_max_length_of_label_rirs(self):
        '''
        Initializes self.max_length_rir to the length of the longest RIR in the dataset.
        This is useful for padding the RIRs to the same length for the dataloader.
        Re-run this if you add to the dataset.
        '''
        for rir_file_name in self.data['rir_file_name']:
            self.sample_rate, label_rir = scipy.io.wavfile.read(rir_file_name)
            if len(label_rir) > self.max_length_rir:
                self.max_length_rir = len(label_rir)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        df = self.data.iloc[index]

        mic_tensor = torch.tensor(df['mic_array'], dtype=torch.float32)
        source_tensor = torch.tensor(df['source_pos'], dtype=torch.float32)
        vertex_tensor = torch.tensor(df['vertex_arr'], dtype=torch.float32)
        input_tensor=torch.cat((mic_tensor, source_tensor, vertex_tensor.flatten()), 0)

        self.sample_rate, label_rir = scipy.io.wavfile.read(df['rir_file_name'])
        label_rir_tensor = torch.tensor(label_rir, dtype=torch.float32)
        # find origin of RIR
        peak_indexes, _ = find_peaks(label_rir/np.max(label_rir),height=0.3)
        label_origin = peak_indexes[0]
        # pad tensor to max_length_rir for dataloader
        label_rir_tensor = torch.nn.functional.pad(label_rir_tensor, (0, self.max_length_rir - len(label_rir_tensor)))

        return input_tensor, label_rir_tensor, label_origin
    
    def plot_datapoint(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        fig, ax = plt.subplots()
        ax.set_title('RoomDataset room' + str(index))
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.grid(True, ls=':', alpha=0.5)

        x = self.data.at[index, 'vertex_arr'][0]
        y = self.data.at[index, 'vertex_arr'][1]
        ax.fill(x, y, alpha=0.5, edgecolor='black', linewidth=2, label='room')

        x = self.data.at[index, 'mic_array'][0]
        y = self.data.at[index, 'mic_array'][1]
        ax.scatter(x, y, marker='x', label='mic')

        x = self.data.at[index, 'source_pos'][0]
        y = self.data.at[index, 'source_pos'][1]
        ax.scatter(x, y, label='source')

        ax.legend()
        ax.set_aspect('equal', 'box')

        plt.show()

    def plot_datapoints(self, index):
        while index > len(self.data):
            index = index - len(self.data)
        while index < 0:
            index = len(self.data) + index
        if torch.is_tensor(index):
            index = index.tolist()
        
        fig1, axes = plt.subplots(3,5, figsize=(17,10))
        for i in range(3*5):
            iy=i%5
            ix=i//5
            axes[ix,iy].set_title('RoomDataset datapoint ' + str(index+i))
            axes[ix,iy].set_xlabel('x (m)')
            axes[ix,iy].set_ylabel('y (m)')
            axes[ix,iy].set_xlim(-6, 6)
            axes[ix,iy].set_ylim(-6, 6)
            axes[ix,iy].grid(True, ls=':', alpha=0.5)

            x = self.data.at[index+i, 'vertex_arr'][0]
            y = self.data.at[index+i, 'vertex_arr'][1]
            axes[ix,iy].fill(x, y, alpha=0.5, edgecolor='black', linewidth=2, label='room')

            x = self.data.at[index+i, 'mic_array'][0]
            y = self.data.at[index+i, 'mic_array'][1]
            axes[ix,iy].scatter(x, y, marker='x', label='mic')

            x = self.data.at[index+i, 'source_pos'][0]
            y = self.data.at[index+i, 'source_pos'][1]
            axes[ix,iy].scatter(x, y, label='source')

            axes[ix,iy].legend()
            axes[ix,iy].set_aspect('equal', 'box')
        plt.tight_layout()

        fig2, axes = plt.subplots(3,5, figsize=(17,10))
        for i in range(3*5):
            iy=i%5
            ix=i//5
            axes[ix,iy].set_title('RoomDataset datapoint ' + str(index+i) + ' RIR')
            axes[ix,iy].set_xlabel('time samples')
            axes[ix,iy].set_ylabel('Amplitude')
            axes[ix,iy].grid(True, ls=':', alpha=0.5)

            _ , rir = scipy.io.wavfile.read(self.data.at[index+i,'rir_file_name'])
            axes[ix,iy].plot(rir)
        plt.tight_layout()

        plt.show()

def main():
    dataset_generation(iterations=1000, dataset_name='simple_room_dataset.csv',
                       rooms_min_sides=4, rooms_max_sides=4,
                       vertex_line_distance_threshold=3,
                       min_source_mic_distance=3, min_source_wall_distance=1)
    dataset=RoomDataset('simple_room_dataset.csv', pre_filtering={'dim':3,'n_vertex':4})
    dataset.plot_datapoints(-20)

if __name__ == '__main__':
    main()