import json
import os
import argparse
from tqdm import tqdm
from preprocessing.json2obj import json2obj
from preprocessing.obj_preprocessing import obj_preprocessing

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument(
        '--json_path',
        default = './3D-FRONT',
        help = 'path to 3D FRONT'
        )
parser.add_argument(
        '--future_path',
        default = './3D-FUTURE-model',
        help = 'path to 3D FUTURE'
        )
parser.add_argument(
        '--obj_path',
        default = './obj_meshes',
        help = 'path to save (temporarily converted) obj meshes (from jsons)'
        )
parser.add_argument(
        '--preprocessed_obj_path',
        default = './preprocessed_obj_meshes',
        help = 'path to save simplified obj meshes'
        )
parser.add_argument(
        '--target_faces',
        default = 2000,
        help = 'target number of faces'
        )
args = parser.parse_args()

# Create folders if they don't exist
if not os.path.exists(args.obj_path): os.mkdir(args.obj_path)
if not os.path.exists(args.preprocessed_obj_path): os.mkdir(args.preprocessed_obj_path)

# Get overall list of rooms, and of furniture
files = os.listdir(args.json_path)
with open(os.path.join(args.future_path, 'model_info.json'), 'r', encoding='utf-8') as file:
    model_info = json.load(file)

# Iterate over all rooms in 3D-FRONT
i=0
for json_file_name in tqdm(files):
#     if json_file_name == '402de23a-6681-4cb5-9d8a-7db57aba4a3c.json':
#         continue

    # convert json to obj
    temp_file_path = json2obj(json_file_name, args.json_path, args.future_path,
                                        args.obj_path, model_info)
    if temp_file_path is None:
        print('skipping', json_file_name)
        continue

    # simplify obj
    simple_file_path = obj_preprocessing(temp_file_path, args.preprocessed_obj_path, args.target_faces)

    # remove temp file
    os.remove(temp_file_path)
    # remove temp folder
    temp_folder = os.path.join(args.obj_path, json_file_name[:-5])
    os.rmdir(temp_folder)

    i+=1
    if i >= 100:
        break
