'''
This is a blender python script.
Run it on vscode on windows if you so desire. Instructions are in the README.md file in this folder.
'''

import bpy
from pathlib import Path

import os, sys
path_to_append = os.path.expanduser(Path("C:/Users/liamk/Documents/Work/Post-PFE/RIR-in-a-Box/datasets/GWA_3DFRONT/")) # Set up path for imports
sys.path.append(path_to_append)
from datasets.GWA_3DFRONT.fakeblenderproc.fakebproc_load_front3d import load_front3d_no_materials

# Define the paths using Path objects
json_path = Path("C:/Users/liamk/Documents/Work/Post-PFE/RIR-in-a-Box/datasets/GWA_3DFRONT/3D-FRONT/0a8d471a-2587-458a-9214-586e003e9cf9.json")
future_model_path = Path("C:/Users/liamk/Documents/Work/Post-PFE/RIR-in-a-Box/datasets/GWA_3DFRONT/3D-FUTURE-model")

# Display the paths (optional)
print(json_path)
print(future_model_path)

# Load the 3D-Front scene 3D-Future models
loaded_objects = load_front3d_no_materials(
    json_path=json_path,
    future_model_path=future_model_path
)
print("Loaded objects: ", len(loaded_objects))
print(loaded_objects)

