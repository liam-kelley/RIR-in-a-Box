'''
This is a blender python script.
Run it on vscode on windows if you so desire. Instructions are in the README.md file in this folder.
'''

import bpy
from pathlib import Path

# Define the paths using Path objects
json_path = Path("C:/Users/liamk/Documents/Work/Post-PFE/RIR-in-a-Box/datasets/GWA_3DFRONT/3D-FRONT/0a8d471a-2587-458a-9214-586e003e9cf9.json")
future_model_path = Path("C:/Users/liamk/Documents/Work/Post-PFE/RIR-in-a-Box/datasets/GWA_3DFRONT/3D-FUTURE-model")

# Display the paths (optional)
print(json_path)
print(future_model_path)

