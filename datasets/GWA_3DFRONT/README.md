# Folder to manage the GWA 3D Front dataset and the 3D-FRONT dataset

Please add the 3D-FRONT, 3D-FUTURE-model and GWA_Dataset_small folders to this directory.

## 3D-FRONT : .json to .obj

These are the steps for a windows machine to run the blender script to go from the raw 3D-FRONT data to preprocessed meshes ready for training.
I didn't manage to make it work on WSL2, so I'm directly using windows.

### First setup

1. Git pull this repository using git bash.
2. Download the 3D-FRONT, 3D-FUTURE datasets into this folder.
3. Download the latest miniconda installer, open a miniconda powershell and:

```bash

conda create -n rirbox python=3.9
conda activate rirbox
conda install numpy
conda install pytorch -c pytorch -c nvidia -c pyg
python -m pip install fake-bpy-module
python -m pip install pandas
python -m pip install librosa
python -m pip install pymeshlab

```

4. open up vscode on windows, install the Blender Development extension for vscode.
5. On vscode, Cntrl-shift-P > Python: select interpreter > Select your rirbox environment.
6. On vscode, Cntrl-shift-P > Blender: Start > Choose a new blender executable > Choose your blender executable.
It should look like ``` C:\Program Files\Blender Foundation\Blender 4.0\blender.exe ```
7. 'Blender: start' should now have an error message (if not, that's ok). In case of an error, go to your blender's python folder ```C:\Program Files\Blender Foundation\Blender 4.0\4.0\python``` and give it [writing privileges](https://www.youtube.com/watch?v=YUytEtaVrrc&t=469s) by :
8. right-click 'blender python folder' > properties > Security > Select Users group > Edit > Select Users group > Click on Allow write checkbox > Apply.
9. On vscode, Cntrl-shift-P > Blender: Start > ... again. Blender should start, and in vscode you should see 'debug client attached'
10. On vscode, go to blender_test.py and Cntrl-shift-P > Blender: Run Script

A big cube should have appeared! Congrats. Now, on to the actual scripting.

### Quick setup for next times

1. On vscode, Cntrl-shift-P > Blender: Start
2. On vscode, go to format_3D_front.py and Cntrl-shift-P > Blender: Run Script
