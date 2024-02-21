# RIR-in-a-Box V.0.4

Please wait for RIR-in-a-Box V.0.5 to be released for a complete installation tutorial.

I'm pretty sure the repo should work with these requirements.
Know that there are extra librairies that are not needed for the project to work.
Make sure to have the cuda 12.1 toolkit installed (use apt to install it).

```bash

conda create -n rirbox python=3.9
conda activate rirbox
conda install numpy
conda install pytorch torchvision torchaudio pyg pytorch-cuda=12.1 -c pytorch -c nvidia -c pyg
python -m pip install python-dateutil
python -m pip install soundfile
python -m pip install pandas
python -m pip install scipy
python -m pip install librosa
python -m pip install easydict
python -m pip install cupy-cuda12x
python -m pip install wavefile
python -m pip install torchfile
python -m pip install pyyaml==5.4.1
python -m pip install pymeshlab
python -m pip install openmesh
python -m pip install gdown
python -m pip install matplotlib
python -m pip install IPython
python -m pip install pydub
python -m pip install auraloss
python -m pip install wandb
python -m pip install pyroomacoustics
python -m pip install fake-bpy-module
python -m pip install trimesh
python -m pip install tqdm
python -m pip install libigl
<!-- conda install torch-scatter torch-sparse torch-cluster torch-spline-conv-c pyg -c nvidia  -->

```

<!-- ## Installation

This installation tutorial has not been fully tested again.

From a completely clean Ubuntu 20.04 installation

```bash
sudo apt install build-essential
sudo apt install git
```

Setup your ssh key to clone this repository

```bash
git clone git@github.com:liam-kelley/RIR-in-a-Box.git
```

Conda installation
Use the [installer](https://www.anaconda.com/download/#linux).

```bash
echo "export PATH=~/anaconda3/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

```bash
conda init
conda create --name rirbox python=3.8
conda activate rirbox
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-scatter -c pyg
which python
python -m pip install librosa auraloss torch-geometric
python -m pip install pymeshlab matplotlib pandas wandb pandas shapely pyroomacoustics
python -m pip install pymeshfix trimesh pyglet
```

Nvidia drivers
Go to [this website](https://www.nvidia.com/Download/index.aspx?lang=en-us)
and download the correct driver for your GPU.
I had to run the driver installation while in recovery mode to not mess up my system.
Restart your computer after the installation.

## Preparation

Shoebox MeshDataset generation.

```bash
mkdir meshdataset
cd meshdataset
mkdir rirs
mkdir meshes
cd ..
```

You can then use the shoebox_mesh_dataset_generation function from the mesh_dataset.py file to generate a dataset of shoebox meshes and RIRs.
Example usage is in the main function from mesh_dataset.py

```bash
python mesh_dataset.py
```

It sometimes bugs and freezes a bit. Please kill it and restart it if it does. -->
