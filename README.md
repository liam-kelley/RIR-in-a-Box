# RIR-in-a-Box V.0.3

## Installation

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

It sometimes bugs and freezes a bit. Please kill it and restart it if it does.
