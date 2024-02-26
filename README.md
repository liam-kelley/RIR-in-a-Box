# RIR-in-a-Box V.0.5

## Setup

This setup tutorial is functional, but beware there may be too many librairies installed.

Install the cuda 12.1 toolkit. Other versions may work, but this is the one I used.
If you use another other, please change the pytorch-cuda version in the conda install command below.

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
conda install pytorch-scatter -c pyg -c nvidia

```

Once you set up the environment, go to the RIR-in-a-Box/datasets/GWA_3DFRONT folder for extra instructions.

Then, go to the RIR-in-a-Box/models folder for extra instructions.

Congrats! Enjoy RIR-in-a-Box.

## Usage

```bash

conda activate rirbox
# to try and avoid loading tensors into shared memory
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.4 # instead of 0.8

```

To train a model.

```bash

python train_gwa.py --config "./training/rirbox_model2_finetune.json"

```

To validate model performance.

```bash

python validation.py --rirbox_path "./models/rirbox_model2_finetune.pth"

```

To monitor gpu usage.

```bash

watch -d -n 0.5 nvidia-smi

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
