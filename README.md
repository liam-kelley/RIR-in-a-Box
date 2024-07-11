# RIR-in-a-Box V.0.5

This is the code for our paper [**RIR-in-a-Box: Estimating Room Acoustics from 3D Mesh Data through Shoebox Approximation**](https://theses.hal.science/LTCI/hal-04632526v1 "Paper on HAL"), accepted at Interspeech 2024.

## Setup

This setup tutorial is functional, but work is still being done to simplify the reuse process.

### Step 1 : cuda 12.1 toolkit

Install the cuda 12.1 toolkit. Other versions may work, but this is the one I used.
If you use another other, please change the pytorch-cuda version in the conda install command below.

### Step 2 : python environment

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
python -m pip install shapely
python -m pip install pyvista
conda install pytorch-scatter -c pyg -c nvidia

```

### Step 3 : go elsewhere for more instructions

Go to the RIR-in-a-Box/datasets/GWA_3DFRONT folder for extra instructions.

### Step 4 : go elsewhere for more instructions

Then, go to the RIR-in-a-Box/models folder for extra instructions.

### Step 5 : Have fun

Congrats! Enjoy RIR-in-a-Box.

## Usage

```bash

conda activate rirbox

```

To train models. You can choose your own config files by modifying the script.

```bash

python run_trainings.py

```

To validate model performance / view validation results.
You can choose your own config files by modifying the script.
You can also choose which validation experiments to run / show by modifying the script.

```bash

python run_validation.py

```

To monitor nvidia gpu usage in another terminal.

```bash

watch -d -n 0.5 nvidia-smi

```

## Citation

If you find our work or code repository useful in your research, please cite our paper:

```citation

Liam Kelley, Diego Di Carlo, Aditya Arie Nugraha, Mathieu Fontaine, Yoshiaki Bando, et al.. RIR-in-a-Box: Estimating Room Acoustics from 3D Mesh Data through Shoebox Approximation. INTERSPEECH, Sep 2024, Kos International Convention Center, Kos Island, Greece. ⟨hal-04632526⟩

```

Thank you for your support!

<!--
# to try and avoid loading tensors into shared memory (you might not need this)
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.4 # instead of 0.8

## Repository description

Updated 22/04/2024

### root folder (RIR-IN-A-BOX)

In the root folder, you can find:

- a script to performe inference for models
- a script to run trainings on multiple configuration files
- a script to run different types of validation on multiple configuration files.

### backpropagatable_ISM

In this folder, you can find my immplementation of the backpropagatable ISM, and some filters used for the fractional delays in the computation.
This backpropagatable ISM can be easily used through the **ShoeboxToRIR** nn.Module available in models/rirbox_models.py.

Notes:

- For now, I have not reimplmented multiband-absorption processing due to gpu memory concerns.
- My implementation is very memory-hungry, so you should use low reflection orders, low RIR lengths and single-band absorptions.
- I implemented a experimental option to not include the initial silent portion before the earliest Time of Arrival to save on gpu memory. This is only useful if the user respatializes the RIR afterwards using the ground truth distance to the listener. But this will surely sound weird.

### datasets

#### GWA_3DFRONT

In this folder, you can find:

- The torch.utils.data.Dataset for the GWA (audio) + 3DFRONT (Room Meshes) dataset. It has a custom collate function for dataloaders.
- Preprocessing scripts for the datasets and the lists of the different subsets used.

The readme file has instructions on what to prepare in this folder.

#### ValidationDataset

In this folder, you can find:

- The torch.utils.data.Dataset for the Validation dataset. It has a custom collate function for dataloaders.
- The scripts used for the dataset creation.
- A description of the dataset along with which scripts were used for dataset creation.

### losses

In this folder, you can find my implementations of the losses on the RIR :

- EnergyDecay_Loss
- MRSTFT_Loss
- AcousticianMetrics_Loss (D, C80, DRR, and RT60 simultaneously)

and the losses on the simulated shoeboxes :

- SBoxRoomDimensionsLoss
- SBoxAbsorptionLoss
- MicSrcConfigurationLoss

### models

In this folder, you can find:

- the implementations of the mesh2ir models (MESH_NET, STAGE1_G, and the nice MESH2IR_FULL class for easy inference).
- the implementations of the rirbox models (MeshToShoebox, ShoeboxToRIR, and the nice RIRBox_FULL class for easy inference)
- nice utility functions for easy config and model loadings.

### script_archive

In this folder, you can find deprecated scripts.

### tools

In this folder, you can find :

- pyLiam (a collection of my personal implementations of a csv Logger, a cuda memory checker, and a timer)
- a gcc_phat tdoa implementation.
- an Image Source Model visualization jupyter notebook.

### training

In this folder, you can find :

- configuration json files for the different models.
- a training script for training on the gwa dataset.
- a few utility functions for training

### validation

In this folder, you can find :

- the beamforming validation experiment (+ its visualization)
- the metric accuracy validation experiment (+ its visualization)
- the sound source spatialization experiment (+ its visualization)
- experiment results
- visualization for all experiments all together.

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

It sometimes bugs and freezes a bit. Please kill it and restart it if it does. -->
