# RIR-in-a-Box

README updated 30/08/2023

## Installation

```bash
python -m pip install librosa auraloss torch-geometric torch-scatter
```

There are probably more dependencies to install, like torch.

## Repository description

---

### complex_room_dataset.py

This file contains a torch dataset class called ***RoomDataset***. ***RoomDataset*** contains 2D rooms of varying complexity with a microphone and source within them, with the label Room Impulse Response (RIR) which corresponds to that room.

Its "**\_\_getitem\_\_**" method returns an input tensor for the ***RoomToShoeboxEncoder***, along with label_rir_tensor and label_origin tensors.

New datapoints can be generated for the dataset quickly using the **dataset_generation** function.

Options for the room/RIR dataset generation

- *Number of vertices*
- *Min/Max size*
- *Min Area*
- *Min Vertex — walls distance*
  - Lowering this makes rooms crazier
- *Min source — mic distance*
  - The “Source-Mic Position” experiment showed that the metrics are sensitive to “source — mic distance” when “source — mic distance” <1 m
- *Min source/mic — wall distance*
  - The “rooms” experiment showed that the metrics don’t work at all if the Mic/Source are near walls (<1m) and work better if the sources are =1m away from walls.
- *Force line of sight?*
  - Enforces there being a straight path between the source and the mic*
- *Wall absorption*
- *Wall scattering*
- *Max order of reflections for label RIR calculation*

The ***RoomDataset*** class also has methods for plotting datapoints.

---

### compute_rir.py

This is a reimplementation of the pyroomacoustics function compute_rir. It is modified to work with pytorch tensors with autograd.
The original function is not differentiable, and therefore cannot be used in a neural network.
Absorption is not backpropagatable yet. It is a constant for now.

The function is called **simulate_rir_ism**. The summing of the image sources is done with torch, in the frequency domain, rather than with Cython with fractionnal delays.

---

### encoder.py

This file contains my torch encoder modules.

***RoomToShoeboxEncoder*** was the first attempt at implementing a [...]toShoeboxEncoder.
It's a simple encoder module.
It takes in a **batch of input features** (mic_position (3d), src_position (3d), vertices(nx (8) + ny (8))) of shape = [batch_size, 22].
It has an **intermediate shoebox representation** ((9,1) tensor = room_dim, mic_pos, src_pos)
And finally outputs **shoebox_rir** and **shoebox_rir_origin**.

Here, 8 2d vertices are used, but conceivably any amount of vertices is ok. Please initialize the input length to the right size.

***GraphToShoeboxEncoder*** is the generalization of the ***RoomToShoeboxEncoder*** to graph inputs. It is still under construction. Sorry!

---

### LKLogger.py

This file contains the ***LKLogger*** class, my own custom logger for csv files. This is used in RIRMetricsExperiments.py and for the ***complex_room_dataset*** data generation.

---

### mesh_dataset.py

This file contains a torch dataset class called ***GraphDataset***. ***GraphDataset*** contains mesh representations of rooms (ONLY SHOEBOXES for now) of varying complexity with a microphone and source within them, with the label Room Impulse Response (RIR) which corresponds to that room.

Its "**\_\_getitem\_\_**" method returns an input tensor for the ***GraphToShoeboxEncoder***, along with label_rir_tensors and label_origin tensors, and label_mic_pos, label_src_pos, label_energy_absorption, label_scattering.

New datapoints can be generated for the dataset quickly using the **shoebox_mesh_dataset_generation** function. (I plan to implement generation from rooms from the rooms dataset rather than just shoeboxes).

Options for generation and example use of this dataset are provided in the **main** function.

This file also has functions to plot a mesh from either an edge_index or a triangle list.

---

### RIRMetricsExperiments.py

Many helpful functions are in this file. It might need to be cleaned up.

I might have made the metrics in this file bug out, but they are properly implemented in RIRMetricsLoss.py.

I used the ***RIRMetricsExperiments*** class to run experiments on my metrics to test them and how efficient they are in different scenarios.

Three main experiments were done:

#### overfit_training

Use gradient descent to try and fit the encoder inputs to have the intermediate shoebox room fit one target shoebox room.

#### grid_losses

Localize the target source position in an oracle room using the metrics.

#### rooms_losses

Identify room size with oracle source and mic positions using the metrics.

---

### RIRMetricsLoss.py

This file implements the torch loss module ***RIRMetricsLoss***.
This module produces a loss value from two Room Impulse Responses (provided with their time of first/earliest reflection)

This loss can be parametrized by its different lambda parameters.
Only the metrics with their index in the lambda_param dictionnary and with non-zero values will be involved in the calculated. It also has options for origin syncronizing and toggling filtering in ms_env.

There are 6 Losses/metrics:

- **multi-resolution stft difference loss** 'mrstft' from auraloss. This has the option of having the two RIRs being compared having their origins synchronized to focus more on the actual reverberation shape.
- **Filtered Envelope Sum Difference Loss** 'ms_env' (needs renaming). The envelopes of both RIRs are filtered by many different filters. These filtered envelopes are then summed. The two Filtered envelope sums are then compared and the total difference is the value of this Loss.
- **'rt60', 'd', 'c80', 'center_time'**. These metrics are calculated for both RIRs, and the difference between these metrics is the value of the loss.

All of these are averaged over the batch, and then summed according to the lambda values for the final value of the RIRMetricsLoss.

This file also has a function to visualize theses metrics called **plot_rir_metrics**.
It visualizes metrics for a single rir pair or the first element of a batch of rir pairs.

- Graph 1: Envelope Visualisation
- Graph 2: RT60 Visualisation
- Graph 3: C80, D, CENTER_TIME Visualisation
- Graph 4: Filtered Envelope Sum (ms_env) Visualisation
- Graph 5: Multi-Resolution STFT Visualisation
- Graph 6: Bar graphs between compared metrics.

Center time visualization needs a bit of work.

---

### train_2D_rooms.py

This file is used to train the RoomToRIR encoder. Run logging is implemented with wandb.

Models aren't saved yet.

### train_graphs.py

Not fully implemented yet.
