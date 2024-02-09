# RIR-in-a-Box Validation Dataset

This is to be the validation dataset for RIR-in-a-Box.
It contains 55 audio recordings of 10 ESS each.
It also contains 30 room meshes recorded on the HoloLens of 1 (**TODO** : 2) room configuration(s) with 3 different recording methods.

(Updated 09/02/2024)

## Information

### Dataset creation

- I obtained mesh data from the HoloLens using the device portal. As you scan the room longer, the data becomes more precise, so I made sure to record the mesh multiple times to account for the amount of time spent scanning. So far I only recorded the confroom open configuration, so **TODO** record the confroom closed configuration. **TODO** add the mesh names into 'validation_dataset.csv'. (Right now they're only implemented as booleans).
- I manually measured mic and src positions in the office and wrote them down in 'validation_dataset.csv'. (see my notes room_geometry/manually_measured/).
- I used 'ESS_play.ipynb' to play/save the ESS on my PC.
- I recorded the ESS using the HoloLens and saved the recordings in 'audio_recordings/'. **TODO** add the audio filepaths into 'validation_dataset.csv'.
- I used 'deconvolve_audio_recs.py' to deconvolve the audio recordings into 'deconvolved_audio_recs/'.
- I then used 'slice_10xrirs.py' to get 10 rirs from the deconvolved audio recordings.

### Other scripts

- 'get_sweep.py' provides a utility to get an ESS, an inverse filter, the ESS length and the sampling rate used for the ESS.
- 'visualize_10xrirs.ipynb' was used to identify the correct 'find_peaks' properties. (OUTDATED)

### Simplified summary of validation configurations

This totals up to 481 usable configurations.
**TODO** For every mesh Diego/Arie/Fuijta :'\) records with the confroom closed, we'll add 35 configurations.
**TODO**  For every mesh Diego/Arie/Fuijta :'\) records with the confroom open, we'll add 42 configurations.

| Room Configuration | Source Position          | Listener Position | Corresponding Meshes                                               |
|--------------------|--------------------------|-------------------|--------------------------------------------------------------------|
| Confroom Closed    | Src A (Door)             | Point 1-7         | /                                                                  |
| Confroom Closed    | Src A (Close Corner)     | Point 1-7         | /                                                                  |
| Confroom Closed    | Src B (Far Corner)       | Point 1-7         | /                                                                  |
| Confroom Closed    | Src B (Far Corner)       | Point 1-7         | /                                                                  |
| Confroom Closed    | Src B (Close Corner)     | Point 1-7         | /                                                                  |
| Confroom Open      | Src B (Close Corner)     | Point 1-2, 4-6    | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src B (Close Corner)     | Point 7           | Walk around both rooms 0-4, Walk around rooms sequential 0-11, Looking Around Seated Conference room open 0 - 12 |
| Confroom Open      | Src A (Door)             | Point 1-6         | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src A (Door)             | Point 7           | Walk around both rooms 0-4, Walk around rooms sequential 0-11, Looking Around Seated Conference room open 0 - 12 |
| Confroom Open      | Src C (Open Door)        | Point 1-6         | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src C (Open Door)        | Point 7           | Walk around both rooms 0-4, Walk around rooms sequential 0-11, Looking Around Seated Conference room open 0 - 12 |
| Confroom Open      | Src C (Open Door)        | Point 8           | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src A (Open Door)        | Point 1-5         | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |

### Source, Listener recording positions  (manually measured)

| Point            | x (in m) | y (in m) | z (in m) | Orientation theta (rad) |
|------------------|----------|----------|----------|-------------------------|
| Origin           | 0        | 0        | 0        | /                       |
| Point 1          | 0.87     | -1       | 1.75     | 0                       |
| Point 2          | 1.59     | -1.73    | 1.75     | 0                       |
| Point 3          | 1.18     | -2.52    | 1.75     | 0                       |
| Point 4          | 2.43     | -2.52    | 1.75     | 0                       |
| Point 5          | 2.92     | -0.8     | 1.75     | 0                       |
| Point 6          | 4.89     | -1.26    | 1.75     | 0                       |
| Point 7          | 2.73     | -2.36    | 1.25     | π/2                     |
| Point 8          | 4.29     | -3.46    | 1.75     | 0                       |
| Src A (Door)     | 4.55     | -2.33    | 1.18     | 2.815233837             |
| Src A (Open Door)| 4.55     | -2.33    | 1.18     | 1.796948763             |
| Src A (Close Corner)| 4.55  | -2.33    | 1.18     | 0.6027864726            |
| Src B (Close Corner)| 3.94  | -1.24    | 1.18     | 0.7744576809            |
| Src B (Far Corner)| 3.94    | -1.24    | 1.18     | 2.606922419             |
| Src C (Open Door)| 4.78     | -5.79    | 1.18     | 1.778076243             |
| Room Ceiling     | /        | /        | 3.14     | /                       |
|Door              | 0        | -0.79    | 1.18     | /                       |
|Open door         | 4.29     | -3.46    | 1.18     | /                       |
|Close corner      | 6.25     | -3.5     | 1.18     | /                       |
|Far corner        | 0.09     | -3.52    | 1.18     | /                       |
