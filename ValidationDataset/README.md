# RIR-in-a-Box Validation Dataset

This is to be the validation dataset for RIR-in-a-Box.
It contains 55 audio recordings of 10 ESS each.
It also contains 30 room meshes recorded on the HoloLens of 1 room configuration with 3 different recording methods.
I'll ask Diego/Arie/Fujita to record extra room meshes for the other room configuration.

(Info checked 08/02/2024)

## Details

### All configurations

| Room Configuration | Source Position          | Listener Position | Corresponding Meshes                                               |
|--------------------|--------------------------|-------------------|--------------------------------------------------------------------|
| Confroom Closed    | Src A (Door)             | Point 1-7         | /                                                                  |
| Confroom Closed    | Src A (Close Corner)     | Point 1-7         | /                                                                  |
| Confroom Closed    | Src B (Far Corner)       | Point 1-7         | /                                                                  |
| Confroom Closed    | Src B (Far Corner)       | Point 1-7         | /                                                                  |
| Confroom Closed    | Src B (Close Corner)     | Point 1-7         | /                                                                  |
| Confroom Open      | Src B (Close Corner)     | Point 1-2, 4-6    | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src B (Close Corner)     | Point 7           | Walk around both rooms 0-4, Walk around rooms sequential 0-11, Looking Around Seated Conference room open 0 - 13 |
| Confroom Open      | Src A (Door)             | Point 1-6         | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src A (Door)             | Point 7           | Walk around both rooms 0-4, Walk around rooms sequential 0-11, Looking Around Seated Conference room open 0 - 13 |
| Confroom Open      | Src C (Open Door)        | Point 1-6         | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src C (Open Door)        | Point 7           | Walk around both rooms 0-4, Walk around rooms sequential 0-11, Looking Around Seated Conference room open 0 - 13 |
| Confroom Open      | Src C (Open Door)        | Point 8           | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |
| Confroom Open      | Src A (Open Door)        | Point 1-5         | Walk around both rooms 0-4, Walk around rooms sequential 0-11     |

### Source, Listener recording positions

| Point            | x (in m) | y (in m) | z (in m) | Orientation theta (rad) |
|------------------|----------|----------|----------|-------------------------|
| Origin           | 0        | 0        | 0        | /                       |
| Point 1          | 0.87     | -1       | 1.75     | 0                       |
| Point 2          | 1.59     | -1.73    | 1.75     | 0                       |
| Point 3          | 1.18     | -2.52    | 1.75     | 0                       |
| Point 4          | 2.43     | -2.52    | 1.75     | 0                       |
| Point 5          | 2.92     | -0.8     | 1.75     | 0                       |
| Point 6          | 4.89     | -1.26    | 1.75     | 0                       |
| Point 7          | 2.73     | -2.36    | 1.25     | π/2           |
| Point 8          | 4.29     | -3.46    | 1.75     | 0                       |
| Src A (Door)     | 4.55     | -2.33    | 1.18     | 2.815233837             |
| Src A (Open Door)| 4.55     | -2.33    | 1.18     | 1.796948763             |
| Src A (Close Corner)| 4.55  | -2.33    | 1.18     | 0.6027864726            |
| Src B (Close Corner)| 3.94  | -1.24    | 1.18     | 0.7744576809            |
| Src B (Far Corner)| 3.94    | -1.24    | 1.18     | 2.606922419             |
| Src C (Open Door)| 4.78     | -5.79    | 1.18     | 1.778076243             |
| Room Ceiling          | /        | /        | 3.14     | /                       |
