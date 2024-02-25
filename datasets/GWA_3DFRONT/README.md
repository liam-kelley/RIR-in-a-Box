# Folder to manage the GWA 3D Front dataset and the 3D-FRONT dataset

Please add the 3D-FRONT, 3D-FUTURE-model and GWA_Dataset_small folders to this directory.
If you are specifically members of my team, feel free ask me for the much smaller preprocessed datasets, I'll put them on google drive for you guys.

Then, run the following script while here:

```bash

python preprocess_3D_front.py

```

Congrats, the datasets should be ready.
You can now proceed to the 'RIR-in-a-Box/models' folder for extra instructions.

## If the the csv file isn't right

If the gwa_3Dfront.csv file isn't right, please go into the preprocessing folder and run the following script:

```bash

python format_csv.py

```
