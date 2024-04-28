from tools.pyLiam.LKLogger import LKLogger
import argparse
from json import load
from datasets.SBAM_Dataset.dataset_generation_scripts.config_and_csv_utility import get_empty_csv_row
from datasets.SBAM_Dataset.dataset_generation_scripts.generate_one_SBAM_datapoint import generate_one_SBAM_datapoint

'''
Use this script to generate dataset data points.

Usage in root RIRbox folder: python -m datasets.SBAM_Dataset.generate_SBAM_dataset
Debugging :                  python -m datasets.SBAM_Dataset.generate_SBAM_dataset --plot --dontsave

For each iteration, this script:
1. Generates a random shoebox with random monoband absorption values on walls, floor and ceiling.
2. Computes its rir using PyroomAcoustics.
3. Randomly translates + rotates the shoebox + mic and src 
4. Creates a mesh with small random normal displacements following the shoebox surfaces.
--> We do this by creating 6 mesh surfaces, then sticking them together using a mesh reconstruction algorithm.
5. Optionally degrades the mesh with holes
6. Reduces the mesh to 2000 faces
5. Saves the mesh in obj files and saves the rir in wav
3. Logs the dataset information in a log file using my custom logger class.

As far as weight goes, PER datapoint you're looking at about
a 175kb mesh
and a 10kb to 13kb rir
'''

# get arparse args
parser = argparse.ArgumentParser()
parser.add_argument('--datasetname', type=str, default="datasets/SBAM_Dataset/subsets/sbam.csv", help='Dataset csv name')
parser.add_argument('--config', type=str, default="datasets/SBAM_Dataset/dataset_generation_scripts/generation_configs/default.json", help='Path to configuration file.')
parser.add_argument('--plot', action='store_true', help='Plot every shoebox mesh you are creating')
parser.add_argument('--dontsave',action='store_false', help="Don't save the dataset (use for debugging)")
args, _ = parser.parse_known_args()

# load config
with open(args.config, 'r') as file: config = load(file)

# init logger (initializes csv file, writes out the csv file line per line)
empty_log_row = get_empty_csv_row()
logger=LKLogger(filename=args.datasetname, columns_for_a_new_log_file = empty_log_row.keys())

# Iterate generation
errors=0
for _ in range(1000):
    try:
        log_row=generate_one_SBAM_datapoint(config, args)
        logger.add_line_to_log(log_row)
    except Exception as e:
        print("Exception occured during generation, skipping this datapoint")
        print(e)
        errors+=1
        continue

print("Generation complete")
print(f"Dataset size : {len(logger.get_df())}")
print(f"Failed generations : {errors}")
print("Generation parameters :")
for key, value in config.items():
    print(f"    {key} : {value}")
