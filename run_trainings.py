import os
import glob

'''
Runs multiple training scripts.
'''

configs = glob.glob("training/configs/best_models/*.json")
# configs.extend(glob.glob(///))
configs = sorted(configs)

# configs=[
#     "training/configs/best_models/rirbox_Model2_dp_MRSTFT_EDR_superfast_MSDist_DistInLatent_NormByDist_12epochs.json",
#     "training/configs/best_models/rirbox_Model3_dp_HIQMRSTFT_EDR_superfast_MSDist_DistInLatent_noNormByDist_12epochs.json"
# ]

print("Training following config files")
for config in configs:
    print(config)

for i in range(len(configs)):
    os.system(f"python -m training.train_gwa --config {configs[i]} --dowandb")
