import os
import glob

configs = glob.glob("training/configs/ablation12_M3_noHiQ/*.json")
# configs.extend(glob.glob("training/configs/ablation11_justMRSTFT/*.json"))

# configs = sorted(configs)
# for config in configs:
#     print(config)

# configs=[
#     "training/configs/ablation8_model2again/ablation_Model2_nonzero_HIQMRSTFT_EDR_superfast_MSDist_DistInLatent_NormByDist_5epochs.json"
# ]

for i in range(len(configs)):
    os.system(f"python -m training.train_gwa --config {configs[i]} --scheduler")

# os.system(f"python -m run_validation")
