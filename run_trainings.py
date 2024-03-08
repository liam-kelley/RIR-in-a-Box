import os
import glob

# configs = glob.glob("training/ablation4_model_type_and_depth_fixed/*.json")

configs = [
    "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_SyncOnset_HIQMRSTFT_EDR.json"
]

for i in range(len(configs)):
    os.system(f"python -m training.train_gwa --config {configs[i]}")