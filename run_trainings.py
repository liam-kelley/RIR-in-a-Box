import os
import glob

configs = glob.glob("training/configs/ablation6_Loss_Option_Subset_Architecture/*.json")

# configs = [
#     # "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_SyncOnset_HIQMRSTFT_EDR_superfast.json",
#     # "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_SyncOnset_HIQMRSTFT_EDR_RT60_superfast.json",
#     # "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_SyncOnset_HIQMRSTFT_EDR_RT60.json"
# ]

for i in range(len(configs)):
    os.system(f"python -m training.train_gwa --config {configs[i]}")