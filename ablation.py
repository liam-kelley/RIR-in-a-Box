import os

configs = [
    "training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH2.json",
    "training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH3.json",
    "training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH2.json",
    "training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH3.json"
]

for i in range(len(configs)):
    os.system(f"python train_gwa.py --config {configs[i]}")