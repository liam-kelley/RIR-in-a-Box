import os

# os.system(f"python train_gwa.py --config ./training/default/rirbox_default.json")

configs = [
    "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MSDist_MLPDEPTH4_1m.json",
    "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MSDist_MLPDEPTH4_dp.json",
    "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MSDist_MLPDEPTH4_1m.json",
    "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MSDist_MLPDEPTH4_dp.json",
    "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MLPDEPTH4_1m.json",
    "training/ablation3_different_datasets/rirbox_model2_MRSTFT_MLPDEPTH4_dp.json",
    "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MLPDEPTH4_1m.json",
    "training/ablation3_different_datasets/rirbox_model3_MRSTFT_MLPDEPTH4_dp.json",
    # "./training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH2.json",
    # "./training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH3.json",
    # "./training/ablation2/rirbox_model3_MRSTFT_MLPDEPTH4.json",
    # "./training/ablation2/rirbox_model3_MRSTFT_EDR_MLPDEPTH4.json",
    # "./training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH2.json",
    # "./training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH3.json",
    # "./training/ablation2/rirbox_model2_MRSTFT_MLPDEPTH4.json",
    # "./training/ablation2/rirbox_model2_MRSTFT_EDR_MLPDEPTH4.json",
]

for i in range(len(configs)):
    os.system(f"python train_gwa.py --config {configs[i]}")