import os
import glob

# configs = glob.glob("training/ablation4_model_type_and_depth_fixed/*.json")

configs = [
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model2_DistInLatent_RT60_dp.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model2_DistInLatent_dp.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model3_DistInLatent_RT60_dp.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model3_DistInLatent_dp.json",
]

for i in range(len(configs)):
    os.system(f"python training/configs/train_gwa.py --config {configs[i]}")