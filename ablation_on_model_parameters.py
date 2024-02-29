import os

os.system(f"python train_gwa.py --config rirbox_model2_MRSTFT.json")
os.system(f"python train_gwa.py --config rirbox_model2_MRSTFT_EDR.json")
os.system(f"python train_gwa.py --config rirbox_model2_MRSTFT_EDR_D.json")
os.system(f"python train_gwa.py --config rirbox_model2_MRSTFT_EDR_D_RT60.json")

