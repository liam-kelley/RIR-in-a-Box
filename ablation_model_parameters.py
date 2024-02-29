import os


for i in range(2,3):
    os.system(f"python3 train_gwa.py --MLP_DEPTH {i}")