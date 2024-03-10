import os
import glob

configs = glob.glob("training/configs/ablation7/*.json")
configs = sorted(configs)
for config in configs:
    print(config)

for i in range(len(configs)):
    os.system(f"python -m training.train_gwa --config {configs[i]} --scheduler false")

os.system(f"python -m run_validation")

configs = glob.glob("training/configs/ablation8_with_Scheduler/*.json")
configs = sorted(configs)
for config in configs:
    print(config)

for i in range(len(configs)):
    os.system(f"python -m training.train_gwa --config {configs[i]} --scheduler true")

os.system(f"python -m run_validation8")