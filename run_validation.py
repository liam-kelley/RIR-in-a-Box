import os
import glob
from validation.metric_accuracy import metric_accuracy_mesh2ir_vs_rirbox, view_results_metric_accuracy_mesh2ir_vs_rirbox

DO_METRIC_ACCURACY = False
VISUALIZE_METRIC_ACCURACY = True

model_configs = [
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model2_DistInLatent_dp.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model2_DistInLatent_RT60_dp.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model2_DistInLatent_RT60.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model2_DistInLatent.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model3_DistInLatent_dp.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model3_DistInLatent_RT60_dp.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model3_DistInLatent_RT60.json",
    "training/configs/ablation5_letsmakeitwork/rirbox_MSDist_HIQMRSTFT_Dropout_MLP4_Hidden128_Model3_DistInLatent.json",
]

if DO_METRIC_ACCURACY:
    validation_csv = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"
    for model_config in model_configs:
        metric_accuracy_mesh2ir_vs_rirbox(model_config, validation_csv)


if VISUALIZE_METRIC_ACCURACY:
    results_csvs = model_configs
    for i in range(len(results_csvs)):
        results_csvs[i] = "validation/results_acc/" + results_csvs[i].split("/")[-2] + "/" + results_csvs[i].split("/")[-1].split(".")[0] + ".csv" 

    for results_csv in results_csvs:
        view_results_metric_accuracy_mesh2ir_vs_rirbox(results_csv=results_csv)
