import os
import glob
from validation.metric_accuracy import metric_accuracy_mesh2ir_vs_rirbox, view_results_metric_accuracy_mesh2ir_vs_rirbox
from validation.sound_source_localization import sss_mesh2ir_vs_rirbox, view_results_sss_mesh2ir_vs_rirbox
from validation.visualize_all_results import view_all_validation_results, multiple_models_validation_comparison
import copy

DO_METRIC_ACCURACY = True
DO_SSL = False

VISUALIZE_METRIC_ACCURACY = True
VISUALIZE_SSL = False
VISUALIZE_ALL = False
COMPARE_ALL_RESULTS = False

# Best of /ablation6_Loss_Option_Subset_Architecture so far:
# "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_HIQMRSTFT_EDR_superfast.json",

model_configs = [
    "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_HIQMRSTFT_EDR_RT60_superfast.json",
    "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_HIQMRSTFT_EDR_RT60.json",
    "training/configs/ablation6_Loss_Option_Subset_Architecture/rirbox_Model2_dp_HIQMRSTFT_EDR_superfast.json",
]

results_csvs = copy.deepcopy(model_configs)
for i in range(len(results_csvs)):
    results_csvs[i] = results_csvs[i].split("/")[-2] + "/" + results_csvs[i].split("/")[-1].split(".")[0] + ".csv" 


if DO_METRIC_ACCURACY:
    validation_csv = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"
    for model_config in model_configs:
        metric_accuracy_mesh2ir_vs_rirbox(model_config, validation_csv, ISM_MAX_ORDER = 18)


if DO_SSL:
    validation_csv = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"
    for model_config in model_configs:
        print(model_config)
        sss_mesh2ir_vs_rirbox(model_config=model_config,
                              validation_csv=validation_csv,
                              validation_iterations=30,
                              RESPATIALIZE_RIRBOX=False,
                              ISM_MAX_ORDER = 15,
                              SHOW_TAU_PLOTS = False,
                              SHOW_SSL_PLOTS = False,
                              CONVOLVE_SIGNALS = False)
        

if VISUALIZE_METRIC_ACCURACY:
    for csv in results_csvs:
        view_results_metric_accuracy_mesh2ir_vs_rirbox("validation/results_acc/" + csv)


if VISUALIZE_SSL:    
    for csv in results_csvs:
        view_results_sss_mesh2ir_vs_rirbox("validation/results_sss/" + csv)


if VISUALIZE_ALL:
    for csv in results_csvs:
        view_all_validation_results(acc_csv="./validation/results_acc/" + csv,
                                            ssl_csv="validation/results_sss/" + csv)

if COMPARE_ALL_RESULTS:
    multiple_models_validation_comparison(results_csvs, acc_folder="./validation/results_acc/",
                                            ssl_folder="validation/results_sss/")