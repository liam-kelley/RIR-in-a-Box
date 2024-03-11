import os
import glob
from validation.metric_accuracy import metric_accuracy_mesh2ir_vs_rirbox_GWA
from validation.metric_accuracy import metric_accuracy_mesh2ir_vs_rirbox_HL2
from validation.metric_accuracy import view_results_metric_accuracy_mesh2ir_vs_rirbox, view_results_metric_accuracy_mesh2ir_vs_rirbox_multiple_models
from validation.sound_source_spatialization import sss_mesh2ir_vs_rirbox, view_results_sss_mesh2ir_vs_rirbox
from validation.visualize_all_results import view_all_validation_results, multiple_models_validation_comparison
from validation.beamforming import beamforming_mesh2ir_vs_rirbox
import copy

DO_METRIC_ACCURACY_HL2 = False
DO_METRIC_ACCURACY_GWA = False
DO_SSL = False
DO_BEAMFORMING = True

VISUALIZE_METRIC_ACCURACY = False
VISUALIZE_METRIC_ACCURACY_MULTIPLE_MODELS = False
VISUALIZE_SSL = True
VISUALIZE_ALL = False
COMPARE_ALL_RESULTS = False

RESPATIALIZE_RIRBOX = False

configs = glob.glob("training/configs/best_models/*.json")
configs.extend(glob.glob("training/configs/ablation13/*.json"))
configs = sorted(configs)


configs = [
    "training/configs/best_models/rirbox_Model2_dp_MRSTFT_EDR_superfast_MSDist_DistInLatent_NormByDist_12epochs.json",
    # "training/configs/best_models/rirbox_Model3_dp_HIQMRSTFT_EDR_superfast_MSDist_DistInLatent_noNormByDist_12epochs.json",
    "training/configs/ablation13/rirbox_Model3_dp_MRSTFT_EDR_superfast_MSDist_DistInLatent_noNormByDist_12pochs.json"
]

for config in configs:
    print(config)


results_csvs = copy.deepcopy(configs)
for i in range(len(results_csvs)):
    results_csvs[i] = results_csvs[i].split("/")[-2] + "/" + results_csvs[i].split("/")[-1].split(".")[0] + ".csv" 


if DO_METRIC_ACCURACY_HL2:
    validation_csv = "datasets/ValidationDataset/subsets/realval_dataset.csv"
    for model_config in configs:
        metric_accuracy_mesh2ir_vs_rirbox_HL2(model_config, validation_csv,
                                          RESPATIALIZE_RIRBOX=RESPATIALIZE_RIRBOX,
                                          ISM_MAX_ORDER = 18)

if DO_METRIC_ACCURACY_GWA:   
    validation_csv = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"
    for model_config in configs:
        metric_accuracy_mesh2ir_vs_rirbox_GWA(model_config, validation_csv,
                                          RESPATIALIZE_RIRBOX=RESPATIALIZE_RIRBOX,
                                          ISM_MAX_ORDER = 18)

if DO_SSL:
    validation_csv = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"
    for model_config in configs:
        print(model_config)
        sss_mesh2ir_vs_rirbox(model_config=model_config,
                              validation_csv=validation_csv,
                              validation_iterations=30,
                              RESPATIALIZE_RIRBOX=RESPATIALIZE_RIRBOX,
                              ISM_MAX_ORDER = 15,
                              SHOW_TAU_PLOTS = False,
                              SHOW_SSL_PLOTS = False,
                              CONVOLVE_SIGNALS = False)

if DO_BEAMFORMING:
    validation_csv = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"
    for model_config in configs:
        beamforming_mesh2ir_vs_rirbox(model_config, validation_csv,
                                      RESPATIALIZE_RIRBOX=RESPATIALIZE_RIRBOX,
                                      ISM_MAX_ORDER = 18)

if VISUALIZE_METRIC_ACCURACY:
    validation_csv = "datasets/GWA_3DFRONT/subsets/gwa_3Dfront_validation_dp_only.csv"
    for model_config in configs:
        metric_accuracy_mesh2ir_vs_rirbox_HL2(model_config, validation_csv,
                                          RESPATIALIZE_RIRBOX=RESPATIALIZE_RIRBOX,
                                          ISM_MAX_ORDER = 18)


if VISUALIZE_METRIC_ACCURACY_MULTIPLE_MODELS:
    view_results_metric_accuracy_mesh2ir_vs_rirbox_multiple_models(results_csvs)


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