import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def view_all_validation_results(acc_csv="./validation/results_acc/***.csv",
                            ssl_csv="./validation/results_ssl/***.csv"):
    df_acc = pd.read_csv(acc_csv)
    df_ssl = pd.read_csv(ssl_csv)

    fig, axs = plt.subplots(1,6, figsize=(14, 5))
    fig.suptitle(f'Metric accuracy validation. MESH2IR vs {acc_csv.split("/")[-1].split(".")[0]}')

    # Prepare the data for the box plot
    model_names = ["Baseline", "RIRBOX"]#, "Hybrid"]
    colors = ['C0', 'C1', 'C2']

    mean_marker = Line2D([], [], color='w', marker='^', markerfacecolor='green', markersize=10, label='Mean')

    # EDR
    # axs[0].boxplot([df_acc["mesh2ir_edr"], df_acc["rirbox_edr"], df_acc["hybrid_edr"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[0].boxplot([df_acc["mesh2ir_edr"], df_acc["rirbox_edr"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[0].set_title('EDR')
    axs[0].set_ylabel('EDR Error')
    axs[0].legend(handles=[mean_marker])

    # MRSTFT
    # axs[1].boxplot([df_acc["mesh2ir_mrstft"], df_acc["rirbox_mrstft"], df_acc["hybrid_mrstft"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[1].boxplot([df_acc["mesh2ir_mrstft"], df_acc["rirbox_mrstft"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[1].set_title('MRSTFT')
    axs[1].set_ylabel('MRSTFT Error')
    axs[1].legend(handles=[mean_marker])

    # C80
    # axs[2].boxplot([df_acc["mesh2ir_c80"], df_acc["rirbox_c80"], df_acc["hybrid_c80"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[2].boxplot([df_acc["mesh2ir_c80"], df_acc["rirbox_c80"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[2].set_title('C80')
    axs[2].set_ylabel('C80 Error')
    axs[2].legend(handles=[mean_marker])

    # D
    # axs[3].boxplot([df_acc["mesh2ir_D"], df_acc["rirbox_D"], df_acc["hybrid_D"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[3].boxplot([df_acc["mesh2ir_D"], df_acc["rirbox_D"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[3].set_title('D')
    axs[3].set_ylabel('D Error')
    axs[3].legend(handles=[mean_marker])

    # RT60
    # axs[4].boxplot([df_acc["mesh2ir_rt60"], df_acc["rirbox_rt60"], df_acc["hybrid_rt60"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[4].boxplot([df_acc["mesh2ir_rt60"], df_acc["rirbox_rt60"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[4].set_title('RT60')
    axs[4].set_ylabel('RT60 Error')
    axs[4].legend(handles=[mean_marker])

    # SSL
    axs[5].boxplot([df_ssl["mse_mesh2ir"], df_ssl["mse_rirbox"]], labels=model_names, patch_artist=True, showmeans=True, showfliers=False)
    axs[5].set_title('SSL')
    axs[5].set_ylabel('Mean TODA Error')
    axs[5].legend(handles=[mean_marker], loc='lower left')
    axs[5].set_ylim(0,0.0005)

    for ax in axs:
        ax.grid(ls="--", alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

def multiple_models_validation_comparison(results_csvs, acc_folder="./validation/results_acc/",
                                        ssl_folder="validation/results_ssl/"):
    
    fig, ax = plt.subplots(figsize=(14, 9))

    for csv in results_csvs:
        df_acc = pd.read_csv(acc_folder + csv)
        df_ssl = pd.read_csv(ssl_folder + csv)

        # drop all columns that contain the word "hybrid"
        df_acc = df_acc[df_acc.columns.drop(list(df_acc.filter(regex='hybrid')))]
        df_ssl = df_ssl[df_ssl.columns.drop(list(df_ssl.filter(regex='hybrid')))]

        # drop all unnamed columns
        df_acc = df_acc.loc[:, ~df_acc.columns.str.contains('Unnamed')]
        df_ssl = df_ssl.loc[:, ~df_ssl.columns.str.contains('Unnamed')]

        # Get means and stds
        means_acc = df_acc.mean()
        means_ssl = df_ssl.mean()
        stds_acc = df_acc.std()
        stds_ssl = df_ssl.std()

        means = pd.concat([means_acc, means_ssl])
        stds = pd.concat([stds_acc, stds_ssl])

        mesh2ir_rows = [row for row in means.index if 'mesh2ir' in row]
        rirbox_rows = [row for row in means.index if 'rirbox' in row]

        means_mesh2ir = means[mesh2ir_rows]
        means_rirbox = means[rirbox_rows]
        stds_mesh2ir = stds[mesh2ir_rows]
        stds_rirbox = stds[rirbox_rows]

        # convert these series to np.arrays
        means_mesh2ir = means_mesh2ir.to_numpy()
        means_rirbox = means_rirbox.to_numpy()
        stds_mesh2ir = stds_mesh2ir.to_numpy()
        stds_rirbox = stds_rirbox.to_numpy()

        # normalize the means and stds by mesh2ir values
        normalized_means_rirbox = means_rirbox / means_mesh2ir
        normalized_stds_rirbox = stds_rirbox / means_mesh2ir

        # plot the means and stds as a single line plot with an error area around the line representing the stds
        metrics = ['EDR', 'MRSTFT', 'C80', 'D', 'RT60', 'SSL']

        data_for_plotting = pd.DataFrame({
            'Mean': normalized_means_rirbox,
            'Std': normalized_stds_rirbox
        }, index=metrics)

        data_for_plotting['Mean'].plot(kind='line', yerr=data_for_plotting['Std'], ax=ax, capsize=4, marker='o', linestyle='-', label=csv.split("/")[-1].split(".")[0])
        # data_for_plotting['Mean'].plot(kind='line', ax=ax, marker='o', linestyle='-', label=csv.split("/")[-1].split(".")[0])
        print("plotting", csv.split("/")[-1].split(".")[0])

    
    plt.axhline(y=1, color='black', linestyle='--', label='Baseline MESH2IR', lw=1, alpha=1)
    plt.title('Means and std of RIRBOX Validations normalized by MESH2IR validations')
    plt.ylabel('Normalized Value')
    plt.xlabel('Metric')
    plt.xticks(rotation=45)
    plt.grid(True, ls="dotted", alpha=0.8)
    plt.legend()
    plt.ylim(0, 3)
    plt.tight_layout()

    plt.show()

