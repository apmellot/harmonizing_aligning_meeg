from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.close('all')

metric = 'r2'
lemon_in_training = False

results_riemann = pd.read_csv('./results/eeg_tuab_lemon_results_method=riemann.csv',
                              index_col=0)
results_riemann.reset_index(inplace=True, drop=True)
FIGURES_FOLDER = Path('./figures')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
dummy_riemann_r2 = np.mean(results_riemann[results_riemann['method'] == 'dummy'].r2.values)
dummy_riemann_mae = np.mean(results_riemann[results_riemann['method'] == 'dummy'].mae.values)
results_riemann = results_riemann.drop(results_riemann[results_riemann['method'] == 'dummy'].index)
results_riemann = results_riemann.drop(results_riemann[results_riemann['method'] == 'align_procrustes'].index)
results_riemann = results_riemann.replace(['no_alignment',
                                           'align_z_score',
                                           'align_recenter',
                                           'align_recenter_rescale',
                                           'align_procrustes',
                                           'align_procrustes_trunc',
                                           'align_procrustes_paired'],
                                          ['No\n alignment',
                                           'z-score',
                                           'Re-center',
                                           'Re-scale',
                                           'Procrustes',
                                           'Procrustes truncated',
                                           'Procrustes paired'])
results_spoc = pd.read_csv('./results/eeg_tuab_lemon_results_method=spoc.csv',
                           index_col=0)
results_spoc.reset_index(inplace=True, drop=True)
FIGURES_FOLDER = Path('./figures')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
dummy_spoc_r2 = np.mean(results_spoc[results_spoc['method'] == 'dummy'].r2.values)
dummy_spoc_mae = np.mean(results_spoc[results_spoc['method'] == 'dummy'].mae.values)
results_spoc = results_spoc.drop(results_spoc[results_spoc['method'] == 'dummy'].index)
results_spoc = results_spoc.drop(results_spoc[results_spoc['method'] == 'align_procrustes'].index)
results_spoc = results_spoc.replace(['no_alignment',
                                     'align_z_score',
                                     'align_recenter',
                                     'align_recenter_rescale',
                                     'align_procrustes',
                                     'align_procrustes_trunc',
                                     'align_procrustes_paired'],
                                    ['No\n alignment',
                                     'z-score',
                                     'Re-center',
                                     'Re-scale',
                                     'Procrustes',
                                     'Procrustes truncated',
                                     'Procrustes paired'])

sns.set_theme(style="whitegrid", font_scale=2)
sns.set_palette('colorblind')

fig, axes = plt.subplots(1, 2, figsize=(19, 7))

if metric == 'r2':
    # Riemann results
    sns.boxplot(x="r2", y="method", data=results_riemann, orient='h', ax=axes[0])
    axes[0].set_ylabel(None)
    axes[0].axvline(x=dummy_riemann_r2, color='k', linestyle='--')
    axes[0].axvline(x=0.54, color='gray', linestyle='--')
    axes[0].set_xlabel(r"$R^2$")
    axes[0].annotate(text='A', xy=(-0.1, 1.05), xycoords=('axes fraction'),
                     fontsize=30, weight='bold')
    axes[0].set_title('Riemann', y=1.05)
    # Spoc results
    sns.boxplot(x="r2", y="method", data=results_spoc, orient='h', ax=axes[1])
    axes[1].set_ylabel(None)
    axes[1].axvline(x=dummy_spoc_r2, color='k', linestyle='--')
    axes[1].axvline(x=0.54, color='gray', linestyle='--')
    axes[1].set_xlabel(r"$R^2$")
    axes[1].annotate(text='B', xy=(-0.1, 1.05), xycoords=('axes fraction'),
                     fontsize=30, weight='bold')
    axes[1].set_title('SPoC', y=1.05)

elif metric == 'mae':
    # Riemann results
    sns.boxplot(x="mae", y="method", data=results_riemann, orient='h', ax=axes[0])
    axes[0].set_ylabel(None)
    axes[0].set_xlabel("MAE")
    if not lemon_in_training:
        axes[0].axvline(x=dummy_riemann_mae, color='k', linestyle='--')
    axes[0].set_title('Riemann', y=1.05)
    axes[0].annotate(text='A', xy=(-0.1, 1.05), xycoords=('axes fraction'),
                     fontsize=30, weight='bold')
    # Spoc results
    sns.boxplot(x="mae", y="method", data=results_spoc, orient='h', ax=axes[1])
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel(None)
    if not lemon_in_training:
        axes[1].axvline(x=dummy_spoc_mae, color='k', linestyle='--')
    axes[1].annotate(text='B', xy=(-0.1, 1.05), xycoords=('axes fraction'),
                     fontsize=30, weight='bold')
    axes[1].set_title('SPoC', y=1.05)

plt.tight_layout()

if metric == 'r2':
    plt.savefig(
        FIGURES_FOLDER / "eeg_tuab_lemon_r2.pdf"
    )
elif metric == 'mae':
    plt.savefig(
        FIGURES_FOLDER / "eeg_tuab_lemon_mae.pdf"
    )
