from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.close('all')

metric = 'r2'
method = 'riemann'
all_results = pd.read_csv(f'./results/camcan_different_subjects_results_method={method}.csv',
                          index_col=0)
all_results.reset_index(inplace=True, drop=True)
FIGURES_FOLDER = Path('./figures/')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
all_results = all_results.drop(all_results[all_results['method'] == 'align_procrustes'].index)
all_results = all_results.drop(all_results[all_results['method'] == 'dummy'].index)
all_results = all_results.replace(['no_alignment',
                                   'align_z_score',
                                   'align_recenter',
                                   'align_recenter_rescale',
                                   'align_procrustes',
                                   'align_procrustes_trunc',
                                   'align_procrustes_paired',
                                   'dummy'],
                                  ['No alignment',
                                   'z-score',
                                   'Re-center',
                                   'Re-scale',
                                   'Procrustes unpaired',
                                   'Procrustes truncated',
                                   'Procrustes paired',
                                   'Dummy'])

sns.set_theme(style="whitegrid", font_scale=2)
sns.set_palette('colorblind')
task_source = ['rest', 'rest', 'passive']
task_target = ['passive', 'smt', 'smt']
letters = ['A', 'B', 'C']

if metric == 'mae':
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
    for i in range(3):
        sns.boxplot(x="mae", y="method",
                    data=all_results[(
                        all_results['task_source'] == task_source[i]) & (all_results['task_target'] == task_target[i])
                    ],
                    orient='h', ax=axes[i])
        axes[i].set_xlim(6.5, 16.5)
        axes[i].set_xticks([8, 10, 12, 14, 16], ['8', '10', '12', '14', '16'])
        # Dummy score
        axes[i].axvline(x=16, color='k', linestyle='--')
        # Display elements
        axes[i].set_title(f'{task_source[i]}' + r'$\rightarrow$' + f'{task_target[i]}', x=0.4, y=1.04)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel(None)
        axes[i].annotate(text=letters[i], xy=(-0.1, 1.05), xycoords=('axes fraction'),
                         fontsize=30, weight='bold')
    fig.supxlabel("MAE", fontsize=22, x=0.58, y=0.08)

elif metric == 'r2':
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 6))
    for i in range(3):
        sns.boxplot(x="r2", y="method",
                    data=all_results[(
                        all_results['task_source'] == task_source[i]) & (all_results['task_target'] == task_target[i])
                    ],
                    orient='h', ax=axes[i])
        axes[i].set_xlim(-0.02, 0.81)
        axes[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8], ['0', '0.2', '0.4', '0.6', '0.8'])
        # Dummy score
        axes[i].axvline(x=0, color='k', linestyle='--')
        # Display elements
        axes[i].set_title(f'{task_source[i]}' + r'$\rightarrow$' + f'{task_target[i]}', x=0.4, y=1.04)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel(None)
        axes[i].annotate(text=letters[i], xy=(-0.1, 1.05), xycoords=('axes fraction'),
                         fontsize=30, weight='bold')
    fig.supxlabel(r"$R^2$", fontsize=22, x=0.58, y=0.08)

fig.tight_layout(w_pad=0)

if metric == 'mae':
    plt.savefig(
        FIGURES_FOLDER / f"camcan_different_subjects_method={method}_mae.pdf"
    )
if metric == 'r2':
    plt.savefig(
        FIGURES_FOLDER / f"camcan_different_subjects_method={method}_r2.pdf"
    )
