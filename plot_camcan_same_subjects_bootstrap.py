from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.close('all')

metric = 'r2'
method = 'riemann'
all_results = pd.read_csv(f'./results/camcan_same_subjects_bootstrap_method={method}.csv',
                          index_col=0)
all_results.reset_index(inplace=True, drop=True)
FIGURES_FOLDER = Path('./figures/')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
all_results = all_results.drop(all_results[all_results['method'] == 'align_procrustes'].index)
all_results = all_results.drop(all_results[all_results['method'] == 'dummy'].index)
all_results = all_results.replace(['no_alignment',
                                   'align_recenter',
                                   'align_recenter_rescale',
                                   'align_z_score',
                                   'align_procrustes',
                                   'align_procrustes_trunc',
                                   'align_procrustes_paired',
                                   'dummy'],
                                  ['No alignment',
                                   'Re-center',
                                   'Re-scale',
                                   'z-score',
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
    fig, axes = plt.subplots(1, 12, sharey=True, figsize=(16, 7),
                             gridspec_kw={'width_ratios': [1, 0.05, 0.125, 0.2] * 3,
                                          'wspace': 0})
    for i in range(3):
        ax1 = axes[4 * i]
        ax2 = axes[4 * i + 1]
        ax3 = axes[4 * i + 2]
        ax4 = axes[4 * i + 3]
        ax1.set_xlim(2, 11)
        ax3.set_xlim(15.5, 16.5)
        sns.boxplot(x="mae", y="method",
                    data=all_results[(
                        all_results['task_source'] == task_source[i]) & (all_results['task_target'] == task_target[i])
                    ], orient='h', ax=ax1)
        sns.boxplot(x="mae", y="method",
                    data=all_results[(
                        all_results['task_source'] == task_source[i]) & (all_results['task_target'] == task_target[i])
                    ], orient='h', ax=ax3)
        ax1.spines.right.set_visible(False)
        ax1.set_xticks([4, 6, 8, 10], ['4', '6', '8', '10'])
        ax3.spines.left.set_visible(False)
        ax3.set_xticks([16], ['16'])
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=ax3.transAxes)
        ax3.plot((-8 * d, +8 * d), (-d, +d), **kwargs)
        ax3.plot((-8 * d, +8 * d), (1 - d, 1 + d), **kwargs)
        # Dummy score
        ax3.axvline(x=16, color='k', linestyle='--')
        # Display elements
        ax2.set_visible(False)
        ax4.set_visible(False)
        ax1.set_title(f'{task_source[i]}' + r'$\rightarrow$' + f'{task_target[i]}', x=0.7, y=1.04)
        ax1.set_xlabel(None)
        ax1.set_ylabel(None)
        ax3.set_xlabel(None)
        ax3.set_ylabel(None)
        ax1.annotate(text=letters[i], xy=(-0.15, 1.05), xycoords=('axes fraction'),
                     fontsize=30, weight='bold')
    fig.supxlabel("MAE", fontsize=22, x=0.58, y=0.08)

elif metric == 'r2':
    fig, axes = plt.subplots(1, 9, sharey=True, figsize=(17, 6),
                             gridspec_kw={'width_ratios': [0.25, 2, 0.1] * 3})
    for i in range(3):
        # Broken axis
        ax1 = axes[3 * i]
        ax2 = axes[3 * i + 1]
        ax3 = axes[3 * i + 2]
        sns.boxplot(x="r2", y="method",
                    data=all_results[(
                        all_results['task_source'] == task_source[i]) & (all_results['task_target'] == task_target[i])
                    ],
                    orient='h', ax=ax1)
        sns.boxplot(x="r2", y="method",
                    data=all_results[(
                        all_results['task_source'] == task_source[i]) & (all_results['task_target'] == task_target[i])
                    ],
                    orient='h', ax=ax2)
        ax1.spines.right.set_visible(False)
        ax1.set_xticks([0], ['0'])
        ax2.spines.left.set_visible(False)
        ax2.tick_params(left=False, labelleft=False)
        ax2.set_xticks([0.4, 0.6, 0.8, 1], ['0.4', '0.6', '0.8', '1'])
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1 - 8 * d, 1 + 8 * d), (-d, +d), **kwargs)
        ax1.plot((1 - 8 * d, 1 + 8 * d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax1.set_xlim(-0.01, 0.05)
        ax2.set_xlim(0.4, 1)
        # Dummy score
        ax1.axvline(x=0, color='k', linestyle='--')
        # Display elements
        ax3.set_visible(False)
        ax2.set_title(f'{task_source[i]}' + r'$\rightarrow$' + f'{task_target[i]}', x=0.3, y=1.04)
        ax1.set_xlabel(None)
        ax1.set_ylabel(None)
        ax2.set_xlabel(None)
        ax2.set_ylabel(None)
        ax1.annotate(text=letters[i], xy=(-1, 1.05), xycoords=('axes fraction'),
                     fontsize=30, weight='bold')
    fig.supxlabel(r"$R^2$", fontsize=22, x=0.58, y=0.08)

fig.tight_layout(w_pad=0)

if metric == 'mae':
    plt.savefig(
        FIGURES_FOLDER / f"camcan_same_subjects_bootstrap_method={method}_mae.pdf"
    )
if metric == 'r2':
    plt.savefig(
        FIGURES_FOLDER / f"camcan_same_subjects_bootstrap_method={method}_r2.pdf"
    )
