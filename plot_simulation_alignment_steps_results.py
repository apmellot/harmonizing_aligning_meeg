from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.close('all')

method = 'riemann'
metric = 'r2'
DEBUG = False
all_results = pd.read_csv(
    f'./results/simulations_alignment_steps_method={method}.csv',
    index_col=0)
if DEBUG:
    all_results = pd.read_csv(
        './results/simulations_alignment_steps_debug.csv',
        index_col=0)

all_results.reset_index(inplace=True, drop=True)
FIGURES_FOLDER = Path('./figures')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

all_results = all_results.replace(['Recenter',
                                   'Rescale',
                                   'Rotation',
                                   'Everything',
                                   'same_mixing_noise_A'],
                                  ['Translation',  # : \n'+r'$A^{(t)} = t \mathbf{I}_P A^{(s)} $',
                                   'Scale',  # : \n'+r'$p_i^{(t)} = (p_i^{(s)})^{\sigma_p}$',
                                   'Translation and rotation',  # :\n'+r'$A^{(t)} = m A_m + (1 - m) A^{(s)}$',
                                   'Translation, scale and rotation',  #: \n'+r'$\sigma_p =1.5$ and $A^{(t)} = m A_m + (1 - m) A^{(s)}$'
                                   'Noise on mixing matrix'])

all_results = all_results.replace(['no_alignment',
                                   'align_recenter',
                                   'align_recenter_rescale',
                                   'align_procrustes',
                                   'align_procrustes_paired',
                                   'align_z_score',
                                   'dummy'],
                                  ['No alignment',
                                   'Recenter',
                                   'Rescale',
                                   'Procrustes unpaired',
                                   'Procrustes paired',
                                   'z-score',
                                   'Dummy'])

sns.set_theme(style="whitegrid", font_scale=1.3)
sns.set_palette('colorblind')
sns.despine()
g = sns.FacetGrid(all_results, col="scenario",
                  col_wrap=2,
                  legend_out=True,
                  hue='method',
                  hue_kws=dict(marker=['s', 'o', '<', 'X', 'v', '>', 'P'],
                               ls=['--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), '-']),
                  height=4, aspect=1,
                  margin_titles=True,
                  sharex=False)
if metric == 'r2':
    g.map_dataframe(sns.lineplot, 'parameter', 'r2', markersize=10)
    g.set(ylim=(-0.15, 1.1))
    g.set_axis_labels("Parameter value", "R2")
elif metric == 'mae':
    g.map_dataframe(sns.lineplot, 'parameter', 'mae', markersize=10)
    g.set(ylim=(-0.1, 1.2))
    # g.set(yscale='log')
    g.set_axis_labels("Parameter value", "Normalized MAE")
# g.set(xscale='log')
g.add_legend(title='Methods')
g.set_titles(col_template='{col_name}')
letters = ['A', 'B', 'C', 'D']
parameters = [r'$\alpha$', r'$\sigma_p$', r'$m$', r'$\sigma_{A}^{(t)}$']
for i, ax in enumerate(g.axes.ravel()):
    if i == 1:
        ax.axvline(x=1, c='gray', ls='dashed')
    if i == 3:
        ax.axvline(x=1e-2, c='gray', ls='dashed')
        ax.set_xscale('log')
    ax.set_xlabel(parameters[i])
    ax.annotate(text=letters[i], xy=(-0.1, 1.03), xycoords=('axes fraction'),
                fontsize=22, weight='bold')
sns.move_legend(g, "upper left", bbox_to_anchor=(0.75, 0.67), frameon=True)
g.tight_layout()
if DEBUG:
    g.savefig(
        FIGURES_FOLDER / "simulation_alignment_steps_debug.png"
    )
elif metric == 'r2':
    g.savefig(
        FIGURES_FOLDER / f"simulation_alignment_steps_method={method}_r2.pdf"
    )
elif metric == 'mae':
    g.savefig(
        FIGURES_FOLDER / f"simulation_alignment_steps_method={method}_mae.pdf"
    )
