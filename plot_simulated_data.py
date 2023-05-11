import numpy as np
import pandas as pd
from scipy.linalg import expm
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import seaborn as sns
from dameeg.simulation import simulate_reg_source_target
from dameeg.recenter import align_recenter
from dameeg.recenter_rescale import align_recenter_rescale
from dameeg.procrustes import align_procrustes

plt.close('all')

# Generate simulated data for all scenarios
SCENARIOS = ["Original", "Re-center", "Re-scale", "Rotation correction"]
RANDOM_STATE = 42
n_dim = 2
n_sources = 2
n_matrices = 300

rng = check_random_state(RANDOM_STATE)
Q = rng.randn(n_dim, n_dim)
Q = (Q - Q.T) / 2  # Make Q anti-symmetric
Q /= np.linalg.norm(Q)  # Make Q comparable to Identity
Q = expm(1 * Q)  # expm of an anti-symmetric matrix is a orthogonal

df_all = pd.DataFrame()

for i, SCENARIO in enumerate(SCENARIOS):
    sigma_n_source = 0
    sigma_n_target = 0
    sigma_y_source = 0
    sigma_y_target = 0
    sigma_A_source = 0
    sigma_A_target = 0.1
    sigma_p_source = 1
    sigma_p_target = 2
    mixing_difference = 2
    rotation = False
    X_source, y_source, X_target, y_target = simulate_reg_source_target(
        n_matrices, n_dim, n_sources, sigma_A_source, sigma_A_target,
        sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
        mixing_difference, sigma_p_source, sigma_p_target, RANDOM_STATE, rotation)

    if SCENARIO == "Original":
        X_source_aligned, X_target_aligned = X_source, X_target
    elif SCENARIO == "Re-center":
        X_source_aligned, X_target_aligned = align_recenter(X_source, X_target)
    elif SCENARIO == "Re-scale":
        X_source_aligned, X_target_aligned = align_recenter_rescale(X_source, X_target)
    elif SCENARIO == "Rotation correction":
        X_source_aligned, X_target_aligned = align_procrustes(X_source, X_target, method='paired')
    ts = TangentSpace()
    Xt_source_aligned = ts.fit_transform(X_source_aligned)
    Xt_target_aligned = ts.transform(X_target_aligned)
    pca = PCA(n_components=2)
    Xt_source_pca = pca.fit_transform(Xt_source_aligned)
    Xt_target_pca = pca.transform(Xt_target_aligned)
    if SCENARIO != "Rotation correction":
        Xt_target_pca = Xt_target_pca @ Q
    df_pca_source = pd.DataFrame({
        '1st compo': Xt_source_pca[:, 0],
        '2nd compo': Xt_source_pca[:, 1],
        'scenario': [SCENARIO] * Xt_source_pca.shape[0],
        'domain': ['Source'] * Xt_source_pca.shape[0]
    })
    df_pca_target = pd.DataFrame({
        '1st compo': Xt_target_pca[:, 0],
        '2nd compo': Xt_target_pca[:, 1],
        'scenario': [SCENARIO] * Xt_target_pca.shape[0],
        'domain': ['Target'] * Xt_target_pca.shape[0]
    })
    df_all = pd.concat([df_all, df_pca_source, df_pca_target])

# Plot the simulated data
sns.set_theme(style="whitegrid", font_scale=1.5)
sns.set_palette('colorblind')
sns.despine()
g = sns.FacetGrid(df_all, col="scenario",
                  col_wrap=2,
                  legend_out=True,
                  hue='domain',
                  hue_kws=dict(marker=['o', 'X']),
                  height=4, aspect=1,
                  margin_titles=True,
                  sharex=False, sharey=False
                  )
g.map_dataframe(sns.scatterplot, '1st compo', '2nd compo')
g.set_axis_labels('Principal component 1', 'Principal component 2')
g.add_legend(title='Domain')
g.set_titles(col_template='{col_name}')
letters = ['A', 'B', 'C', 'D']
for i, ax in enumerate(g.axes.ravel()):
    ax.annotate(text=letters[i], xy=(-0.1, 1.1), xycoords=('axes fraction'),
                fontsize=22, weight='bold')
sns.move_legend(g, "upper left", bbox_to_anchor=(0.83, 0.6), frameon=True)
g.tight_layout()
g.savefig('./figures/simulated_data.pdf')
