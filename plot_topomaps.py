from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from pyriemann.preprocessing import Whitening
from pyriemann.spatialfilters import SPoC
from utils.spatial_filter import ProjCommonSpace

import h5io

plt.close('all')

BIDS_PATH_SOURCE = Path('/storage/store2/data/TUAB-healthy-bids-bv')
BIDS_PATH_TARGET = Path('/storage/store3/data/LEMON_EEG_BIDS')
DERIVATIVES_PATH_SOURCE = Path(
    '/storage/store3/derivatives/TUAB-healthy-bids3'
)
DERIVATIVES_PATH_TARGET = Path(
    '/storage/store3/derivatives/LEMON_EEG_BIDS_2'
)
FIGURES_FOLDER = Path('./figures/')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

scale = 1
rank_source = 19
rank_target = 19
reg = 0

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}

# Get info files for TUAB and LEMON
source_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                   'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2',
                   'Fz', 'Cz', 'Pz']
target_channels = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'AFz',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7',
    'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4', 'FT8',
    'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2',
    'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
common_channels_source = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                          'T3', 'T4', 'T5', 'T6', 'O1', 'O2',
                          'F7', 'F8', 'Fz', 'Cz', 'Pz']
common_channels_target = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                          'T7', 'T8', 'P7', 'P8', 'O1', 'O2',
                          'F7', 'F8', 'Fz', 'Cz', 'Pz']
tuab_info_path = DERIVATIVES_PATH_SOURCE / 'sub-00000021/ses-001/eeg/sub-00000021_ses-001_task-rest_proc-autoreject_epo.fif'
epochs_source = mne.read_epochs(tuab_info_path)
info_source = epochs_source.pick(common_channels_source).info

lemon_info_path = DERIVATIVES_PATH_TARGET / 'sub-010002/eeg/sub-010002_task-RSEEG_proc-autoreject_epo.fif'
epochs_target = mne.read_epochs(lemon_info_path)
info_target = epochs_target.pick(common_channels_target).info


def no_alignment(X_source, X_target):
    return X_source, X_target


def prepare_data(common_channels):
    # Read info about subjects
    df_subjects_source = pd.read_csv(BIDS_PATH_SOURCE / "participants.tsv",
                                     sep='\t')
    df_subjects_target = pd.read_csv(BIDS_PATH_TARGET / "participants.tsv",
                                     sep='\t')
    df_subjects_source = df_subjects_source.set_index('participant_id')

    df_subjects_target = df_subjects_target.set_index('participant_id')

    # Read features
    features_source = h5io.read_hdf5(
        DERIVATIVES_PATH_SOURCE / 'features_fb_covs_rest.h5')
    features_target = h5io.read_hdf5(
        DERIVATIVES_PATH_TARGET / 'features_fb_covs_pooled.h5')

    # Get subjects ID
    subjects_source = list(df_subjects_source.index)
    subjects_source = [sub for sub in subjects_source
                       if sub in features_source.keys()]
    subjects_target = list(df_subjects_target.index)
    subjects_target = [sub for sub in subjects_target
                       if sub in features_target.keys()]

    # Get age
    y_source = [df_subjects_source.loc[sub].age for sub in subjects_source]
    y_target = [df_subjects_target.loc[sub].age for sub in subjects_target]

    # Get covariances
    covs_source = [features_source[sub]['covs'] for sub in subjects_source]
    covs_target = [features_target[sub]['covs'] for sub in subjects_target]
    X_source_ = np.array(covs_source)
    X_target_ = np.array(covs_target)

    # Select common channels
    if common_channels:
        source_index = [source_channels.index(ch) for ch in common_channels_source]
        target_index = [target_channels.index(ch) for ch in common_channels_target]
        X_source = X_source_[:, :, source_index][:, :, :, source_index]
        X_target = X_target_[:, :, target_index][:, :, :, target_index]
    # or match covariances size
    else:
        X_source = X_source_
        X_target = X_target_

    return X_source, X_target, y_source, y_target, np.array(subjects_source), np.array(subjects_target)


def run_func(X_source, X_target, func, scale, rank_source, rank_target, reg, common_channels):
    # Dimension reduction
    X_source_pca = np.zeros((X_source.shape[0],
                             X_source.shape[1],
                             rank_source, rank_source))
    X_target_pca = np.zeros((X_target.shape[0],
                             X_target.shape[1],
                             rank_target, rank_target))
    proj_source = []
    proj_target = []
    whitenings_source = []
    whitenings_target = []
    for f in range(len(frequency_bands)):
        proj = ProjCommonSpace(scale=scale, n_compo=rank_source, reg=reg)
        X_source_pca[:, f] = proj.fit_transform(X_source[:, f])
        proj_source.append(proj)
        if common_channels:
            X_target_pca[:, f] = proj.transform(X_target[:, f])
            proj_target.append(proj)
        else:
            proj = ProjCommonSpace(scale=scale, n_compo=rank_target, reg=reg)
            X_target_pca[:, f] = proj.fit_transform(X_target[:, f])
            proj_target.append(proj)
    # Alignment
    X_source_pca = X_source
    X_target_pca = X_target
    X_source_aligned = np.zeros_like(X_source_pca)
    X_target_aligned = np.zeros_like(X_target_pca)
    for i in range(len(frequency_bands)):
        if func == 'align_recenter':
            whitening_source = Whitening(metric='riemann')
            X_source_aligned[:, i] = whitening_source.fit_transform(X_source_pca[:, i])
            whitenings_source.append(whitening_source)
            whitening_target = Whitening(metric='riemann')
            X_target_aligned[:, i] = whitening_target.fit_transform(X_target_pca[:, i])
            whitenings_target.append(whitening_target)
        else:
            X_source_aligned[:, i], X_target_aligned[:, i] = X_source_pca[:, i], X_target_pca[:, i]
    return X_source_aligned, X_target_aligned, proj_source, proj_target, whitenings_source, whitenings_target


X_source, X_target, y_source, y_target, subjects_source, subjects_target = prepare_data(common_channels=True)

(X_source_aligned, X_target_aligned, proj_source, proj_target,
 whitenings_source, whitenings_target) = run_func(X_source, X_target,
                                                  'align_recenter', scale,
                                                  rank_source, rank_target, reg,
                                                  common_channels=False)

band = 3

X_source_ = X_source[:, band]
X_target_ = X_target[:, band]
X_source_aligned_ = X_source_aligned[:, band]
X_target_aligned_ = X_target_aligned[:, band]

spoc = SPoC(nfilter=20)

fig = plt.figure(figsize=(18, 7.5), layout='constrained')

subfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[4, 3])

subsubfigs = subfigs[0].subfigures(3, 1)
# Topomaps
n_topos = 5
# Source patterns
subsubfigs[0].suptitle('Source patterns', y=0.9, fontsize=23)
axes = subsubfigs[0].subplots(1, n_topos, width_ratios=[4] * n_topos)
axes[0].annotate(text='A', xy=(-0.1, 1.2), xycoords=('axes fraction'),
                 fontsize=30, weight='bold')
spoc.fit(X_source_, y_source)
evoked = mne.EvokedArray(normalize(spoc.patterns_.T, axis=1, norm='max'), info_source)
evoked.plot_topomap(times=evoked.times[:n_topos], time_format='', colorbar=False, axes=axes, show=False)

# Target not aligned
subsubfigs[1].suptitle('Target patterns', y=1, fontsize=23)
axes = subsubfigs[1].subplots(1, n_topos, width_ratios=[4] * n_topos)
spoc.fit(X_target_, y_target)
evoked = mne.EvokedArray(normalize(spoc.patterns_.T, axis=1, norm='max'), info_target)
evoked.plot_topomap(times=evoked.times[:n_topos], time_format='', colorbar=False, axes=axes, show=False)

# Target aligned
subsubfigs[2].suptitle('Source patterns applied to target data with alignment', y=1, fontsize=23)
axes = subsubfigs[2].subplots(1, n_topos, width_ratios=[4] * n_topos)
spoc.fit(X_source_aligned_, y_source)
pattern = whitenings_target[band].inv_filters_ @ spoc.patterns_.T
evoked = mne.EvokedArray(normalize(pattern, axis=0, norm='max'), info_target)
evoked.plot_topomap(times=evoked.times[:n_topos], time_format='', colorbar=False, axes=axes, show=False)

# Log power scatter plot
log_powers_source = np.mean(spoc.fit_transform(X_source_, y_source), axis=0)
log_powers_target = np.mean(spoc.transform(X_target_), axis=0)
log_powers_source_aligned = np.mean(spoc.fit_transform(X_source_aligned_, y_source), axis=0)
log_powers_target_aligned = np.mean(spoc.transform(X_target_aligned_), axis=0)

axes_scatter = subfigs[1].subplots(1, 1)

df = pd.DataFrame(
    {'Source log powers': np.concatenate((log_powers_source, log_powers_source_aligned)),
     'Target log powers': np.concatenate((log_powers_target, log_powers_target_aligned)),
     'Aligned': ['Without alignment'] * rank_source + ['With alignment'] * rank_source,
     }
)
axes_scatter.annotate(text='B', xy=(-0.2, 1.1), xycoords=('axes fraction'),
                      fontsize=30, weight='bold')
sns.set_theme(style="whitegrid", font_scale=2)
sns.set_palette('colorblind')
sns.scatterplot(x='Source log powers', y='Target log powers', data=df, hue='Aligned',
                style='Aligned', ax=axes_scatter, s=220, alpha=0.7)
axes_scatter.legend().set_title('')
xlims = [-1.1, 0.9]
ylims = [-1.1, 0.9]
axes_scatter.plot([-1.1, 1], [-1.1, 1], 'k--')
axes_scatter.axis("square")
axes_scatter.set_xlim(xlims)
axes_scatter.set_ylim(ylims)
axes_scatter.legend(loc='lower right')

plt.savefig(FIGURES_FOLDER / 'topomaps.pdf')
