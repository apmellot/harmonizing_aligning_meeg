import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

import h5io
import coffeine

from dameeg.recenter import align_recenter
from dameeg.recenter_rescale import align_recenter_rescale
from dameeg.procrustes import align_procrustes
from dameeg.z_score import align_z_score

parser = argparse.ArgumentParser(description="Run CamCAN same subjects.")
parser.add_argument('-s', '--seed', default=42, help='Random seed')
args = parser.parse_args()
seed = int(args.seed)


def no_alignment(X_source, X_target):
    return X_source, X_target


def dummy(y_source, y_target):
    mae_dummy = np.mean(np.abs(y_target - np.median(y_source)))
    r2_dummy = r2_score(y_target, [np.mean(y_source)] * len(y_target))
    return mae_dummy, r2_dummy


BIDS_PATH_SOURCE = Path('/data/parietal/store3/data/TUAB_healthy_BIDS')
BIDS_PATH_TARGET = Path('/data/parietal/store3/data/LEMON_EEG_BIDS')
DERIVATIVES_PATH_SOURCE = Path('/data/parietal/store3/work/amellot/derivatives/TUAB_healthy')
DERIVATIVES_PATH_TARGET = Path(
    '/data/parietal/store3/work/amellot/derivatives/LEMON'
)

N_JOBS = 64
N_REPEATS = 100
# rng = np.random.RandomState(42)
rng = np.random.RandomState(seed)
RANDOM_STATES = rng.randint(0, 100000, N_REPEATS)
method = 'riemann'
scale = 1
rank = 15
reg = 0

DEBUG = False

if DEBUG:
    RANDOM_STATES = [42]
    N_JOBS = 1
    func_list = [align_z_score]


func_list = [no_alignment,
             align_z_score,
             align_recenter,
             align_recenter_rescale,
             align_procrustes]

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}

mem = Memory(".", verbose=False)
func_list = [mem.cache(func) for func in func_list]


def prepare_data():
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
    common_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                       'O2', 'F7', 'F8', 'Fz', 'Cz', 'Pz']
    source_index = [source_channels.index(ch) for ch in common_channels]
    target_index = [target_channels.index(ch) for ch in common_channels]
    X_source = X_source_[:, :, source_index][:, :, :, source_index]
    X_target = X_target_[:, :, target_index][:, :, :, target_index]

    return X_source, X_target, y_source, y_target


def run_func(X_source, X_target, func):
    X_source_aligned = np.zeros_like(X_source)
    X_target_aligned = np.zeros_like(X_target)
    for i in range(len(frequency_bands)):
        X_source_aligned[:, i], X_target_aligned[:, i] = func(
            X_source[:, i], X_target[:, i])
    return X_source_aligned, X_target_aligned


def run_model(X_source, X_target, y_source, y_target, scale,
              rank, reg, random_state):
    output = []
    # CV
    X_source, y_source = resample(X_source, y_source,
                                  random_state=random_state)
    for func in func_list:
        # Alignment
        X_source_aligned, X_target_aligned = run_func(X_source, X_target, func)
        X_train = X_source_aligned
        y_train = np.array(y_source)
        X_test = X_target_aligned
        y_test = np.array(y_target)
        # Store in DataFrame
        X_train = pd.DataFrame(
            {band: list(X_train[:, i]) for i, band in
                enumerate(frequency_bands)})
        X_test = pd.DataFrame(
            {band: list(X_test[:, i]) for i, band in
                enumerate(frequency_bands)})
        # Regression model
        if method == 'riemann':
            filter_bank_transformer = coffeine.make_filter_bank_transformer(
                names=list(frequency_bands),
                method='riemann',
                projection_params=dict(scale=scale, n_compo=rank, reg=reg)
            )
        elif method == 'spoc':
            filter_bank_transformer = coffeine.make_filter_bank_transformer(
                names=list(frequency_bands),
                method='spoc',
                projection_params=dict(scale=scale, n_compo=rank, reg=reg)
            )
        model = make_pipeline(
            filter_bank_transformer,
            VarianceThreshold(1e-10),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100))
        )
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_predicted)
        r2 = r2_score(y_test, y_predicted)
        output.append(dict(
            method=func.__name__,
            mae=mae,
            r2=r2
        ))
        print('Method: ', func.__name__,
              ', MAE: ', mae,
              ', R2: ', r2)
        mae_dummy, r2_dummy = dummy(y_train, y_test)
        output.append(dict(
            method='dummy',
            mae=mae_dummy,
            r2=r2_dummy
        ))
    return output


def run_cv(scale, rank, reg, random_state):
    X_source, X_target, y_source, y_target = prepare_data()
    output = run_model(X_source, X_target, y_source, y_target, scale,
                       rank, reg, random_state)
    output = pd.DataFrame(output)
    output['random_state'] = random_state
    return output


# Results
results = Parallel(n_jobs=N_JOBS)(
    delayed(run_cv)(scale, rank, reg, random_state)
    for random_state in RANDOM_STATES
)
results = pd.concat(results)

if DEBUG:
    results.to_csv('./results/eeg_tuab_lemon_results_debug.csv')
else:
    results.to_csv(f'./results/eeg_tuab_lemon_results_method={method}_{seed}.csv')
