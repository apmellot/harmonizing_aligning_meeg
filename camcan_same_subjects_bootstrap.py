import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import itertools
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

import h5io
import coffeine
from dameeg.recenter import align_recenter
from dameeg.recenter_rescale import align_recenter_rescale
from dameeg.procrustes import align_procrustes
from dameeg.z_score import align_z_score
from utils.spatial_filter import ProjCommonSpace

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


def align_procrustes_paired(X_source, X_target):
    return align_procrustes(X_source, X_target, method='paired')


def align_procrustes_trunc(X_source, X_target):
    return align_procrustes(X_source, X_target, method='truncated')


BIDS_PATH = Path('/data/parietal/store/data/camcan/BIDSsep/rest')
DERIVATIVES_PATH = Path(
    '/data/parietal/store3/work/amellot/derivatives/camcan/same_epochs'
)

TASKS = [
    ('rest', 'passive'), ('rest', 'smt'),
    ('passive', 'smt')
]

N_JOBS = 32
N_REPEATS = 100
rng = np.random.RandomState(seed)
# rng = np.random.RandomState(42)
RANDOM_STATES = rng.randint(0, 100000, N_REPEATS)
method = 'riemann'
scale = 1
reg = 0
rank = 65

DEBUG = False

func_list = [no_alignment,
             align_z_score,
             align_recenter,
             align_recenter_rescale,
             align_procrustes_paired]

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}

if DEBUG:
    TASKS = [('rest', 'passive')]
    RANDOM_STATES = [10]
    N_JOBS = 1
    # func_list = [align_z_score]


def prepare_data(task_source, task_target):
    # Read info about subjects
    df_subjects = pd.read_csv(BIDS_PATH / "participants.tsv", sep='\t')
    assert df_subjects.shape == df_subjects.drop_duplicates().shape
    df_subjects = df_subjects.set_index('participant_id')

    # Read features
    features_source = h5io.read_hdf5(
        DERIVATIVES_PATH / f'features_fb_covs_{task_source}.h5')
    features_target = h5io.read_hdf5(
        DERIVATIVES_PATH / f'features_fb_covs_{task_target}.h5')
    subjects_source = list(features_source.keys())
    subjects_target = list(features_target.keys())
    subjects_common = [sub for sub in df_subjects.index
                       if sub in subjects_target and sub in subjects_source]
    covs_source = [features_source[sub]['covs'] for sub in subjects_common]
    covs_target = [features_target[sub]['covs'] for sub in subjects_common]

    X_source = np.array(covs_source)
    X_target = np.array(covs_target)
    y = [df_subjects.loc[sub].age for sub in subjects_common]

    return X_source, X_target, np.array(y)


def run_func(X_source, X_target, func, scale, rank, reg, return_df=True):
    # Dimension reduction and regularization
    X_source_pca = np.zeros((X_source.shape[0],
                             X_source.shape[1],
                             rank, rank))
    X_target_pca = np.zeros((X_target.shape[0],
                             X_target.shape[1],
                             rank, rank))
    for f in range(len(frequency_bands)):
        proj = ProjCommonSpace(scale=scale, n_compo=rank, reg=reg)
        X_source_pca[:, f] = proj.fit_transform(X_source[:, f])
        X_target_pca[:, f] = proj.transform(X_target[:, f])

    # Alignment
    X_source_aligned = np.zeros_like(X_source_pca)
    X_target_aligned = np.zeros_like(X_target_pca)
    for i in range(len(frequency_bands)):
        X_source_aligned[:, i], X_target_aligned[:, i] = func(
            X_source_pca[:, i], X_target_pca[:, i])
    if return_df:
        X_source_aligned = pd.DataFrame(
            {band: list(X_source_aligned[:, i]) for i, band in
                enumerate(frequency_bands)})
        X_target_aligned = pd.DataFrame(
            {band: list(X_target_aligned[:, i]) for i, band in
                enumerate(frequency_bands)})
    return X_source_aligned, X_target_aligned


def run_model(X_source, X_target, y, task_source, task_target, scale,
              rank, reg):
    output = []
    # Model fitting and predition
    for func in func_list:
        # Alignment
        X_source_aligned, X_target_aligned = run_func(X_source, X_target,
                                                      func, scale, rank, reg,
                                                      return_df=True)
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
        model.fit(X_source_aligned, y)
        y_predicted = model.predict(X_target_aligned)
        mae = mean_absolute_error(y, y_predicted)
        r2 = r2_score(y, y_predicted)
        output.append(dict(
            task_source=task_source,
            task_target=task_target,
            method=func.__name__,
            mae=mae,
            r2=r2
        ))
        print('Tasks: ', (task_source, task_target),
              ', Method: ', func.__name__,
              ', MAE: ', mae,
              ', R2: ', r2)
    mae_dummy, r2_dummy = dummy(y, y)
    output.append(dict(
        task_source=task_source,
        task_target=task_target,
        method='dummy',
        mae=mae_dummy,
        r2=r2_dummy
    ))
    return output


def run_tasks(task_source, task_target, scale, rank, reg, random_state):
    X_source_full, X_target_full, y_full = prepare_data(task_source,
                                                        task_target)
    age_groups = [int(age // 10) for age in y_full]

    X_source, X_target, y = resample(X_source_full,
                                     X_target_full,
                                     y_full,
                                     stratify=age_groups,
                                     random_state=random_state)
    output = run_model(X_source, X_target, y, task_source, task_target,
                       scale, rank, reg)
    output = pd.DataFrame(output)
    output['random_state'] = random_state
    return output


# Results
results = Parallel(n_jobs=N_JOBS)(
    delayed(run_tasks)(task_source, task_target, scale, rank,
                       reg, random_state)
    for (task_source, task_target), random_state
    in itertools.product(TASKS, RANDOM_STATES)
)
results = pd.concat(results)
if DEBUG:
    results.to_csv('./results/camcan_same_subjects_bootstrap_debug.csv')
else:
    results.to_csv(f'./results/camcan_same_subjects_bootstrap_method={method}_{seed}.csv')
