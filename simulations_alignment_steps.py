from pathlib import Path
import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from coffeine.spatial_filters import ProjSPoCSpace
from coffeine.covariance_transformers import Riemann, LogDiag

from dameeg.simulation import simulate_reg_source_target
from dameeg.recenter import align_recenter
from dameeg.recenter_rescale import align_recenter_rescale
from dameeg.procrustes import align_procrustes
from dameeg.z_score import align_z_score


def no_alignment(X_source, X_target):
    return X_source, X_target


def dummy(y_source, y_target):
    mae_dummy = np.mean(np.abs(y_target - np.median(y_source)))
    r2_dummy = r2_score(y_target, [np.mean(y_source)] * len(y_target))
    return mae_dummy, r2_dummy


def align_procrustes_paired(X_source, X_target):
    return align_procrustes(X_source, X_target, method='paired')


RESULTS_FOLDER = Path('./results/')
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["Recenter", "Rescale", "Rotation", "same_mixing_noise_A"]

N_JOBS = -10
N_REPEATS = 100
rng = np.random.RandomState(42)
RANDOM_STATES = rng.randint(0, 10000, N_REPEATS)
n_dim = 20
n_sources = 20
n_matrices = 300
method = 'riemann'

DEBUG = False

func_list = [
    no_alignment,
    align_z_score,
    align_recenter,
    align_recenter_rescale,
    align_procrustes_paired,
    align_procrustes
]

if DEBUG:
    N_JOBS = 1
    RANDOM_STATES = [42]
    func_list = [align_z_score]


def _run_train(X_source_aligned, X_target_aligned, y_source,
               y_target, method, func):
    # Vectorization
    if method == 'riemann':
        vect = Riemann(metric='riemann', return_data_frame=False)
    elif method == 'spoc':
        spatial_filter = ProjSPoCSpace(shrink=0, scale=1, n_compo='full', reg=0)
        X_source_aligned = spatial_filter.fit_transform(X_source_aligned, y=y_source)
        X_target_aligned = spatial_filter.transform(X_target_aligned)
        vect = LogDiag(return_data_frame=False)
    Xt_source_aligned = vect.fit_transform(X_source_aligned)
    Xt_target_aligned = vect.transform(X_target_aligned)
    del X_source_aligned, X_target_aligned
    # Regression
    reg = Ridge(alpha=1)
    reg = make_pipeline(VarianceThreshold(1e-10), StandardScaler(), reg)
    reg.fit(Xt_source_aligned, y_source)
    y_predicted_aligned = reg.predict(Xt_target_aligned)
    mae_aligned = mean_absolute_error(y_target,
                                      y_predicted_aligned)
    r2_aligned = r2_score(y_target, y_predicted_aligned)
    return mae_aligned, r2_aligned


# @profile
def run_one(SCENARIO, n_sources, sigma_A_source, sigma_A_target,
            sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
            mixing_difference, sigma_p_source, sigma_p_target, method,
            RANDOM_STATE, rotation, translation):

    X_source, y_source, X_target, y_target = simulate_reg_source_target(
        n_matrices, n_dim, n_sources, sigma_A_source, sigma_A_target,
        sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
        mixing_difference, sigma_p_source, sigma_p_target, RANDOM_STATE,
        rotation, translation)

    output = []
    mae_dummy, r2_dummy = dummy(y_source, y_target)

    if SCENARIO == "Recenter":
        parameter = mixing_difference
    elif SCENARIO == "Rescale":
        parameter = sigma_p_target
    elif SCENARIO == "Rotation":
        parameter = mixing_difference
    elif SCENARIO == "Everything":
        parameter = mixing_difference
    elif SCENARIO == "same_mixing_noise_A":
        parameter = sigma_A_target

    for func in func_list:
        print('Scenario: ', SCENARIO,
              ', Method: ', func.__name__,
              ', Parameter value: ', parameter)
        X_source_aligned, X_target_aligned = func(X_source,
                                                  X_target)

        mae, r2 = _run_train(X_source_aligned, X_target_aligned,
                             y_source, y_target, method, func)
        output.append(dict(
            scenario=SCENARIO,
            parameter=parameter,
            method=func.__name__,
            mae=mae / mae_dummy,
            r2=r2
        ))
    output.append(dict(
        scenario=SCENARIO,
        parameter=parameter,
        method='dummy',
        mae=mae_dummy / mae_dummy,
        r2=r2_dummy
    ))
    return output


def run_scenarios(SCENARIO, method, RANDOM_STATE):
    # Init simulation parameters
    sigma_n_source = 0
    sigma_n_target = 0
    sigma_y_source = 0
    sigma_y_target = 0
    if SCENARIO == "Recenter":
        sigma_A_source = 0
        sigma_A_target = 0
        sigma_p_source = 1
        sigma_p_target = 1
        mixing_difference_list = np.linspace(0, 2, 5)
        rotation = False
        translation = True
        output = [
            run_one(
                SCENARIO, n_sources, sigma_A_source, sigma_A_target,
                sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
                mixing_difference, sigma_p_source, sigma_p_target, method,
                RANDOM_STATE, rotation, translation
            ) for mixing_difference in mixing_difference_list
        ]
    elif SCENARIO == "Rescale":
        sigma_A_source = 0
        sigma_A_target = 0
        sigma_p_target_list = np.linspace(0.5, 2, 7)
        sigma_p_source = sigma_p_target_list[2]
        mixing_difference = 0
        rotation = False
        translation = False
        output = [
            run_one(
                SCENARIO, n_sources, sigma_A_source, sigma_A_target,
                sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
                mixing_difference, sigma_p_source, sigma_p_target, method,
                RANDOM_STATE, rotation, translation
            ) for sigma_p_target in sigma_p_target_list
        ]
    elif SCENARIO == "Rotation":
        sigma_A_source = 0
        sigma_A_target = 0
        sigma_p_source = 1
        sigma_p_target = 1
        mixing_difference_list = np.linspace(0, 1, 5)
        rotation = True
        translation = False
        output = [
            run_one(
                SCENARIO, n_sources, sigma_A_source, sigma_A_target,
                sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
                mixing_difference, sigma_p_source, sigma_p_target, method,
                RANDOM_STATE, rotation, translation
            ) for mixing_difference in mixing_difference_list
        ]
    elif SCENARIO == "Everything":
        sigma_A_source = 0
        sigma_A_target = 0
        sigma_p_source = 1
        sigma_p_target = 1.5
        mixing_difference_list = np.linspace(0, 1, 5)
        rotation = True
        translation = False
        output = [
            run_one(
                SCENARIO, n_sources, sigma_A_source, sigma_A_target,
                sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
                mixing_difference, sigma_p_source, sigma_p_target, method,
                RANDOM_STATE, rotation, translation
            ) for mixing_difference in mixing_difference_list
        ]
    elif SCENARIO == "same_mixing_noise_A":
        sigma_A_target_list = np.logspace(-3, 0, 7)
        sigma_A_source = sigma_A_target_list[2]
        sigma_p_source = 1
        sigma_p_target = 1
        mixing_difference = 0
        rotation = False
        translation = False
        output = [
            run_one(
                SCENARIO, n_sources, sigma_A_source, sigma_A_target,
                sigma_n_source, sigma_n_target, sigma_y_source, sigma_y_target,
                mixing_difference, sigma_p_source, sigma_p_target, method,
                RANDOM_STATE, rotation, translation
            ) for sigma_A_target in sigma_A_target_list
        ]
    else:
        raise ValueError("Unknown scenario")

    output = sum(output, [])
    output = pd.DataFrame(output)
    output['random_state'] = RANDOM_STATE
    return output


results = Parallel(n_jobs=N_JOBS)(
    delayed(run_scenarios)(SCENARIO, method, RANDOM_STATE)
    for SCENARIO, RANDOM_STATE in itertools.product(SCENARIOS, RANDOM_STATES)
)
results = pd.concat(results)

if DEBUG:
    results.to_csv(RESULTS_FOLDER / 'simulations_alignment_steps_debug.csv')
else:
    results.to_csv(RESULTS_FOLDER / f'simulations_alignment_steps_method={method}.csv')
