import numpy as np
import pytest

from pyriemann.tangentspace import TangentSpace

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

from dameeg.simulation import simulate_reg_source_target
from dameeg.z_score import align_z_score


@pytest.mark.parametrize('fisher_transform', [False])
@pytest.mark.parametrize('global_scaling', [False, True])
def test_align_z_score_reg(fisher_transform, global_scaling):
    X_source, y_source, X_target, y_target = simulate_reg_source_target(
        sigma_A_source=1e-2, sigma_A_target=3e-2, translation=False)

    X_source_aligned, X_target_aligned = align_z_score(X_source, X_target,
                                                       fisher_transform, global_scaling)
    ts = TangentSpace()
    Xt_source = ts.fit_transform(X_source)
    Xt_target = ts.transform(X_target)
    Xt_source_aligned = ts.fit_transform(X_source_aligned)
    Xt_target_aligned = ts.transform(X_target_aligned)
    reg = make_pipeline(StandardScaler(),
                        VarianceThreshold(1e-10),
                        RidgeCV(alphas=np.logspace(-5, 10, 100)))
    reg.fit(Xt_source, y_source)
    y_predicted = reg.predict(Xt_target)
    mae = mean_absolute_error(y_target, y_predicted)
    reg = clone(reg)
    reg.fit(Xt_source_aligned, y_source)
    y_predicted_aligned = reg.predict(Xt_target_aligned)
    mae_aligned = mean_absolute_error(y_target, y_predicted_aligned)
    assert mae_aligned > mae
