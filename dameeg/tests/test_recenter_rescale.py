import numpy as np
import pytest

from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from dameeg.simulation import simulate_clf_source_target
from dameeg.simulation import simulate_reg_source_target
from dameeg.recenter_rescale import align_recenter_rescale


@pytest.mark.parametrize('scale', [1, 2])
def test_align_recenter_rescale_clf(scale):
    X_source, y_source, X_target, y_target = simulate_clf_source_target(
        scale=scale)
    X_source_aligned, X_target_aligned = align_recenter_rescale(X_source, X_target)
    clf = MDM()
    clf.fit(X_source, y_source)
    score = clf.score(X_target, y_target)
    clf.fit(X_source_aligned, y_source)
    score_aligned = clf.score(X_target_aligned, y_target)
    assert np.linalg.norm(score - score_aligned) < 0.01  # XXX the way we add a scale in this classif problem make no difference
    # if scale == 1:
    #     assert score == score_aligned
    # else:
    #     assert score < score_aligned


@pytest.mark.parametrize('sigma_p_target', [1, 2])
def test_align_recenter_rescale_reg(sigma_p_target):
    X_source, y_source, X_target, y_target = simulate_reg_source_target(
        sigma_p_target=sigma_p_target)

    X_source_aligned, X_target_aligned = align_recenter_rescale(X_source,
                                                                X_target)
    ts = TangentSpace()
    Xt_source = ts.fit_transform(X_source)
    Xt_target = ts.transform(X_target)
    Xt_source_aligned = ts.fit_transform(X_source_aligned)
    Xt_target_aligned = ts.fit_transform(X_target_aligned)
    reg = make_pipeline(StandardScaler(),
                        VarianceThreshold(1e-10),
                        RidgeCV(alphas=np.logspace(-5, 10, 100)))
    reg.fit(Xt_source, y_source)
    y_predicted = reg.predict(Xt_target)
    mae = mean_absolute_error(y_target, y_predicted)
    reg.fit(Xt_source_aligned, y_source)
    y_predicted_aligned = reg.predict(Xt_target_aligned)
    mae_aligned = mean_absolute_error(y_target, y_predicted_aligned)
    if sigma_p_target == 1:
        assert mae < 1e-5
        assert mae_aligned < 1e-5
    else:
        assert mae > mae_aligned
