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
from dameeg.procrustes import align_procrustes


@pytest.mark.parametrize('method', ['paired', 'not paired'])
@pytest.mark.parametrize('rotation', [0, 5])
def test_align_procrustes_clf(rotation, method):
    X_source, y_source, X_target, y_target = simulate_clf_source_target(
        rotation=rotation)

    X_source_aligned, X_target_aligned = align_procrustes(X_source, X_target,
                                                          method=method)
    clf = MDM()
    clf.fit(X_source, y_source)
    score = clf.score(X_target, y_target)
    clf.fit(X_source_aligned, y_source)
    score_aligned = clf.score(X_target_aligned, y_target)
    if rotation == 0:
        if method == 'paired':
            assert np.linalg.norm(score - score_aligned) < 0.01
        else:
            assert score > 0.98
            assert score_aligned > 0.98
    else:
        assert score < score_aligned


@pytest.mark.parametrize('rotation', [True])
@pytest.mark.parametrize('mixing_difference', [0, 1])
@pytest.mark.parametrize('method', ['paired', 'not paired'])
def test_align_procrustes_reg(mixing_difference, rotation, method):
    X_source, y_source, X_target, y_target = simulate_reg_source_target(
        mixing_difference=mixing_difference, rotation=rotation)

    X_source_aligned, X_target_aligned = align_procrustes(X_source, X_target,
                                                          method=method)
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
    if mixing_difference == 0:
        assert mae < 1e-5
        assert mae_aligned < 1e-5
    else:
        assert mae > mae_aligned
