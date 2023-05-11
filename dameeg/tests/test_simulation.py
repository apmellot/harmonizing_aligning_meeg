import numpy as np
import pytest

from dameeg.simulation import simulate_reg_source_target


@pytest.mark.parametrize('n', [(200, 20), (650, 65)])
def test_simulate_reg_source_target(n):
    n_matrices, n_dim = n
    X_source, y_source, X_target, y_target = simulate_reg_source_target(
        n_matrices=n_matrices,
        n_dim=n_dim,
        n_sources=n_dim,
        sigma_A_source=0, sigma_A_target=0,
        sigma_n_source=0, sigma_n_target=0,
        sigma_y_source=0, sigma_y_target=0,
        mixing_difference=1, sigma_p_source=1,
        sigma_p_target=1, random_state=42,
        rotation=True, translation=False,
        return_mixing_matrices=False
    )

    assert X_source.shape == (n_matrices, n_dim, n_dim)
    assert y_source.shape == (n_matrices,)
    assert X_target.shape == (n_matrices, n_dim, n_dim)
    assert y_target.shape == (n_matrices,)

    # check that the matrices are symmetric
    assert np.allclose(X_source, X_source.transpose(0, 2, 1))
    assert np.allclose(X_target, X_target.transpose(0, 2, 1))

    # check that the matrices are PSD
    assert np.all(np.linalg.eigvalsh(X_source) > 0)
    assert np.all(np.linalg.eigvalsh(X_target) > 0)

    # check condition number of the matrices
    assert np.all(np.linalg.cond(X_source, p=2) > 1e2)
    assert np.all(np.linalg.cond(X_source, p=2) < 1e6)
    assert np.all(np.linalg.cond(X_target, p=2) > 1e2)
    assert np.all(np.linalg.cond(X_target, p=2) < 1e6)
