import numpy as np
from scipy.linalg import expm
from pyriemann.tangentspace import TangentSpace
from pyriemann.datasets import make_gaussian_blobs
from pyriemann.utils.base import powm
from sklearn.utils import check_random_state


def _simulate_rotation(n_dim, rotation, random_state):
    """Simulate a rotation matrix more or less close to Id.

    Parameters
    ----------
    n_dim : int
        The dimension of the matrice.
    rotation : float
        If 0 no rotation and if > 0 then it's how much the rotation deviates
        from the identity matrix.
    random_state : int or None
        Random seed.

    Returns
    -------
    Q : ndarray shape (n_dim, n_dim)
        The rotation matrix.
    """
    rng = check_random_state(random_state)
    Q = rng.randn(n_dim, n_dim)
    Q = (Q - Q.T) / 2  # Make Q anti-symmetric
    Q /= np.linalg.norm(Q)  # Make Q comparable to Identity
    Q = rotation * Q  # Make Q deviate from zeros matrix
    Q = expm(Q)  # expm of an anti-symmetric matrix is a orthogonal
    return Q


def simulate_clf_source_target(n_matrices=1000, n_dim=2, scale=1, offset=0,
                               rotation=0, random_state=None):
    """Generate simulated source and target datasets with 2 classes.

    Parameters
    ----------
    n_matrices: int
        Number of matrices in each class.
    n_dim: int
        Dimension of the matrices.
    rotation: float
        Whether to add a rotation between the source and the target datasets.
        If 0 no rotation and if > 0 then it's how much the rotation deviates
        from the identity matrix.
    random_state: int or None
        Random seed.

    Returns
    ----------
    X_source: ndarray shape (2*n_matrices, n_dim, n_dim)
        Matrices of the source dataset.
    y_source: list
        Class labels of the source matrices.
    X_target: ndarray shape (2*n_matrices, n_dim, n_dim)
        Matrices of the target dataset.
    y_target: list
        Class labels of the target matrices.
    """

    n_matrices = n_matrices  # how many matrices to sample on each class
    n_dim = n_dim  # dimensionality of the data points
    sigma = 1.0  # dispersion of the Gaussian distributions
    random_state = 42  # ensure reproducibility

    delta = 3 * sigma

    # generate data points for a classification problem
    X_source, y = make_gaussian_blobs(
        n_matrices=n_matrices,
        n_dim=n_dim,
        class_sep=delta,
        class_disp=sigma,
        random_state=random_state,
    )

    ts = TangentSpace()
    Xt_source = ts.fit_transform(X_source)

    # Creation of the target data
    if rotation != 0:
        Q = _simulate_rotation(Xt_source.shape[1], rotation, random_state)
        Xt_target = scale * Xt_source @ Q + offset
    else:
        Xt_target = scale * Xt_source + offset

    X_target = ts.inverse_transform(Xt_target)

    y_source = y_target = y
    return X_source, y_source, X_target, y_target


def _generate_X_y(n_sources, A_list, powers, sigma_p, beta, sigma_n, sigma_y, rng):
    n_matrices = len(A_list)
    n_dim = A_list[0].shape[0]

    # Generate covariances
    Cs = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Cs[i, :n_sources, :n_sources] = np.diag(powers[i])**sigma_p  # set diag sources
        N_i = sigma_n * rng.randn(n_dim - n_sources, n_dim - n_sources)
        Cs[i, n_sources:, n_sources:] = N_i.dot(N_i.T)  # fill the noise block
    X = np.array([a.dot(cs).dot(a.T) for a, cs in zip(A_list, Cs)])

    # Generate y
    y = np.log(powers).dot(beta)  # + 50
    y += sigma_y * rng.randn(n_matrices)
    return X, y


def simulate_reg_source_target(n_matrices=1000, n_dim=2, n_sources=2,
                               sigma_A_source=0, sigma_A_target=0,
                               sigma_n_source=0, sigma_n_target=0,
                               sigma_y_source=0, sigma_y_target=0,
                               mixing_difference=0, sigma_p_source=1,
                               sigma_p_target=1, random_state=42,
                               rotation=False, translation=True,
                               return_mixing_matrices=False):
    """Generate simulated source and target datasets for a regression situation.

    Parameters
    ----------
    n_matrices: int
        Number of matrices in each class.
    n_dim: int
        Dimension of the matrices.
    n_sources: int
        Number of signal sources.
    sigma_A_source: float
        Variance of individual noise on source mixing matrices.
    sigma_A_target: float
        Variance of individual noise on target mixing matrices.
    sigma_y_source: float
        Variance of noise in source outcome.
    sigma_y_target: float
        Variance of noise in target outcome.
    sigma_p_source: float
        Scale parameter for the source power to compute the covariances.
    sigma_p_target: float
        Scale parameter for the target power to compute the covariances.
    mixing_difference: float
        Should have a value between 0 and 1. If mixing_difference = 0,
        A_target = A_source. If mixing_difference = 1, A_target is a
        completely different matrix.
    random_state: int
        Random seed used to initialize the pseudo-random number generator.
    return_mixing_matrices: bool
        Whether to return the mixing matrices A_source and A_target.

    Returns
    ----------
    X_source: ndarray shape (2*n_matrices, n_dim, n_dim)
        Matrices of the source dataset.
    y_source: list
        Labels of the source matrices.
    X_target: ndarray shape (2*n_matrices, n_dim, n_dim)
        Matrices of the target dataset.
    y_target: list
        Labels of the target matrices.
    """
    rng = check_random_state(random_state)

    # Generate A_source and A_target
    A_source = rng.randn(n_dim, n_dim)
    if rotation:
        A_target = rng.randn(n_dim, n_dim)
        A_target = mixing_difference * A_target + (1 - mixing_difference) * A_source
    elif translation:
        Pv = rng.randn(n_dim, n_dim)  # create random tangent vector
        Pv = (Pv + Pv.T) / 2  # symmetrize
        Pv /= np.linalg.norm(Pv)  # normalize
        P = expm(Pv)  # take it back to the SPD manifold
        M_target = powm(P, alpha=mixing_difference)  # control distance to identity
        A_target = M_target @ A_source
    else:
        A_target = A_source

    # Add individual noise
    A_list_source = [A_source + sigma_A_source * rng.randn(n_dim, n_dim)
                     for _ in range(n_matrices)]
    A_list_target = [A_target + sigma_A_target * rng.randn(n_dim, n_dim)
                     for _ in range(n_matrices)]

    beta = rng.randn(n_sources)
    # Generate powers
    powers = rng.uniform(low=0.01, high=1, size=(n_matrices, n_sources))
    X_source, y_source = _generate_X_y(n_sources, A_list_source, powers, sigma_p_source, beta,
                                       sigma_n_source, sigma_y_source, rng)
    X_target, y_target = _generate_X_y(n_sources, A_list_target, powers, sigma_p_target, beta,
                                       sigma_n_target, sigma_y_target, rng)

    if return_mixing_matrices:
        return X_source, y_source, A_source, X_target, y_target, A_target
    return X_source, y_source, X_target, y_target
