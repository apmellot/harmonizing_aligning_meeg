import numpy as np

from pyriemann.tangentspace import TangentSpace
from .recenter_rescale import align_recenter_rescale


def align_procrustes(X_source, X_target, method='not paired', n_compo_rot=100):
    """Method to align source and target data with a Riemannian geometry
    framework inspired by [1].

    The input are MEEG covariances. First, covariances from both domains are
    re-centered to identity by whitening them with the mean covariance of each
    domain. The dispersion of the target domain is then adjusted to be equal
    to the dispersion of the source domain. Finally a rotation correction is
    applied either only to the target data like in [2] or to both source and
    target data as proposed in [3].

    Parameters
    ----------
    X_source: ndarray, shape (n_samples_source, n_channels, n_channels)
        Covariances of the source domain.
    X_target: ndarray, shape (n_samples_target, n_channels, n_channels)
        Covariances of the target domain.
    method: str, optional
        The method used to rotate data. Defaults to 'paired'. Can be
        'not paired' or 'paired'. The 'not paired' method seems to work only
        for not too big rotation and when there is no noise.

    Raises
    ----------
    ValueError: Error raised if the specified method is unknown.

    Returns
    ----------
    X_source_aligned: ndarray, shape (n_samples_source, n_channels, n_channels)
        Source covariances after alignment.
    X_target_aligned: ndarray, shape (n_samples_target, n_channels, n_channels)
        Target covariances after alignment.

    References
    ----------
    [1] P. L. C. Rodrigues, C. Jutten and M. Congedo, "Riemannian Procrustes
        Analysis: Transfer Learning for Brain-Computer Interfaces," in IEEE
        Transactions on Biomedical Engineering, vol. 66, no. 8, pp. 2390-2401,
        Aug. 2019, doi: 10.1109/TBME.2018.2889705.
    [2] A. Bleuzé, J. Mattout and M. Congedo, "Transfer Learning for the
        Riemannian Tangent Space: Applications to Brain-Computer Interfaces,"
        2021 International Conference on Engineering and Emerging Technologies
        (ICEET), 2021, pp. 1-6, doi: 10.1109/ICEET53442.2021.9659607.
    [3] G. Maman, O. Yair, D. Eytan and R. Talmon, "Domain Adaptation Using
        Riemannian Geometry of SPD Matrices," ICASSP 2019 - 2019 IEEE
        International Conference on Acoustics, Speech and Signal Processing
        (ICASSP), 2019, pp. 4464-4468, doi: 10.1109/ICASSP.2019.8682989.
    """
    # n_source, n_channels, _ = X_source.shape
    n_target, _, _ = X_target.shape

    # First recenter and rescale
    X_source_str, X_target_str = align_recenter_rescale(X_source, X_target)

    # Then rotate around the mean
    ts_source = TangentSpace()
    Xt_source_str = ts_source.fit_transform(X_source_str)
    ts_target = TangentSpace()
    Xt_target_str = ts_target.fit_transform(X_target_str)

    if n_target < Xt_target_str.shape[1]:
        full_matrices = True
    else:
        full_matrices = False
    if method == 'paired':
        A = Xt_target_str.T @ Xt_source_str
        A += 1e-10 * np.eye(len(A))
        U, _, Vh = np.linalg.svd(A, full_matrices=full_matrices)

        Q_hat = (U @ Vh)

        X_target_aligned = ts_target.inverse_transform(Xt_target_str @ Q_hat)
        X_source_aligned = X_source_str

    elif method == 'not paired':
        # Source domain
        U_source, _, _ = np.linalg.svd(Xt_source_str.T,
                                       full_matrices=full_matrices)
        Xt_source_rot = Xt_source_str @ U_source

        # Target domain
        U_target, _, _ = np.linalg.svd(Xt_target_str.T,
                                       full_matrices=full_matrices)
        for j in range(U_target.shape[0]):
            U_target[:, j] = (
                np.sign(U_source[:, j] @ U_target[:, j]) * U_target[:, j]
            )

        Xt_target_rot = Xt_target_str @ U_target

        X_source_aligned = ts_source.inverse_transform(Xt_source_rot)
        X_target_aligned = ts_target.inverse_transform(Xt_target_rot)

    else:
        raise ValueError('Unknown method')

    return X_source_aligned, X_target_aligned
