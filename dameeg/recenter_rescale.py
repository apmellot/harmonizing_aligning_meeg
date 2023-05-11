import numpy as np

from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.tangentspace import untangent_space


def align_recenter_rescale(X_source, X_target, return_in_ts=False):
    """Method to align source and target data by recentering and rescaling them.

    The input are M/EEG covariances. First, covariances from both domains are
    re-centered to identity by whitening them with the mean covariance of each
    domain and then projected to the tangent space at identity. The dispersion
    of the target domain is then adjusted to be equal to the dispersion of the
    source domain.

    Parameters
    ----------
    X_source: ndarray, shape (n_samples_source, n_channels, n_channels)
        Covariances of the source domain.
    X_target: ndarray, shape (n_samples_target, n_channels, n_channels)
        Covariances of the target domain.
    return_in_ts: boolean
        If True, returns points belonging to the tangent space at identity.
        Otherwise, returns covariance matrices.

    Returns
    ----------
    X_source_aligned: ndarray, shape (n_samples_source, n_channels, n_channels)
        Source covariances after alignment.
    X_target_aligned: ndarray, shape (n_samples_target, n_channels, n_channels)
        Target covariances after alignment.
    """
    ts = TangentSpace()

    Xt_source = ts.fit_transform(X_source)
    Xt_source /= np.sqrt(np.linalg.norm(Xt_source)**2 / Xt_source.shape[0])

    Xt_target = ts.fit_transform(X_target)
    Xt_target /= np.sqrt(np.linalg.norm(Xt_target)**2 / Xt_target.shape[0])

    if return_in_ts:
        return Xt_source, Xt_target
    else:
        eye = np.eye(X_source.shape[-1])
        return untangent_space(Xt_source, eye), untangent_space(Xt_target, eye)
