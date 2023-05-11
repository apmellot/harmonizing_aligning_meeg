import numpy as np


def align_z_score(X_source, X_target, fisher_transform=False, global_scaling=False):
    """Method to transform covariance matrices into correlation matrices.

    Parameters
    ----------
    X_source: ndarray, shape (n_samples_source, n_channels, n_channels)
        Covariances of the source domain.
    X_target: ndarray, shape (n_samples_target, n_channels, n_channels)
        Covariances of the target domain.
    fisher_transform: bool
        If true, inverse hyperbolic tangent element-wise is applied to
        correlation matrices.
    global_scaling: bool
        Whether to apply global or individual correction.

    Returns
    ----------
    X_source_aligned: ndarray, shape (n_samples_source, n_channels, n_channels)
        Source covariances after alignment.
    X_target_aligned: ndarray, shape (n_samples_source, n_channels, n_channels)
        Target covariances after alignment.
    """

    n_dim_source = X_source.shape[0]
    n_dim_target = X_target.shape[0]
    if global_scaling:
        D_source = np.sqrt(np.array(
            [np.diag(X_source[i]) for i in range(n_dim_source)]).max(axis=0))
        D_target = np.sqrt(np.array(
            [np.diag(X_target[i]) for i in range(n_dim_target)]).max(axis=0))
        outer_D_source = np.outer(D_source, D_source)
        outer_D_target = np.outer(D_target, D_target)
    else:
        D_source = [np.sqrt(np.diag(X_source[i])) for i in range(n_dim_source)]
        D_target = [np.sqrt(np.diag(X_target[i])) for i in range(n_dim_target)]
        outer_D_source = [np.outer(D_source[i], D_source[i]) for i in range(n_dim_source)]
        outer_D_target = [np.outer(D_target[i], D_target[i]) for i in range(n_dim_target)]
    X_source_aligned = X_source / outer_D_source
    X_target_aligned = X_target / outer_D_target
    if fisher_transform:
        X_source_aligned = np.arctanh(X_source_aligned)
        X_target_aligned = np.arctanh(X_target_aligned)
    return X_source_aligned, X_target_aligned
