from pyriemann.preprocessing import Whitening


def align_recenter(X_source, X_target):
    """Method to align source and target mean covariances to Identity.

    Parameters
    ----------
    X_source: ndarray, shape (n_samples_source, n_channels, n_channels)
        Covariances of the source domain.
    X_target: ndarray, shape (n_samples_target, n_channels, n_channels)
        Covariances of the target domain.

    Returns
    ----------
    X_source_aligned: ndarray, shape (n_samples_source, n_channels, n_channels)
        Source covariances after alignment.
    X_target_aligned: ndarray, shape (n_samples_source, n_channels, n_channels)
        Target covariances after alignment.
    """
    X_source_aligned = Whitening(metric='riemann').fit_transform(X_source)
    X_target_aligned = Whitening(metric='riemann').fit_transform(X_target)
    return X_source_aligned, X_target_aligned
