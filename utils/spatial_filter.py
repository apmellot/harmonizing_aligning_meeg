import numpy as np
from scipy.linalg import eigh, pinv
from sklearn.base import BaseEstimator, TransformerMixin


class ProjCommonSpace(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1, n_compo='full', reg=1e-7):
        self.scale_ = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y=None):
        if self.scale_ == 'auto':
            self.scale_ = 1 / np.mean([np.trace(x) for x in X])
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        self.filters_ = []
        self.patterns_ = []
        C = X.mean(axis=0)
        eigvals, eigvecs = eigh(C)
        ix = np.argsort(np.abs(eigvals))[::-1]
        evecs = eigvecs[:, ix]
        evecs = evecs[:, :self.n_compo].T
        self.filters_.append(evecs)  # (fb, compo, chan) row vec
        self.patterns_.append(pinv(evecs).T)  # (fb, compo, chan)
        return self

    def transform(self, X):
        n_sub, _, _ = X.shape
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        Xout = np.empty((n_sub, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        filters = self.filters_[0]  # (compo, chan)
        for sub in range(n_sub):
            Xout[sub] = filters @ Xs[sub] @ filters.T
            Xout[sub] += self.reg * np.eye(self.n_compo)
        return Xout # (sub , compo, compo)
