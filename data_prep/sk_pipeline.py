from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


class Cut_last_sample(BaseEstimator, TransformerMixin):
    def __init__(self, samples_per_participant):
        """number of samples per participant."""
        self.samples_per_participant = samples_per_participant

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        """Remove last sample per participant."""
        samples_per_participant = self.samples_per_participant
        mask_cut_last_sample = np.ones((X.shape[0]), dtype=bool)
        mask_cut_last_sample[samples_per_participant
                             - 1::samples_per_participant] = False
        X = X.loc[mask_cut_last_sample, :]
        return X


class AddCategoricalLabels(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.est = KBinsDiscretizer(**kwargs)

    def fit(self, X, y=None):
        return self.est.fit(X)

    def transform(self, X, y=None):
        X_cat = self.est.transform(X)
        # print(X_cat)
        return np.concatenate([X, X_cat], axis=1)
