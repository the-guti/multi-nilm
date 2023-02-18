import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class MySignal2Vec(BaseEstimator, TransformerMixin):

    def __init__(
        self, word_size: int = 4, n_bins: int = 4, window_size: int = 10, window_step: int = 1,
    ) -> None:
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        print('Data')
        print(X.shape)

        raise Exception('MySignal2Vec doesn\'t support fit yet.')

    def transform(self, X) -> np.ndarray:

        raise Exception('MySignal2Vec doesn\'t support transform yet.')

    def fit_transform(self, X, y=None, **fit_params):

        raise Exception('MySignal2Vec doesn\'t support fit_transform yet.')