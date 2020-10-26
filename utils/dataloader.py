import pandas as pd
from sklearn import preprocessing
import numpy as np


class JunoLoader:
    def __init__(self, target_variable="Edep"):  # edepX, edepY, edepZ
        self._columns = ['R_lmpt', 'x_c_lpmt', 'y_c_lpmt', 'z_c_lpmt', 'ht_std', 'ht_mean', 'allHits_lpmt']
        self._target_variable = target_variable
        self._mean = [8., 0., 0., 0., 76., 122., 9705.]
        self._std = [3., 6., 6., 6., 10.5, 10., 4851.]

    def preprocess_data(self, path):
        df = pd.read_csv(path)
        df['R_c_lpmt'] = np.sqrt(
            df['x_c_lpmt'] ** 2 +
            df['y_c_lpmt'] ** 2
        )
        return df

    def fit_transform(self, path):
        df = self.preprocess_data(path)
        X = (df[self._columns].values - self._mean) / self._std
        y = df[self._target_variable].values
        return X, y

    def fit(self, path):
        pass
        return self

    def transform(self, path):
        df = self.preprocess_data(path)
        X = (df[self._columns].values - self._mean) / self._std
        y = df[self._target_variable].values
        return X, y
