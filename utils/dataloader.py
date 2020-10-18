import pandas as pd
from sklearn import preprocessing
import numpy as np


class JunoLoader:
    def __init__(self):
        self._normalizer = None
        self._columns = ['R_c_lpmt', 'z_c_lpmt', 'std', 'mean', 'allHits_lpmt']

    def preprocess_data(self, path):
        df = pd.read_csv(path)
        df['R_c_lpmt'] = np.sqrt(
            df['x_c_lpmt'] ** 2 +
            df['y_c_lpmt'] ** 2
        )
        return df

    def fit_transform(self, path):
        df = self.preprocess_data(path)
        self._normalizer = preprocessing.Normalizer()
        X = self._normalizer.fit_transform(
            df[self._columns]
        )
        y = df['Edep'].values
        return X, y

    def fit(self, path):
        df = self.preprocess_data(path)
        self._normalizer = preprocessing.Normalizer()
        self._normalizer.fit(
            df[self._columns]
        )
        return self

    def transform(self, path):
        if self._normalizer is None:
            raise ValueError("JunoLoader is not fitted!")
        df = self.preprocess_data(path)
        X = self._normalizer.transform(df[self._columns])
        y = df['Edep'].values
        return X, y