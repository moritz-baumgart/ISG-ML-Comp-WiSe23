from .fileop import load_X_y
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import scale
from pandas import DataFrame

"""
This file contains some basic preprocessing (outlier removal using isolation forest and normalization).
Besides that I used it for loading the classification data.
"""


def load_data():
    return load_X_y('../data/classification/train_features.csv', '../data/classification/train_label.csv')


def i_forest(X: DataFrame, y: DataFrame):

    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    mask = yhat != -1
    if y is not None:
        X, y = X[mask], y[mask]

        return X, y
    return X[mask]


def normalize(X: DataFrame, y: DataFrame):
    scaled_X = scale(X)
    df = DataFrame(scaled_X)
    df.columns = X.columns
    return df, y
