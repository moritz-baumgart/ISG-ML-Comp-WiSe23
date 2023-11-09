from .fileop import load_X_y
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import scale
from pandas import DataFrame


def load_data():
    return load_X_y('data/train_features.csv', 'data/train_label.csv')


def i_forest(X: DataFrame, y: DataFrame):

    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    mask = yhat != -1
    X, y = X[mask], y[mask]

    return X, y


def normalize(X: DataFrame, y: DataFrame):
    scaled_X = scale(X)
    return scaled_X, y
