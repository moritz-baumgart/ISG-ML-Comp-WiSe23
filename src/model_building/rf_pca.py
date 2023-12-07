import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import numpy as np

from load import get_class_train_data


class PCARF(BaseEstimator, ClassifierMixin):
    def __init__(self, pca_components=5, rf_estimator_num=100):
        self.pca_components = pca_components
        self.rf_estimator_num = rf_estimator_num

        self.pca = PCA(n_components=pca_components)
        self.rf_clf = RandomForestClassifier(n_estimators=rf_estimator_num)

    def set_params(self, **params):
        self.pca_components = params['pca_components']
        self.rf_estimator_num = params['rf_estimator_num']
        return self
    
    def score(self, X, y, sample_weight=None):
        return self.rf_clf.score(X, y, sample_weight)

    def fit(self, X, y):
        pca_res = self.pca.fit_transform(X, y)
        return self.rf_clf.fit(pca_res, y)

    #def predict(self, X):
     #   return self.rf_clf.predict(X)


def main():
    X_df, y_df = get_class_train_data(split_label=True)

    # Drop ft2, it's just ones
    X_df.drop(columns=["feature_2"], inplace=True)

    iso_f = IsolationForest()
    outlier_mask = iso_f.fit_predict(X_df)

    X_df = X_df[outlier_mask == 1]
    y_df = y_df[outlier_mask == 1]

    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

    model = PCARF()
    param_grid = {
        'pca_components': np.arange(2, 16),
        'rf_estimator_num': np.arange(50, 200, 20),
    }

    X_train, X_test, y_train, y_test = train_test_split(scaled_df, y_df, test_size=0.2)

    grid = GridSearchCV(model, param_grid=param_grid)
    grid.fit(X_train, y_train)

    pred = grid.predict(X_test)

    print(f1_score(y_test, pred))


    '''
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_df)

    X_train, X_test, y_train, y_test = train_test_split(pca_res, y_df, test_size=0.2)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    print(f1_score(y_test, pred))
    '''


if __name__ == "__main__":
    main()
