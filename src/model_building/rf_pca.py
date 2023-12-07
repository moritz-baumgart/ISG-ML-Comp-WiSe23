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

    X_train, X_test, y_train, y_test = train_test_split(scaled_df, y_df, test_size=0.2)


    scores = {}

    for pca_comp_n in range(2, 16):
        for rf_esimator_n in range(50, 200, 20):

            pca = PCA(n_components=pca_comp_n)
            pca_res = pca.fit_transform(scaled_df)

            X_train, X_test, y_train, y_test = train_test_split(pca_res, y_df, test_size=0.2)

            clf = RandomForestClassifier(random_state=42, n_estimators=rf_esimator_n)
            clf.fit(X_train, y_train)

            pred = clf.predict(X_test)

            score = f1_score(y_test, pred)

            print(score)
            scores[(pca_comp_n, rf_esimator_n)] = score


if __name__ == "__main__":
    main()
