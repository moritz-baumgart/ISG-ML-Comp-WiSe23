import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

from load import get_class_train_data

"""
In this file I used grid search and tried to increase the performance of a random forest.
"""


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

    params = {
        "bootstrap": [True, False],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        "max_features": ["auto", "sqrt"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    }

    X_train, X_test, y_train, y_test = train_test_split(scaled_df, y_df, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=1000, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', max_depth=40, bootstrap=False)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    print(f1_score(y_test, pred))


    '''
    random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=40)

    random_search.fit(scaled_df, y_df)

    print(random_search.best_estimator_)
    print(random_search.best_params_) # {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 40, 'bootstrap': False}
    '''


if __name__ == "__main__":
    main()
