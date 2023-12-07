from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib


def main():

    X = pd.read_csv('../../data/regression/train_features.csv')
    X_test = pd.read_csv('../../data/regression/test_features.csv')
    y = pd.read_csv('../../data/regression/train_label.csv')

    clf = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    gs_params = {
        'n_neighbors': [2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 75]
    }

    '''
    gs = GridSearchCV(clf, gs_params, n_jobs=-1)

    gs.fit(X_train, y_train)

    joblib.dump(gs, 'gs_res.joblib')
    return

    print(gs.cv_results_)
    '''

    clf.fit(X_train, y_train)
    
    val_pred = clf.predict(X_val) 

    print(mean_squared_error(y_val, val_pred, squared=False))

    pred = clf.predict(X_test)

    pred_df = DataFrame(pred, columns=['Id', 'rating'])

    print(pred_df)
    print(pred_df['rating'].value_counts())



if __name__ == '__main__':
    main()