from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

from preprocess import preprocess

"""
In this file I used the feature mask obtained using recursive feature elimination to train a random forest
and see which impact removing the features has on the score.
"""

def test_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, '../models/RFWithMask.joblib')

    pred = model.predict(X_test)

    print(f1_score(y_test, pred))

def main():
    X, y = preprocess.load_data()
    X, y = preprocess.i_forest(X, y)
    X, y = preprocess.normalize(X, y)

    mask = [False, True, False, True, False, False, False, False, True, True, False, True, False, False, False, False, True, True, False, False, False, False, False, False, True, False, False, False, False, True, False]

    X_removed = X[X.columns[mask]]

    test_model(X, y)
    test_model(X_removed, y)


if __name__ == '__main__':
    main()