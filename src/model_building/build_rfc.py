import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from preprocess.preprocess import i_forest


X, y = i_forest()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': np.linspace(100, 1000, 8).round().astype('int'),
    'max_features': ['sqrt', 'log2'],
}
gs = GridSearchCV(clf, param_grid)

gs.fit(X_train, y_train)

pred = gs.predict(X_test)

print(accuracy_score(y_test, pred))

filename = 'models/RFCv3.joblib'
dump(gs, filename)
print(f'Saved as {filename}')
