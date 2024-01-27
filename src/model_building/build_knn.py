import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


"""
This file load the classification dataset, drops ft2, removes outliers using IsolationForest and scales the data using StandardScaler.
It does a train test split, find the best k for kNN using grid search, plots the results for the different Ks and prints the test end result.
"""


train_ft = pd.read_csv('../data/classification/train_features.csv')
train_label = pd.read_csv('../data/classification/train_label.csv')

df = pd.merge(train_ft, train_label, on='Id', how='inner')

X = df.drop(columns=['Id', 'label', 'feature_2'])
y = df['label']

iso = IsolationForest()
yhat = iso.fit_predict(X)
mask = yhat != -1
X, y = X[mask], y[mask]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


clf = KNeighborsClassifier()

params = {
    'n_neighbors': np.arange(1, 15)
}

grid = GridSearchCV(clf, param_grid=params)

grid.fit(X_train, y_train)

pred = grid.predict(X_test)


plt.plot(params['n_neighbors'], grid.cv_results_['mean_test_score'], label="mean cv score (f1)")

print('Final test score:')
print(f1_score(y_test, pred))

plt.legend()
plt.show()
