import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.ensemble import IsolationForest
from joblib import dump
import matplotlib.pyplot as plt


train_ft = pd.read_csv('./train_features.csv')
train_label = pd.read_csv('./train_label.csv')

df = pd.merge(train_ft, train_label, on='Id', how='inner')

X = df.drop(columns=['Id', 'label'])
y = df['label']

iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X)
mask = yhat != -1
X, y = X[mask], y[mask]

X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

train_err = []
test_err = []

'''
for k in range(1, 100):

    clf = KNeighborsClassifier(k)

    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    train_err.append(1 - train_acc)
    test_err.append(1 - test_acc)

    print(f'{k} : {train_acc}, {test_acc}')
'''

plt.plot(train_err, label="train")
plt.plot(test_err, label="test")

plt.legend()
plt.show()

clf = KNeighborsClassifier(27)

clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))

filename = 'KNNv0.joblib'
dump(clf, filename)
print(f'Saved as {filename}')
