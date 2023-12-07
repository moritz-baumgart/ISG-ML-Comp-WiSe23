import pandas as pd
from joblib import load

from model_building.preprocess import preprocess

df = pd.read_csv('data/classification/test_features.csv')

ids = df['Id']
X = df.drop(columns='Id')
X.index = ids

mask = [False, True, False, True, False, False, False, False, True, True, False, True, False, False, False, False, True, True, False, False, False, False, False, False, True, False, False, False, False, True, False]

X = X[X.columns[mask]]


X, y = preprocess.normalize(X, None)

clf = load('models/RFWithMask.joblib')

pred = clf.predict(X)

res = pd.DataFrame({'Id': X.index, 'label': pred})

res.to_csv('predictions.csv', index=False)
