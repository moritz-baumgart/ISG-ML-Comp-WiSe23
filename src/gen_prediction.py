import pandas as pd
from joblib import load


df = pd.read_csv('data/test_features.csv')

ids = df['Id']
X = df.drop(columns='Id')

clf = load('models/NNv0.joblib')

pred = clf.predict(X)

res = pd.DataFrame({'Id': ids, 'label': pred})

res.to_csv('predictions.csv', index=False)
