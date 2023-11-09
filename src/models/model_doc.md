# RFCv0.joblib:

```
train_size=0.8
random_state=42
param_grid = {
    'n_estimators': np.linspace(100, 1000, 8).round().astype('int'),
    'max_depth': np.linspace(1, 6, 6).round().astype('int'),
    'max_features': ['sqrt', 'log2'],
}

test_score = 0.7942122186495176
```


# RFCv1.joblib:

```
train_size=0.8
random_state=42
param_grid = {
    'n_estimators': np.linspace(100, 1000, 5).round().astype('int'),
    'max_features': ['sqrt', 'log2'],
}


test_score = 0.8135048231511254
```


# RFCv2.joblib:

```
train_size=0.667
random_state=42
param_grid = {
    'n_estimators': np.linspace(100, 1000, 5).round().astype('int'),
    'max_features': ['sqrt', 'log2'],
}


test_score = 0.8011583011583011
```

# RFCv3.joblib:

```

Preprocessing: IsolationForest(contamination=0.1)

train_size=0.8
random_state=42
param_grid = {
    'n_estimators': np.linspace(100, 1000, 8).round().astype('int'),
    'max_features': ['sqrt', 'log2'],
}


test_score = 0.8214285714285714
```



