import pandas as pd
import os

d = os.path.dirname(__file__)


# test

def load_regression_train():
    X = pd.read_csv(d + '/reg/train_features.csv')
    y = pd.read_csv(d + '/reg/train_label.csv')
    df = pd.merge(X, y)
    
    # Drop id column, since it is already in the index of the df
    return df.drop(columns=['Id'])

def load_regression_test():
    X = pd.read_csv(d + '/reg/test_features.csv')
    
    # Drop id column, since it is already in the index of the df
    return X.drop(columns=['Id'])