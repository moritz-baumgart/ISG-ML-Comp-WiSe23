import pandas as pd
from pandas import DataFrame
import pathlib


def load_X_y(trainft, trainlabel):

    train_ft = pd.read_csv(trainft)
    train_label = pd.read_csv(trainlabel)

    df = pd.merge(train_ft, train_label, on='Id', how='inner')

    X = df.drop(columns=['Id', 'label'])
    y = df['label']

    return X, y

def save_df(df: DataFrame, filename: str):
    df.to_csv(filename, index=False)