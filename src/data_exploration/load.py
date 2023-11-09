import pandas as pd


def get_train_data(ft_file='../data/train_features.csv', label_file='../data/train_label.csv', split_label=False):
    train_ft = pd.read_csv(ft_file)
    train_label = pd.read_csv(label_file)

    df = pd.merge(train_ft, train_label, on='Id', how='inner')

    if split_label:

        X = df.drop(columns=['Id', 'label'])
        y = df['label']

        return X, y
    else:
        return df
