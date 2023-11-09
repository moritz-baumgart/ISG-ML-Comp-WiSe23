import pandas as pd


def get_train_data(ft_file: str, label_file: str, split_label=False):
    train_ft = pd.read_csv(ft_file)
    train_label = pd.read_csv(label_file)

    df = pd.merge(train_ft, train_label, on='Id', how='inner')

    if split_label:

        X = df.drop(columns=['Id', 'label'])
        y = df['label']

        return X, y
    else:
        return df

def get_class_train_data(split_label=False):
    return get_train_data('../data/classification/train_features.csv', '../data/classification/train_label.csv', split_label)

def get_reg_train_data(split_label=False):
    return get_train_data('../data/classification/train_features.csv', '../data/classification/train_label.csv', split_label)

