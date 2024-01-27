import pandas as pd

"""
This file is used for loading the data from the CSV files.
"""


def get_train_data(ft_file: str, label_file: str, label_name: str, split_label=False):
    train_ft = pd.read_csv(ft_file)
    train_label = pd.read_csv(label_file)

    df = pd.merge(train_ft, train_label, on='Id', how='inner')

    if split_label:
        X = df.drop(columns=['Id', label_name])
        y = df[label_name]

        return X, y
    
    return df

def get_class_train_data(split_label=False):
    return get_train_data('../data/classification/train_features.csv', '../data/classification/train_label.csv', 'label', split_label)

def get_reg_train_data(split_label=False):
    return get_train_data('../data/regression/train_features.csv', '../data/regression/train_label.csv', 'rating', split_label)


def get_all_ft_data(train_ft_file: str, test_ft_file: str, train_test_split=False):
    train_ft = pd.read_csv(train_ft_file)
    test_ft = pd.read_csv(test_ft_file)

    if train_test_split:
        return train_ft, test_ft
    else:
        return pd.concat([train_ft, test_ft])

def get_all_class_ft_data(train_test_split=False):
    return get_all_ft_data('../data/classification/train_features.csv', '../data/classification/test_features.csv', train_test_split)

def get_all_reg_ft_data(train_test_split=False):
    return get_all_ft_data('../data/regression/train_features.csv', '../data/regression/test_features.csv', train_test_split)
