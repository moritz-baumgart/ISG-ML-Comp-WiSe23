import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def main():
    X = pd.read_csv('test_features.csv')
    y = pd.read_csv('train_label.csv')

    # Id is already inside the dataframes index, ft2 is just ones
    X.drop(columns=['Id', 'feature_2'], inplace=True)
    y.drop(columns='Id', inplace=True)

    iso_f = IsolationForest()
    outlier_mask = iso_f.fit_predict(X)

    #X = X[outlier_mask == 1]
    #y = y[outlier_mask == 1]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    #df = pd.merge(X, y, left_index=True, right_index=True)

    X.to_csv('test_ft_iso_f_std_scale.csv', index=False)

if __name__ == '__main__':
    main()