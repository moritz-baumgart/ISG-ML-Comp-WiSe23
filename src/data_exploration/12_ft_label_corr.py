import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from load import get_class_train_data


def do(remove_ft_name: str):
    X_df, y_df = get_class_train_data(split_label=True)
    print(f'Dropping {remove_ft_name}...')
    X_df.drop(columns=[remove_ft_name], inplace=True)

    isoF = IsolationForest()
    outlier_mask = isoF.fit_predict(X_df)

    X_df = X_df[outlier_mask == 1]
    y_df = y_df[outlier_mask == 1]
    y_df = y_df.reset_index(drop=True)

    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

    pca = PCA(n_components=3)
    pca_res = pd.DataFrame(pca.fit_transform(scaled_df))

    df = pd.merge(pca_res, y_df, left_index=True, right_index=True)

    g: sns.PairGrid = sns.pairplot(df, hue='label')
    g.figure.suptitle(f'{remove_ft_name} dropped')

    plt.savefig(f'drops/drop_{remove_ft_name}')


def main():

    for i in range(31):
        do(f'feature_{i}')


if __name__ == "__main__":
    main()
