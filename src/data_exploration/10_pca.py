import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from load import get_class_train_data


def main():
    X_df, y_df = get_class_train_data(split_label=True)

    # Drop ft2, it's just ones
    X_df.drop(columns=["feature_2"], inplace=True)

    isoF = IsolationForest()
    outlier_mask = isoF.fit_predict(X_df)

    X_df = X_df[outlier_mask == 1]
    y_df = y_df[outlier_mask == 1]

    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(scaled_df)

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, projection="3d")

    ax1.bar(np.arange(pca.n_components_), pca.explained_variance_ratio_)

    ax2.scatter(pca_res[:, 0], pca_res[:, 1], pca_res[:, 2], c=y_df)

    plt.show()


if __name__ == "__main__":
    main()
