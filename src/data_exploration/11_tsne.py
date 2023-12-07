from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

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

    tsne = TSNE(random_state=42)
    tsne_res = tsne.fit_transform(scaled_df)

    plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=y_df)
    plt.show()


if __name__ == "__main__":
    main()
