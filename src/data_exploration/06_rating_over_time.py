import matplotlib.pyplot as plt
import pandas as pd

from load import get_reg_train_data


"""
This file creates and shows the rating of the regression task over time. It bins them in 5 day large bins and takes the mean rating for each bin.
"""


def main():
    df = get_reg_train_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df.set_index("timestamp", inplace=True)

    df.sort_index(inplace=True)

    df_binned = df.resample(pd.Timedelta(days=5)).mean()

    plt.plot(df_binned.index, df_binned["rating"])

    plt.title("Mean rating over time (binned in 5 day large bins)")
    plt.xlabel("Time")
    plt.ylabel("Rating")
    plt.show()


if __name__ == "__main__":
    main()
