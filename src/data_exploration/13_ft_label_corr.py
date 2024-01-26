import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from load import get_class_train_data


def main():
    df = get_class_train_data()
    df.drop(columns=['Id', 'feature_2'], inplace=True)

    g: sns.PairGrid = sns.pairplot(df, hue='label')
    g.figure.suptitle(f'some suptitle')

    plt.savefig(f'test')


if __name__ == "__main__":
    main()
