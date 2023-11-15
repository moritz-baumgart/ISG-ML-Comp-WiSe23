import seaborn as sns
import numpy as np

from load import get_class_train_data


def main():
    df = get_class_train_data()
    df.drop(columns='Id', inplace=True)

    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(230, 40, as_cmap=True)

    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=0.5, annot=True, cbar_kws={"shrink": .5})

    fig = heatmap.get_figure()
    fig.set_layout_engine('tight')
    fig.set_size_inches((20, 20))
    fig.savefig('corr_matrix.pdf')


if __name__ == "__main__":
    main()
