import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from load import get_class_train_data


def main():
    df = get_class_train_data()
    df.drop(columns='Id', inplace=True)

    fig, axs = plt.subplots(4, 8)

    axs = axs.flatten()

    for index, (ft_name, ft_values) in enumerate(df.items()):
        ax: Axes = axs[index]
        ax.boxplot(ft_values)
        ax.set_title(ft_name)

    fig.set_layout_engine('tight')
    fig.set_size_inches((60, 30))
    plt.savefig('boxplot.pdf')


if __name__ == '__main__':
    main()