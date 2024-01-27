from load import get_class_train_data
from util import make_diagram_foreach


"""
This file creates boxplots for all features/the label of the classification tasks and saves them in "boxplot.pdf".
"""

def main():
    df = get_class_train_data()
    df.drop(columns='Id', inplace=True)

    def make(index, ft_name, ft_values, ax):
        ax.boxplot(ft_values)
        ax.set_title(ft_name)

    make_diagram_foreach(4, 8, df, 'boxplot.pdf', (60, 30), make)


if __name__ == '__main__':
    main()