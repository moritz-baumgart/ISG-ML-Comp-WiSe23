from load import get_class_train_data
from util import make_diagram_foreach


"""
This file creates and shows a diagram with the class distributions of the categorical features/the label of the classification task.
"""


def main():
    df = get_class_train_data()

    df = df[['feature_10', 'feature_12', 'feature_20', 'label']]

    print(df['label'].value_counts())

    def make(index, ft_name, ft_values, ax):
        categories = ft_values.value_counts()
        ax.bar(categories.index.map(lambda x: str(x)), categories.values)
        ax.set_title(ft_name)

    make_diagram_foreach(1, 4, df, None, None, make)


if __name__ == '__main__':
    main()