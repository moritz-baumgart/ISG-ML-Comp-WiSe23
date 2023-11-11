import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from load import get_class_train_data


def main():
    df = get_class_train_data()

    df = df[['feature_10', 'feature_12', 'feature_20', 'label']]

    fig, axs = plt.subplots(1, 4)

    for index, (ft_name, ft_values) in enumerate(df.items()):
        ax: Axes = axs[index]
        categories = ft_values.value_counts()
        print(categories)
        ax.bar(categories.index.map(lambda x: str(x)), categories.values)
        ax.set_title(ft_name)

    plt.show()



if __name__ == '__main__':
    main()