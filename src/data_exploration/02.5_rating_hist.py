import matplotlib.pyplot as plt

from load import get_reg_train_data


"""
This file creates and shows a diagram with the distribution of rating in the regression task.
"""


def main():
    df = get_reg_train_data()

    categories = df['rating'].value_counts().sort_index()
    plt.bar(categories.index.map(lambda x: str(x)), categories.values)

    plt.title('Rating distribution')

    plt.show()


if __name__ == '__main__':
    main()