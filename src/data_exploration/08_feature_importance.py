from load import get_class_train_data
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def main():
    X, y = get_class_train_data(split_label=True)

    model = DecisionTreeClassifier()

    model.fit(X, y)

    plt.bar(X.columns, model.feature_importances_)
    plt.xticks(rotation=90)
    plt.show()          




if __name__ == '__main__':
    main()