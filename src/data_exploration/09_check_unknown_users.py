from load import get_all_reg_ft_data

from pandas import Index


"""
This file prints the number of unique users in the train and test set of the regression task, as well as the intersection of both.
This way we can see if we have any users in the test set that were not in the train set (Spoiler: There are none).
"""


def main():
    df_train, df_test = get_all_reg_ft_data(train_test_split=True)

    train_users = Index(df_train['user'])
    test_users = Index(df_test['user'])

    train_users = train_users.drop_duplicates()
    test_users = test_users.drop_duplicates()

    inter = train_users.intersection(test_users)

    print(len(train_users))
    print(len(test_users))
    print(len(inter))

if __name__ == '__main__':
    main()