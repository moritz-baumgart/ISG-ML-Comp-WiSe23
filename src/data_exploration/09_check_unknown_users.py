from load import get_all_reg_ft_data

from pandas import Index

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