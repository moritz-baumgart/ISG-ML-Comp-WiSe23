from load import get_class_train_data, get_reg_train_data
import pandas as pd

"""
This file prints some basic statistics/values about the data (pandas describe, unique values and
datatype of columns, number of duplicate rows, number of NaNs).

It stops after each step and waits for you to press enter to continue.
"""


def main():
    class_df = get_class_train_data()
    print('####################\nClassification Task:\n####################')
    do_basic_stats(class_df)

    reg_df = get_reg_train_data()
    print('################\nRegression Task:\n################')
    do_basic_stats(reg_df)
    

def do_basic_stats(df: pd.DataFrame):
    pd.options.display.max_columns = None

    print('Pandas describe:')
    print(df.describe())

    wait()

    print('Number of unique values/feature and their datatype:')
    uniqueValAndDtype = pd.concat([df.nunique(), df.dtypes], axis=1)
    uniqueValAndDtype.columns = ['#unique values', 'dtype']
    print(uniqueValAndDtype)

    wait()

    # Remove the id for duplicate check, since it is unique in every row
    df_no_id = df.drop(columns='Id')
    print('Number of duplicate rows/instances:')
    print(df_no_id.duplicated().sum())

    # I removed the following for now, since I dont know how much sense it makes
    #print('Number of duplicate columns/features:')
    #print(df.T.duplicated().sum())

    wait()

    print('Number of missing/NaN values:')
    print(df.isna().sum())

    wait()


def wait():
    input("\nPress enter to continue...\n")


if __name__ == '__main__':
    main()
