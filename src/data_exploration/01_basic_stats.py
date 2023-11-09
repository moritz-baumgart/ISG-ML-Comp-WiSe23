from load import get_class_train_data
import pandas as pd

def main():
    df = get_class_train_data()
    
    pd.options.display.max_columns = None

    print('Pandas describe:')
    print(df.describe())

    wait()

    print('Number of unique values/feature and their datatype:')
    uniqueValAndDtype = pd.concat([df.nunique(), df.dtypes], axis=1)
    uniqueValAndDtype.columns = ['#unique values', 'dtype']
    print(uniqueValAndDtype)

    wait()

    print('Number of duplicate rows/instances:')
    print(df.duplicated().sum())

    print('Number of duplicate columns/features:')
    print(df.T.duplicated().sum())

    wait()

    print('Number of missing/NaN values:')
    print(df.isna().sum())



def wait():
    input("\nPress enter to continue...\n")


if __name__ == '__main__':
    main()
