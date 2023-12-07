from surprise import KNNWithMeans, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

import pandas as pd

def main():
    X = pd.read_csv('../data/regression/train_features.csv')
    y = pd.read_csv('../data/regression/train_label.csv')

    df = pd.merge(X, y)

    df = df.loc[df['timestamp'] > 1514761200000]
    df = df.iloc[::15]

    reader = Reader(rating_scale=(1, 5))

    ds = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

    sim_options = {
        'name': 'cosine'
    }

    algo = KNNWithMeans(sim_options)

    trainset, testset = train_test_split(ds, test_size=0.2)

    print(type(trainset))
    print(type(test))
    

    algo.fit(trainset)

    prediction = algo.test(testset)

    print(prediction)
    return

    print(rmse(prediction))






if __name__ == '__main__':
    main()