from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from pandas import DataFrame
import joblib

def main():
    X = pd.read_csv('../../data/regression/train_features.csv')
    y = pd.read_csv('../../data/regression/train_label.csv')

    df = pd.merge(X, y)
    df = df.loc[df['timestamp'] > 1514761200000]

    print(df)



    df = df[['user', 'item', 'rating']]
    df = df.drop_duplicates()

    df = df.pivot(index='user', columns='item', values='rating')

    df = df.fillna(0)

    joblib.dump(df, 'pivot.joblib')

    nn = NearestNeighbors()
    nn.fit(df)
    distances, indices = nn.kneighbors(df, n_neighbors=5)

    distances_df = DataFrame(distances)
    distances_df.index = df.index

    indices_df = DataFrame(indices)
    indices_df.index = df.index

    joblib.dump(distances_df, 'distances.joblib')
    joblib.dump(indices_df, 'indices.joblib')

if __name__ == '__main__':
    main()