import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
from lightfm import LightFM
from lightfm.data import Dataset
import joblib
from pprint import pprint


def main():

    X = pd.read_csv('../../data/regression/train_features.csv')
    y = pd.read_csv('../../data/regression/train_label.csv')

    df = pd.merge(X, y)
    df = df.loc[df['timestamp'] > 1514761200000]

    # When testing you can comment in the following line to only work with 10 instances
    #df = df.iloc[:10]

    df = df[['user', 'item', 'rating']]
    df = df.drop_duplicates()

    dataset = Dataset()
    dataset.fit(df['user'], df['item'])

    #joblib.dump(dataset, 'ds_lfm-user_noskip_gt18_nosplit.joblib')

    user_item_tuples = [(row['user'], row['item']) for index, row in df.iterrows()]
    (interactions, weights) = dataset.build_interactions(user_item_tuples)

    # REMOVE COMMENT TO TRAIN TEST SPLIT
    train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=42)

    scores = {}

    for n_comp in range(1, 31):
        lfm = LightFM(no_components=n_comp, learning_schedule='adadelta')
        lfm.fit(train, num_threads=8)

        #joblib.dump(lfm, 'lfm-user_noskip_gt18_nosplit.joblib')
        score = auc_score(lfm, test).mean()
        print(score)
        scores[n_comp] = score
    
    joblib.dump(scores, 'adadelta_1-30_score.joblib')


if __name__ == '__main__':
    main()