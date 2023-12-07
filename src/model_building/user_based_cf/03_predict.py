import joblib
import pandas as pd
from pandas import DataFrame
import numpy as np

###############
N_NEIGHBORS = 5
###############

n_fb1 = 0
n_fb2 = 0
n_calc = 0

def predict(itemid: int, userid: int, pivot: DataFrame, indices: DataFrame, distances: DataFrame):
    
    global n_fb1
    global n_fb2
    global n_calc

    if itemid not in pivot:
        '''
        If the requested item does not exist in the dataset, we just return
        5, because this is the most common rating over all
        '''
        n_fb1 += 1
        return 5

    if userid not in indices.index:
        '''
        If the user is unknown (i.e. not in our training data),
        we just return the average rating for that item
        (it is given that this exists bc of the check above)
        '''
        n_fb2 += 1

        item_ratings = pivot[itemid]
        return np.round(item_ratings.mean())
        
    n_calc += 1
    
    # otherwise calculate a rating:
    sim_user_indices = indices.loc[userid]
    sim_user = pivot.iloc[sim_user_indices]
    sim_user_ratings = sim_user[itemid]
    sim_user_distances = distances.loc[userid]

    '''
    R(i, u) = rating for item i by user u
    S(u, w) = similarity between user u und w
    We calculate:
    R(i, u) = (SUM_k S(u, k) * R(i, k)) / SUM_k S(u, k) for all k element kNN of u
    '''

    return np.round(sum(sim_user_distances.values * sim_user_ratings.values) / sum(sim_user_distances.values))


def main():
    indices = joblib.load('indices.joblib')
    distances = joblib.load('distances.joblib')
    pivot: DataFrame = joblib.load('pivot.joblib')

    test_data = pd.read_csv('../../data/regression/test_features.csv')

    predictions_colums = ['Id', 'rating']
    predictions = DataFrame(columns=predictions_colums)

    for index, test_instance in test_data.iterrows():
        itemid = test_instance['item']
        userid = test_instance['user']

        # (id is a reserved keyword)
        the_id = test_instance['Id']
        pred = predict(itemid, userid, pivot, indices, distances)

        pred_df = DataFrame([{'Id': the_id, 'rating': pred}])

        predictions = pd.concat([predictions, pred_df], ignore_index=True)

    print(predictions['rating'].value_counts())

    print(n_fb1)
    print(n_fb2)
    print(n_calc)

    predictions.to_csv('prediction.csv', index=None)


if __name__ == '__main__':
    main()