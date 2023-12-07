import pandas as pd
import numpy as np
import joblib
from lightfm import LightFM
from lightfm.data import Dataset

def main():
    df = pd.read_csv('../../data/regression/test_features.csv')

    dataset: Dataset = joblib.load('ds_lfm-user_noskip_gt18_nosplit.joblib')
    user_map = dataset.mapping()[0]
    item_map = dataset.mapping()[2]

    lfm: LightFM = joblib.load('lfm-user_noskip_gt18_nosplit.joblib')

    pred = np.array([])

    # TODO: Optimize this shit:
    for index, instance in df.iterrows():
        userid = instance['user']
        itemid = instance['item']

        if userid in user_map.keys() and itemid in item_map.keys():
            pred = np.append(pred, lfm.predict([user_map[userid]], [item_map[itemid]]))
        else:
            pred = np.append(pred, 4)

    pred_rounded = np.round(pred)
    pred_df = pd.DataFrame(pred_rounded, columns=['Predicted'])

    print(pred_df.value_counts())

    pred_df.to_csv('pred.csv', index_label='Id')

    

    


if __name__ == '__main__':
    main()