from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFECV

from preprocess import preprocess




def main():

    X, y = preprocess.load_data()
    X, y = preprocess.i_forest(X, y)
    X, y = preprocess.normalize(X, y)

    rfclf = RandomForestClassifier()

    selector = RFECV(rfclf, scoring='f1_macro')
    selector = selector.fit(X, y)

    print(selector.support_)
    # default scoring: [False, True, False, True, False, False, False, False, True, True, False, True, False, False, False, False, True, True, False, False, False, False, False, False, True, False, False, False, False, True, False]

   # f1-macro: [False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, True, False, True, False, False, False, False, True, False, False, False, False, True, False]

   # ---> The two ones swapped look quite identidal in the histogram

    print(selector.ranking_)
    # [13  1 23  1  7 16 12 10  1  1 21  1 20 17  8 14  1  1  6  2 22 15 18  9 1  4  5 11 19  1  3]



if __name__ == '__main__':
    main()


