import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale

from load import get_class_train_data

def main():

    # load iris dataset
    df = get_class_train_data()
    df.drop(columns=['Id', 'feature_2'], inplace=True)

    X = df.drop(columns=['label'])
    cols = X.columns

    # apply PCA
    pca = decomposition.PCA(n_components=2)
    X = pca.fit_transform(X)

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=cols)
    
    print(loadings)

if __name__ == '__main__':
    main()