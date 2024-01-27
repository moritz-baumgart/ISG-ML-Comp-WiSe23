### Ensembles

In this folder I tried out the different algorithms provided by `scikit-learn` inside `sklearn.ensemble`, since we already found out that ensembles seem to perform relatively good.

Inside `ensembles_from_sk.ipynb` I trained and evaluated the different algorithms from the package. I tried different preprocessing techniques as well. You can see the results near the end of that file.

I have also tried to use permutation importance to find features to drop to increase performance. This was done inside `perm_imp.ipynb`, but I also combined the results from that into `ensembles_from_sk.ipynb` 