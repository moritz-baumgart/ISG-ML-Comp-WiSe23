import joblib
from sklearn.model_selection import GridSearchCV


def main():
    gs: GridSearchCV = joblib.load('gs_res.joblib')
    print(gs.best_estimator_)

if __name__ == '__main__':
    main()