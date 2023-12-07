import joblib
from typing import Dict
import matplotlib.pyplot as plt

adagrad: Dict[int, float] = joblib.load('adagrad_1-30_score.joblib')
adadelta: Dict[int, float] = joblib.load('adadelta_1-30_score.joblib')

#s = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
#print(s)

plt.plot(adagrad.keys(), adagrad.values(), label='adagrad (auc)')
plt.plot(adadelta.keys(), adadelta.values(), label='adadelta (auc)', color='red')
plt.legend()
plt.show()