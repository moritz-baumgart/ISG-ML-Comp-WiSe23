import keras
from keras import layers
from keras.optimizers import Adam
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.src.callbacks import History
from joblib import dump

"""
In this file I played around a bit with keras/tensorflow and tried to build and evaluate a simple neural network
on the classification dataset and see how it behaves with different parameters.
"""


### PARAMS
random_state=42
train_size=0.8
activation='relu'
learning_rate=0.0001
###

X, y = preprocess.load_data()
X, y = preprocess.i_forest(X, y)
X, y = preprocess.normalize(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)


input_shape = X_train.shape[1]

print(input_shape)

model = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

opt = Adam(learning_rate=learning_rate)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history: History = model.fit(X_train, y_train, epochs=128, validation_data=(X_test, y_test), verbose=1)

pred = history.model.predict(X_test).round()

print(f'Acc: {accuracy_score(y_test, pred)}')
print(f'F1: {f1_score(y_test, pred)}')

dump(pred, 'test.ndarr')
dump(history, 'models/NNv2.joblib')


