## LightFM

In this directory I played around a bit with LightFM and tried to obtain a first few prediction for the regression task.

Inside `lfm.py` I trained the model itself and saved it using joblib. Afterwards I then was able to load it inside `predict.py` and generate predictions. 

I also played around a bit with the ideas I had for the regression task (like dropping instances based on the timestamp).