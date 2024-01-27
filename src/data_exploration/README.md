## Data Exploration

This folder contains all the code I wrote during the data exploration milestone. I wrote this code before a created [the library](https://github.com/moritz-baumgart/ISG-ML-Comp-WiSe23-Lib). This is why the code uses its own `load.py` for loading data, which in turn expects the data to be in a relative path to your CWD. This means that your CWD has to be the directory you also find this README inside, otherwise it won't find the data.

You can look around the individual files; Each one contains a short description at the top of what it does.

### Dependencies
Most files only need `matplotlib` and `pandas` (which the former also depends on). For some files you will need `scikit-learn`.