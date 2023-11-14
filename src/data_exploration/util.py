from typing import Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

def make_diagram_foreach(nrows: int, ncols: int, df: pd.DataFrame, file_name: str, size_inches: Optional[Tuple[int, int]], callback):
    fig, axs = plt.subplots(nrows, ncols)

    axs = axs.flatten()

    for index, (ft_name, ft_values) in enumerate(df.items()):
        callback(index, ft_name, ft_values, axs[index])

    fig.set_layout_engine('tight')
    if file_name is None or size_inches is None:
        plt.show()
    else:
        fig.set_size_inches(size_inches)
        plt.savefig(file_name)
