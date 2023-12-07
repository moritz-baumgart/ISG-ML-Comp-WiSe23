import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from load import get_class_train_data
from util import make_diagram_foreach

####################
REMOVE_OUTLIERS=True
####################

mask = [False, True, False, True, False, False, False, False, True, False, False, True, False, False, False, False, True, True, False, True, False, False, False, False, True, False, False, False, False, True, False]

def main():
    df = get_class_train_data()
    df.drop(columns='Id', inplace=True)

    if REMOVE_OUTLIERS:
        df_no_outliers = df.copy()

        for column in df_no_outliers.columns:
            q1 = df_no_outliers[column].quantile(0.25)
            q3 = df_no_outliers[column].quantile(0.75)

            IQR = q3 - q1

            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR

            df_no_outliers[column] = df_no_outliers[column][(df_no_outliers[column] >= lower_bound) & (df_no_outliers[column] <= upper_bound)]

        df = df_no_outliers

    def make(index, ft_name, ft_values, ax: Axes):
        try:
            if mask[index]:
                color = 'red'
            else:
                color = 'blue'
        except:
            color = 'green'
        ax.hist(ft_values, bins=100, color=color)
        ax.set_title(ft_name)
    
    make_diagram_foreach(4, 8, df, 'hist.pdf', (60, 30), make)


if __name__ == "__main__":
    main()
