import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from load import get_reg_train_data


"""
This file creates and shows the average rating and the number of rating received for each item in the regression task.
It plots 3 variants of these graphs: unsorted (or sorted by id), sorted by average rating, sorted by number of ratings.
This showcases the long tail distribution that we would typically see.
Warning: This file takes some time to run because of the amount of data it has to aggregate. Just give it a bit of time. 
"""

def make_plot(aggregated_ratings, hide_xticks=False):
    fig, axes = plt.subplots(2, 1)

    ax0: Axes = axes[0]
    ax0.set_xlabel("Item")
    ax0.set_ylabel("Average rating (linear)")
    ax0.bar(aggregated_ratings.index, aggregated_ratings["mean"], color="red")

    ax1: Axes = axes[1]
    ax1.set_xlabel("Item")
    ax1.set_ylabel("Number of ratings (log)")
    ax1.bar(aggregated_ratings.index, aggregated_ratings["count"])

    # currently a bit hacky, but when we have reset the index the id and the value does not match anymore,
    # so we just hide it (we couldnt display the actual one anyway, it would just be a big mess of numbers).
    if hide_xticks:
        ax0.set_xticks([])
        ax1.set_xticks([])


def main():
    df = get_reg_train_data()

    aggregated_ratings = df.groupby("item")["rating"].agg(["mean", "count"])

    make_plot(aggregated_ratings)
    make_plot(aggregated_ratings.sort_values(by="mean").reset_index(), hide_xticks=True)
    make_plot(
        aggregated_ratings.sort_values(by="count").reset_index(), hide_xticks=True
    )

    plt.show()


if __name__ == "__main__":
    main()
