import matplotlib.pyplot as plt
import numpy as np
import os

np.random.seed(0)

def visualize(df):
    for score_name in df.score_name.unique():
        print(score_name)
        score_name_df = df[df['score_name'] == score_name]
        print(score_name_df)
        _plot(score_name_df.config, score_name_df.score, score_name)


def _save(title):
    dir = "visualization/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + title)


def _plot(x, y, title):
    title = title.replace("_", " ").title()
    plt.bar(x, y)
    plt.title(title)
    _save(title)
    plt.show()
