import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def compare_fairness(other, baseline):
  config = other["config"].iloc[0]
  other = other.drop(columns=["bias_type", "config_name"])
  baseline = baseline.drop(columns=["bias_type", "config_name"])

  merged = pd.merge(
      other, 
      baseline,
      how="inner",
      on="score_name"
    )

  score_names = merged["score_name"].values
  config_measurements = {
      'baseline': merged["score_y"].values,
      config: merged["score_x"].values,
  }

  x = np.arange(len(score_names))  # the label locations
  width = 0.25  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')

  for config, measurement in config_measurements.items():
      offset = width * multiplier
      rects = ax.bar(x + offset, measurement, width, label=config)
      ax.bar_label(rects, padding=3)
      multiplier += 1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Fairness Score')
  ax.set_title('Fairness Score by Task')
  ax.set_xticks(x + width, score_names)
  ax.legend(loc='upper left', ncols=3)
  
  high = max(merged["score_y"].values)
  low = min(merged["score_y"].values)
  plt.ylim(bottom=low-0.5*(high-low))
  plt.ylim(top=high+0.5*(high-low))

  plt.show()
