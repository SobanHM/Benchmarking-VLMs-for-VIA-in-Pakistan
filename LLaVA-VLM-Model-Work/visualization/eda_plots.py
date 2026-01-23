import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pycocoevalcap.cider.cider import Cider

def plot_metric_distribution(scores_dict, metric_name):
    """
    Plots a histogram of scores for a single metric to see spread.
    scores_dict: list of scores per image
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(scores_dict, kde=True, color='skyblue')
    plt.title(f'LLaVA Supermarket responses\nDistribution of {metric_name} Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(f'results_{metric_name}_dist.png')
    plt.close()

def plot_radar_chart(results_dict):

    categories = list(results_dict.keys())
    values = list(results_dict.values())
    # number of variables
    N = len(categories)

    # angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop
    values += values[:1]  # close the loop

    ax = plt.subplot(111, polar=True)
    # draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    plt.title("Lexical Performance Overview")
    plt.savefig('semantic_results_radar.png')
    plt.close()
