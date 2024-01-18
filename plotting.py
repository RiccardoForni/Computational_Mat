import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr(matrix):
    sns.heatmap(matrix, cmap="Greens",annot=True)
    plt.savefig("correlation matrix.png")