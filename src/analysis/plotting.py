import matplotlib.pyplot as plt
import numpy as np


COLORS = [
    "mediumseagreen",
    "darkblue",
    "mediumvioletred",
    "darkorchid",
    "dodgerblue",
    "salmon",
    "darkgreen",
]


def plot_mean_e_values(
    distance_list: list,
    e_values_list: list,
    biases: list,
    min_threshold: int = 0,
    max_threshold: int = 300,
    outputfilename: str = "evaluemeans",
    plot_stds: bool = True,
    _plot_lengths: bool = False,
    title: str = "",
    scatter_size: int = 3,
):
    """This plots the correlation between the average
    hmmer e values to the model's similarity score
    at different thresholds of similarity

    A good result will show that low evalues
    are correlated with high similarity
    so that the average e value decreases as
    the similarity threshold increases"""
    plt.clf()
    thresholds = np.linspace(min_threshold, max_threshold, 100)
    all_distance = np.array(distance_list)
    all_e_values = np.array(e_values_list)
    biases = np.array(biases)

    means = []
    stds = []
    lengths = []
    mean_bias = []
    for threshold in thresholds:
        idx = np.where(all_distance > threshold)[0]
        mean = np.mean(np.ma.masked_invalid(np.log10(all_e_values[idx])))
        std = np.std(np.ma.masked_invalid(np.log10(all_e_values[idx])))
        bias = np.mean(biases[idx])
        means.append(mean)
        stds.append(std)
        length = len(idx)
        lengths.append(length)
        mean_bias.append(bias)
    lengths = np.log(lengths)
    plt.scatter(thresholds, means, c=mean_bias, cmap="Greens", s=scatter_size)
    plt.plot(thresholds, means)
    if plot_stds:
        plt.fill_between(
            thresholds,
            np.array(means) - np.array(stds) / 2,
            np.array(means) + np.array(stds) / 2,
            alpha=0.5,
        )
    if _plot_lengths:
        plt.fill_between(
            thresholds,
            np.array(means) - np.array(lengths),
            np.array(means) + np.array(lengths),
            alpha=0.5,
            color="orange",
        )
    # plt.title(title + ", Full")
    # plt.ylabel("Log E value means")
    # plt.xlabel("Similarity Threshold")
    # plt.savefig(f"ResNet1d/results/{outputfilename}-full.png")

    plt.title(title)
    plt.ylim(-20, 0)
    plt.xlim(0, 100)
    plt.ylabel("Log E value means")
    plt.xlabel("Similarity Threshold")
    plt.savefig(f"ResNet1d/results/{outputfilename}.png")


def plot_roc_curve(
    figure_path: str,
    filtrations: list,
    recalls: list,
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
):
    """This plots the ROC curve comparing the
    recall and filtration of this model to hmmer hits
    The filtration is currently being thresholded to be
    greater than 80%
    Before running this function you need to construct
    a path (argument filename) that counts the recall
    and filtration line by line for a given evalue threshold

    The numpos arguments refer to the number of positives
    for a given evalue threshold"""
    print("Generating ROC plot")

    num_thresholds = len(evalue_thresholds)
    _, axis = plt.subplots(figsize=(10, 10))
    axis.set_title(f"{os.path.splitext(os.path.basename(figure_path))[0]}")

    for i in range(num_thresholds):
        axis.plot(
            np.array(filtrations)[:, i],
            np.array(recalls)[:, i],
            f"{COLORS[i]}",
            linewidth=2,
            label=evalue_thresholds[i],
        )
    axis.set_xlabel("filtration")
    axis.set_ylabel("recall")
    axis.ticklabel_format(style="plain", axis="x", useOffset=False)
    axis.grid()
    axis.set_ylim(0, 105)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    plt.legend()
    plt.savefig(f"{figure_path}", bbox_inches="tight")
    plt.close()
