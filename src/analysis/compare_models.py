"""Model comparison"""

from src.analysis.benchmarking import (
    get_data,
    generate_roc,
    get_data_for_roc,
    plot_mean_e_values,
    get_roc_data,
    COLORS,
)
from src.analysis.eval_data_config import (
    load_inputs,
    all_hits_max_file_4,
    all_hits_normal_file_4,
)
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import resource

resource.setrlimit(resource.RLIMIT_DATA, (500 * 1024**3, -1))
import os


def load_hmmer_hits(query_id: int = 4):
    """Loads pre-saved hmmer hits dictionaries for a given
    evaluation query id, currently can only be 4 or 0"""
    if query_id == 4:
        with open(all_hits_max_file_4 + ".pkl", "rb") as file:
            all_hits_max_4 = pickle.load(file)
        with open(all_hits_normal_file_4 + ".pkl", "rb") as file:
            all_hits_normal_4 = pickle.load(file)
        return all_hits_max_4, all_hits_normal_4
    else:
        raise Exception(f"No evaluation data for given query id {query_id}")


class Results:
    def __init__(
        self,
        model_results_path: str,
        hmmer_hits_dict: dict,
        data_savedir: str,
        evaluemeansfile: str,
        evaluemeanstitle: str,
        roc_filepath: str,
        plot_roc: bool = False,
        plot_e_values: bool = False,
        temp_file: str = None,
    ):
        """evaluates a given model"""

        if plot_e_values:
            (self.similarities, self.e_values, self.biases, sorted_pairs) = get_data(
                model_results_path,
                hmmer_hits_dict,
                data_savedir=data_savedir,
                plot_roc=plot_roc,
            )
            print("Plotting e values and saving to")
            print(evaluemeansfile)
            plot_mean_e_values(
                self.similarities,
                self.e_values,
                self.biases,
                min_threshold=0,
                max_threshold=np.max(self.similarities),
                outputfilename=evaluemeansfile,
                plot_stds=True,
                _plot_lengths=False,
                title=evaluemeanstitle,
            )
            if plot_roc:
                generate_roc(roc_filepath, hmmer_hits_dict, temp_file, sorted_pairs)
        elif plot_roc:
            if not os.path.exists(f"{temp_file}_filtration.pickle"):
                sorted_pairs = get_data_for_roc(
                    model_results_path,
                    hmmer_hits_dict,
                    data_savedir=data_savedir,
                    plot_roc=plot_roc,
                )
                generate_roc(roc_filepath, hmmer_hits_dict, temp_file, sorted_pairs)
            else:
                generate_roc(roc_filepath, hmmer_hits_dict, temp_file, None)


def compare_models(
    modelname: str = "CPU-5K-40",
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
):
    print(f"Comparing models with {modelname}")
    all_hits_max, _ = load_hmmer_hits(4)

    neat_max = load_inputs(all_hits_max, "max", modelname)

    esm = load_inputs(all_hits_max, "max", "esm")
    knn = load_inputs(all_hits_max, "max", "knn-for-homology")
    mmseqs = load_inputs(all_hits_max, "max", "mmseqs")
    protbert = load_inputs(all_hits_max, "max", "protbert")
    last = load_inputs(all_hits_max, "max", "last")
    hmmer_normal = load_inputs(all_hits_max, "max", "msv")

    all_recalls = []
    all_filtrations = []

    for inputs in [esm, protbert, neat_max, hmmer_normal, last, mmseqs, knn]:
        if os.path.exists(f"{inputs['temp_file']}_filtration.pickle"):
            print("Loading filtration and recall directly")
            with open(f"{inputs['temp_file']}_filtration.pickle", "rb") as pickle_file:
                filtrations = pickle.load(pickle_file)
            with open(f"{inputs['temp_file']}_recall.pickle", "rb") as pickle_file:
                recalls = pickle.load(pickle_file)
        else:
            print(f" No such file {inputs['temp_file']}_filtration.pickle")

            (_, _, _, sorted_pairs) = get_data(**inputs)

            filtrations, recalls = get_roc_data(**inputs, sorted_pairs=sorted_pairs)
        all_recalls.append(recalls)
        all_filtrations.append(filtrations)

    print(len(all_filtrations))
    print(len(all_recalls))

    labels = [
        "ESM",
        "ProtBERT",
        "NEAR-40",
        "MSV filter",
        "LAST",
        "MMseqs2",
        "ProtTransT5",
    ]

    for evalue_index in [-1, -2, -3, -4]:
        _, axis = plt.subplots(figsize=(10, 10))
        idx = -1
        print(f"Evalue index:{evalue_index}")
        for recalls, filtrations in zip(all_recalls, all_filtrations):
            idx += 1
            print(f"IDX: {idx}")

            if labels[idx] in ["LAST", "MMseqs2", "ProtTransT5"]:
                axis.scatter(
                    np.array(filtrations)[-1, evalue_index],
                    np.array(recalls)[-1, evalue_index],
                    c=COLORS[idx],
                    s=100,
                    label=labels[idx],
                    marker="x",
                )
            else:
                axis.plot(
                    np.array(filtrations)[:, evalue_index],
                    np.array(recalls)[:, evalue_index],
                    f"{COLORS[idx]}",
                    linewidth=2,
                    label=labels[idx],
                )
        axis.set_xlabel("filtration")
        axis.set_ylabel("recall")
        axis.grid()
        axis.legend(loc="lower left")
        axis.set_xlim(97.5, 100.1)
        axis.set_xticks([97.5, 98, 98.5, 99, 99.5, 100])

        plt.title(f"Evalue threshold: {evalue_thresholds[evalue_index]}")

        plt.savefig(f"ResNet1d/results/compared_roc-{evalue_thresholds[evalue_index]}-zoomed.png")
        plt.clf()


def compare_nprobe(evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10], normal=False):
    styles = ["dashed", "solid"]

    print(f"Comparing NEAT models")
    all_hits_max, _all_hits_normal = load_hmmer_hits(4)

    if normal:
        align = load_inputs(_all_hits_normal, "normal", "CPU-20K-50")
        align2 = load_inputs(_all_hits_normal, "normal", "CPU-20K-150")
    else:
        align = load_inputs(all_hits_max, "max", "CPU-5K-5")
        align2 = load_inputs(all_hits_max, "max", "CPU-5K-10")
        align3 = load_inputs(all_hits_max, "max", "CPU-5K-20")
        align4 = load_inputs(all_hits_max, "max", "CPU-5K-40")
        align1 = load_inputs(all_hits_max, "max", "CPU-5K-50")
    nprobes = [5, 10, 20, 40, 50]

    all_filtrations = []
    all_recalls = []
    for idx, inputs in enumerate([align, align2, align3, align4, align1]):
        filtrations, recalls = get_roc_data(**inputs)
        all_filtrations.append(filtrations)
        all_recalls.append(recalls)
    for i in [0, 1, 2, 3]:
        idx = 0
        _, axis = plt.subplots(figsize=(10, 10))
        for f, r in zip(all_filtrations, all_recalls):
            axis.plot(
                np.array(f)[:, i],
                np.array(r)[:, i],
                f"{COLORS[idx]}",
                linewidth=2,
                label=f"NEAT-{nprobes[idx]}, <{evalue_thresholds[i]}",
            )
            idx += 1
        axis.set_xlabel("filtration", fontsize=12)
        axis.set_ylabel("recall", fontsize=12)
        axis.grid()
        print("Saving figure")
        plt.legend()
        plt.savefig(f"ResNet1d/results/superimposedCPUmax-{evalue_thresholds[i]}.png")

        plt.clf()

    _, axis = plt.subplots(figsize=(10, 10))

    for i in [0, 1, 2, 3]:
        idx = 0
        _, axis = plt.subplots(figsize=(10, 10))
        for f, r in zip(all_filtrations, all_recalls):
            axis.plot(
                np.array(f)[:, i],
                np.array(r)[:, i],
                f"{COLORS[idx]}",
                linewidth=2,
                label=f"NEAT-{nprobes[idx]}, <{evalue_thresholds[i]}",
                #   linestyle=styles[idx],
            )
            idx += 1
        axis.set_xlabel("filtration", fontsize=12)
        axis.set_ylabel("recall", fontsize=12)
        axis.set_ylim(90, 100.2)
        axis.set_xlim(97.5, 100.1)
        axis.grid()
        axis.set_xticks([97.5, 98, 98.5, 99, 99.5, 100], fontsize=12)
        axis.set_yticks([90, 92, 94, 96, 98, 100], fontsize=12)

        plt.legend()
        print("Saving figure")

        plt.savefig(f"ResNet1d/results/superimposedCPUmax-zoomed-{evalue_thresholds[i]}.png")
        plt.clf()


def evaluate(
    modes: list = ["normal", "max"],
    modelname: str = None,
):
    """Main function for evaluation"""

    print(f"Evaluating {modelname}")

    all_hits_max, all_hits_normal = load_hmmer_hits(4)

    if "max" in modes:
        print("Parsing Alignment Model IVF Query 4 Max")

        align_ivf_max_inputs_4 = load_inputs(all_hits_max, "max", modelname)
        _ = Results(**align_ivf_max_inputs_4)
    if "normal" in modes:
        align_ivf_normal_inputs_4 = load_inputs(all_hits_normal, "normal", modelname)

        print("Parsing Alignment Model IVF Query 4 Normal")
        _ = Results(**align_ivf_normal_inputs_4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, default=4)
    parser.add_argument("--modes", type=str, default=["MNF"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--impose", action="store_true")

    args = parser.parse_args()

    modeinitials = args.modes
    modelname = args.modelname

    modes = []

    if "M" in modeinitials:
        modes.append("max")
    if "N" in modeinitials:
        modes.append("normal")

    if args.compare:
        compare_models(modelname=modelname)
        # plot_recall_by_evalue_threshold()
    elif args.impose:
        compare_nprobe()
    else:
        evaluate(modes, modelname)