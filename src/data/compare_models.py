"""Model comparison"""

from src.data.benchmarking import (
    get_data,
    generate_roc,
    plot_mean_e_values,
    get_roc_data,
    COLORS,
)
from src.data.eval_data_config import (
    load_inputs,
    all_hits_max_file_4,
    all_hits_normal_file_4,
)
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import pdb
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
        query_lengths_file: str,
        target_lengths_file: str,
        evaluemeansfile: str = None,
        evaluemeanstitle: str = None,
        roc_filepath: str = None,
        plot_roc: bool = False,
        temp_file: str = None,
        plot_e_values: bool = False,
        norm_q=False,
        norm_t=False,
        *args,
        **kwargs,
    ):
        """evaluates a given model"""

        if not os.path.exists(f"{temp_file}_filtration.pickle"):
            (_, _, _, sorted_pairs) = get_data(
                model_results_path,
                hmmer_hits_dict,
                data_savedir=data_savedir,
                plot_roc=plot_roc,
                query_lengths_file=query_lengths_file,
                target_lengths_file=target_lengths_file,
                norm_t=norm_t,
                norm_q=norm_q,
            )
        if plot_e_values:
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
            if not os.path.exists(f"{temp_file}_filtration.pickle"):
                (_, _, _, sorted_pairs) = get_data(
                    model_results_path,
                    hmmer_hits_dict,
                    query_lengths_file=query_lengths_file,
                    target_lengths_file=target_lengths_file,
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

    neat_max = load_inputs(all_hits_max, modelname)

    esm = load_inputs(all_hits_max, "esm")
    knn = load_inputs(all_hits_max, "knn-for-homology")
    mmseqs = load_inputs(all_hits_max, "mmseqs")
    protbert = load_inputs(all_hits_max, "protbert")
    last = load_inputs(all_hits_max, "last")
    hmmer_normal = load_inputs(all_hits_max, "msv")

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
        # axis.set_xticks([75, 80, 85, 90, 95, 100])
        axis.set_xticks([97.5, 98, 98.5, 99, 99.5, 100])

        # axis.set_ylim(90, 100.2)
        # axis.set_xlim(99, 100.01)
        # axis.grid()
        # axis.set_xticks([99, 99.2, 99.4, 99.6, 99.8, 100], fontsize=12)
        # axis.set_yticks([90, 92, 94, 96, 98, 100], fontsize=12)

        plt.title(f"Evalue threshold: {evalue_thresholds[evalue_index]}")

        plt.savefig(
            f"ResNet1d/results/compared_roc-{evalue_thresholds[evalue_index]}-zoomed.png"
        )
        plt.clf()


def impose_plots(evalue_thresholds: list = [1e-10, 1e-4, 1e-1]):
    all_hits_max, _ = load_hmmer_hits(4)
    gpu_50 = load_inputs(all_hits_max, "GPU-5K-50-masked", norm_q=True, norm_t=True)
    gpu_150 = load_inputs(all_hits_max, "GPU-5K-150-masked", norm_q=True, norm_t=True)

    cpu_5 = load_inputs(all_hits_max, "CPU-5K-5-masked", norm_q=True, norm_t=True)
    cpu_10 = load_inputs(all_hits_max, "CPU-5K-10-masked", norm_q=True, norm_t=True)
    cpu_20 = load_inputs(all_hits_max, "CPU-5K-20-masked", norm_q=True, norm_t=True)

    hmmer_normal = load_inputs(all_hits_max, "msv")

    # nprobes = [50, 150, 5, 10, 20]
    # runtimes = ["0.019s/q", "0.034s/q", "0.074s/q", "0.129s/q", "0.240s/q", "0.290s/q"]
    nprobes = [150, 20, 50, 10, 5]
    runtimes = [
        "0.290s/q",
        "0.034s/q",
        "0.240s/q",
        "0.019s/q",
        "0.129s/q",
        "0.240s/q",
    ]

    all_filtrations = []
    all_recalls = []
    for idx, inputs in enumerate(
        [hmmer_normal, gpu_150, cpu_20, gpu_50, cpu_10, cpu_5]
        # [gpu_50, gpu_150, cpu_5, cpu_10, cpu_20, hmmer_normal]
    ):
        filtrations, recalls = get_roc_data(**inputs)
        all_filtrations.append(filtrations)
        all_recalls.append(recalls)

    for i in [0, 1, 2]:
        idx = 0
        _, axis = plt.subplots(figsize=(10, 10))
        for f, r in zip(all_filtrations, all_recalls):
            if idx in [1, 3]:
                label = f"NEAT-GPU-{nprobes[idx]} <{evalue_thresholds[i]}, run-time: {runtimes[idx]}"
                linestyle = "dashed"
            elif idx == 0:
                label = f"MSV filter <{evalue_thresholds[i]}, run-time: {runtimes[idx]}"
                linestyle = "dotted"
            else:
                label = f"NEAT-CPU-{nprobes[idx]} <{evalue_thresholds[i]}, run-time: {runtimes[idx]}"
                linestyle = "solid"
            axis.plot(
                np.array(f)[:, i],
                np.array(r)[:, i],
                f"{COLORS[idx]}",
                linewidth=2,
                label=label,
                linestyle=linestyle,
            )
            idx += 1
        axis.set_xlabel("Percent Filtration", fontsize=15)
        axis.set_ylabel("Percent Recall", fontsize=15)
        axis.set_ylim(90, 100.2)
        axis.set_xlim(97.5, 100.1)
        axis.grid()
        axis.set_xticks([97.5, 98, 98.5, 99, 99.5, 100], fontsize=15)
        axis.set_yticks([90, 92, 94, 96, 98, 100], fontsize=15)

        plt.legend(fontsize=15)
        print("Saving figure")

        filename = "ResNet1d/results/imposedplot"
        plt.title(
            f"NEAT Performance on HMMER Max for E-value Threshold {evalue_thresholds[i]}",
            fontsize=15,
        )
        plt.savefig(f"{filename}-{evalue_thresholds[i]}.png")
        plt.clf()


def compare_nprobe(evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10], gpu=False):
    styles = ["dashed", "solid"]

    print(f"Comparing NEAT models")
    all_hits_max, _ = load_hmmer_hits(4)
    if gpu:
        align = load_inputs(all_hits_max, "GPU-5K-50-masked", norm_q=True, norm_t=True)
        align2 = load_inputs(
            all_hits_max, "GPU-5K-100-masked", norm_q=True, norm_t=True
        )
        align3 = load_inputs(
            all_hits_max, "GPU-5K-120-masked", norm_q=True, norm_t=True
        )
        align4 = load_inputs(
            all_hits_max, "GPU-5K-150-masked", norm_q=True, norm_t=True
        )
        all_inputs = [align, align2, align3, align4]
        nprobes = [50, 100, 120, 150]
    else:
        align = load_inputs(all_hits_max, "CPU-5K-5-masked", norm_q=True, norm_t=True)
        align2 = load_inputs(all_hits_max, "CPU-5K-10-masked", norm_q=True, norm_t=True)
        align3 = load_inputs(all_hits_max, "CPU-5K-20-masked", norm_q=True, norm_t=True)
        align4 = load_inputs(all_hits_max, "CPU-5K-40-masked", norm_q=True, norm_t=True)
        align1 = load_inputs(all_hits_max, "CPU-5K-50-masked", norm_q=True, norm_t=True)

        all_inputs = [align, align2, align3, align4, align1]
        nprobes = [5, 10, 20, 40, 50]

    # _, axis = plt.subplots(figsize=(10, 10))

    all_filtrations = []
    all_recalls = []
    for idx, inputs in enumerate(all_inputs):
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
                # linestyle=styles[idx],
            )
            idx += 1
        axis.set_xlabel("filtration", fontsize=12)
        axis.set_ylabel("recall", fontsize=12)
        axis.grid()
        print("Saving figure")
        plt.legend()
        # if normal:
        # plt.savefig("ResNet1d/results/superimposedCPUnormal.png")
        if gpu:
            filename = "ResNet1d/results/superimposedGPUmax"
        else:
            filename = "ResNet1d/results/superimposedCPUmax"
        plt.savefig(f"{filename}-{evalue_thresholds[i]}.png")

        plt.clf()

    # again with different X limit

    _, axis = plt.subplots(figsize=(10, 10))

    # for idx, inputs in enumerate([align, align2]):
    # filtrations, recalls = get_roc_data(**inputs)

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
        # if normal:
        # plt.savefig("ResNet1d/results/superimposedCPUnormal-zoomed.png")
        # else:
        if gpu:
            filename = "ResNet1d/results/superimposedGPUmax-zoomed"
        else:
            filename = "ResNet1d/results/superimposedCPUmax-zoomed"
        plt.savefig(f"{filename}-{evalue_thresholds[i]}.png")
        plt.clf()


def plot_recall_by_evalue_threshold(
    modelname: str = "CPU-20K-150", evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10]
):
    print(f"Comparing models with {modelname}")
    all_hits_max, all_hits_normal = load_hmmer_hits(4)

    neat_max = load_inputs(all_hits_max, "max", modelname)
    neat_regular = load_inputs(all_hits_normal, "normal", modelname)

    esm = load_inputs(all_hits_max, "max", "esm")
    knn = load_inputs(all_hits_max, "max", "knn-for-homology")
    mmseqs = load_inputs(all_hits_max, "max", "mmseqs")
    protbert = load_inputs(all_hits_max, "max", "protbert-1")
    last = load_inputs(all_hits_max, "max", "")

    evalue_recalls = []
    _, axis = plt.subplots(figsize=(10, 10))

    for idx, inputs in enumerate([esm, knn, protbert, neat_max, mmseqs, last]):
        if os.path.exists(f"{inputs['temp_file']}_filtration.pickle"):
            print("Loading filtration and recall directly")
            with open(f"{inputs['temp_file']}_recall.pickle", "rb") as pickle_file:
                recalls = pickle.load(pickle_file)
        else:
            print(f" No such file {inputs['temp_file']}_filtration.pickle")

            (_, _, _, sorted_pairs) = get_data(**inputs)

            _, recalls = get_roc_data(**inputs, sorted_pairs=sorted_pairs)
        # evalue_recalls.append(recalls[-1])

        plt.plot(
            evalue_thresholds,
            np.array(recalls)[-1, :],
            label=[
                "ESM",
                "ProtTransT5XLU50",
                "ProtBERT",
                "NEAT-150",
                "MMseqs2",
                "LAST",
            ][idx],
        )
    plt.legend()
    plt.title("HMMER Max Recall by Evalue Threshold")
    # plt.savefig(f"ResNet1d/results/compared_recall.png")
    plt.xlabel("E-value thresholds")
    plt.ylabel("Recall")
    plt.xticks([0, 1, 2, 3], labels=evalue_thresholds)
    plt.savefig(f"ResNet1d/results/compared_recall.png")
    plt.clf()

    esm = load_inputs(all_hits_max, "normal", "esm")
    knn = load_inputs(all_hits_max, "normal", "knn-for-homology")
    mmseqs = load_inputs(all_hits_max, "normal", "mmseqs")
    protbert = load_inputs(all_hits_max, "normal", "protbert")
    last = load_inputs(all_hits_max, "normal", "")

    _, axis = plt.subplots(figsize=(10, 10))

    for idx, inputs in enumerate([esm, knn, protbert, neat_regular, mmseqs, last]):
        if os.path.exists(f"{inputs['temp_file']}_filtration.pickle"):
            print("Loading filtration and recall directly")
            with open(f"{inputs['temp_file']}_recall.pickle", "rb") as pickle_file:
                recalls = pickle.load(pickle_file)
        else:
            print(f" No such file {inputs['temp_file']}_filtration.pickle")

            (_, _, _, sorted_pairs) = get_data(**inputs)

            _, recalls = get_roc_data(**inputs, sorted_pairs=sorted_pairs)

        plt.plot(
            evalue_thresholds,
            np.array(recalls)[-1, :],
            label=[
                "ESM",
                "ProtTransT5XLU50",
                "ProtBERT",
                "NEAT-150",
                "MMseqs2",
                "LAST",
            ][idx],
        )
    plt.legend()
    plt.xlabel("E- value thresholds")
    plt.ylabel("Recall")

    plt.title("HMMER Normal Recall by Evalue Threshold")
    plt.savefig(f"ResNet1d/results/compared_recall_normal.png")


def evaluate(modelname, norm_q=False, norm_t=False):
    """Main function for evaluation"""

    print(f"Evaluating {modelname}")

    all_hits_max, _ = load_hmmer_hits(4)

    print("Parsing Alignment Model IVF Query 4 Max")

    align_ivf_max_inputs_4 = load_inputs(
        all_hits_max, modelname, norm_q=norm_q, norm_t=norm_t
    )
    _ = Results(**align_ivf_max_inputs_4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, default=4)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--norm_q", action="store_true")
    parser.add_argument("--norm_t", action="store_true")

    parser.add_argument("--modelname", type=str)
    parser.add_argument("--impose", action="store_true")
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    modelname = args.modelname

    norm_q = args.norm_q
    norm_t = args.norm_t
    print(f"Normalise queries: {norm_q}")
    print(f"Normalise targets: {norm_t}")
    if args.compare:
        compare_models(modelname=modelname)
        # plot_recall_by_evalue_threshold()
    elif args.impose:
        # compare_nprobe(gpu=args.gpu)
        impose_plots()
    else:
        evaluate(modelname, norm_q, norm_t)
