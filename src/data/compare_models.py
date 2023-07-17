"""Model comparison"""

from src.data.benchmarking import (
    get_data,
    generate_roc,
    plot_mean_e_values,
    get_roc_data,
    COLORS,
)
from src.data.eval_data_config import (
    load_alignment_inputs,
    load_knn_inputs,
    load_mmseqs_inputs,
    load_esm_inputs,
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
        evaluemeansfile: str,
        evaluemeanstitle: str,
        roc_filepath: str,
        plot_roc: bool = False,
        temp_file: str = None,
    ):
        """evaluates a given model"""

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


def compare_models(
    modelname: str = "CPU-20K-150",
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
):
    print(f"Comparing models with {modelname}")
    all_hits_max, all_hits_normal = load_hmmer_hits(4)

    neat_max = load_alignment_inputs(all_hits_max, "max", modelname)
    neat_regular = load_alignment_inputs(all_hits_normal, "normal", modelname)

    esm = load_esm_inputs(all_hits_max, "max", "esm")
    knn = load_knn_inputs(all_hits_max, "max", "knn-for-homology")
    mmseqs = load_knn_inputs(all_hits_max, "max", "mmseqs")
    protbert = load_knn_inputs(all_hits_max, "max", "protbert")
    last = load_mmseqs_inputs(all_hits_max, "max", "last")

    all_recalls = []
    all_filtrations = []

    for inputs in [esm, knn, protbert, neat_max, neat_regular, mmseqs, last]:
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
    for evalue_index in [-1, -2, -3, -4]:
        _, axis = plt.subplots(figsize=(10, 10))
        idx = -1
        print(f"Evalue index:{evalue_index}")
        for recalls, filtrations in zip(all_recalls, all_filtrations):
            idx += 1
            print(f"IDX: {idx}")

            axis.plot(
                np.array(filtrations)[:, evalue_index],
                np.array(recalls)[:, evalue_index],
                f"{COLORS[idx]}",
                linewidth=2,
                label=[
                    "ESM",
                    "ProtTransT5XLU50",
                    "ProtBERT",
                    "NEAT-150",
                    "NEAT-150 (HMMER Normal)",
                    "MMseqs2", "LAST",
                ][idx],
            )
        axis.set_xlabel("filtration")
        axis.set_ylabel("recall")
        #axis.grid()
        axis.legend()
        #axis.set_xlim(75, 101)
        #axis.set_xticks([75, 80, 85, 90, 95, 100])
        if evalue_index != -1:
            axis.set_ylim(50, 101)
            axis.set_yticks([50, 60, 70, 80, 90, 100])
        
        #axis.set_ylim(90, 100.2)
        axis.set_xlim(99, 100.01)
        axis.grid()
        axis.set_xticks([99, 99.2, 99.4, 99.6, 99.8, 100], fontsize=12)
        #axis.set_yticks([90, 92, 94, 96, 98, 100], fontsize=12)
    

        plt.savefig(
            f"ResNet1d/results/compared_roczoom2-{evalue_thresholds[evalue_index]}.png"
        )
        plt.clf()


def compare_nprobe(evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10], normal=False):
    styles = ["dashed", "solid"]

    print(f"Comparing NEAT models")
    all_hits_max, _all_hits_normal = load_hmmer_hits(4)

    if normal:
        align = load_alignment_inputs(_all_hits_normal, "normal", "CPU-20K-50")
        align2 = load_alignment_inputs(_all_hits_normal, "normal", "CPU-20K-250")
    else:
        align = load_alignment_inputs(all_hits_max, "max", "CPU-20K-50")
        align2 = load_alignment_inputs(all_hits_max, "max", "CPU-20K-250")

    nprobes = [50, 250]

    _, axis = plt.subplots(figsize=(10, 10))

    for idx, inputs in enumerate([align, align2]):
        filtrations, recalls = get_roc_data(**inputs)

        for i in [0, 1, 2, 3]:
            axis.plot(
                np.array(filtrations)[:, i],
                np.array(recalls)[:, i],
                f"{COLORS[i]}",
                linewidth=2,
                label=f"NEAT-{nprobes[idx]}, <{evalue_thresholds[i]}",
                linestyle=styles[idx],
            )
    axis.set_xlabel("filtration", fontsize=12)
    axis.set_ylabel("recall", fontsize=12)
    axis.grid()
    print("Saving figure")
    plt.legend()
    if normal:
        plt.savefig("ResNet1d/results/superimposedCPUnormal.png")
    else:
        plt.savefig("ResNet1d/results/superimposedCPUmax.png")

    plt.clf()

    # again with different X limit

    _, axis = plt.subplots(figsize=(10, 10))

    for idx, inputs in enumerate([align, align2]):
        filtrations, recalls = get_roc_data(**inputs)

        for i in [0, 1, 2, 3]:
            axis.plot(
                np.array(filtrations)[:, i],
                np.array(recalls)[:, i],
                f"{COLORS[i]}",
                linewidth=2,
                label=f"NEAT-{nprobes[idx]}, <{evalue_thresholds[i]}",
                linestyle=styles[idx],
            )
    axis.set_xlabel("filtration", fontsize=12)
    axis.set_ylabel("recall", fontsize=12)
    axis.set_ylim(90, 100.2)
    axis.set_xlim(95, 100.2)
    axis.grid()
    axis.set_xticks([95, 96, 97, 98, 99, 100], fontsize=12)
    axis.set_yticks([90, 92, 94, 96, 98, 100], fontsize=12)

    plt.legend()
    print("Saving figure")
    if normal:
        plt.savefig("ResNet1d/results/superimposedCPUnormal-zoomed.png")
    else:
        plt.savefig("ResNet1d/results/superimposedCPUmax-zoomed.png")


def plot_recall_by_evalue_threshold(
    modelname: str = "CPU-20K-150", evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10]
):
    print(f"Comparing models with {modelname}")
    all_hits_max, all_hits_normal = load_hmmer_hits(4)

    neat_max = load_alignment_inputs(all_hits_max, "max", modelname)
    neat_regular = load_alignment_inputs(all_hits_normal, "normal", modelname)

    esm = load_esm_inputs(all_hits_max, "max", "esm")
    knn = load_knn_inputs(all_hits_max, "max", "knn-for-homology")
    mmseqs = load_knn_inputs(all_hits_max, "max", "mmseqs")
    protbert = load_knn_inputs(all_hits_max, "max", "protbert")
    last = load_mmseqs_inputs(all_hits_max, "max", "last")

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
        #evalue_recalls.append(recalls[-1])

        plt.plot(
            evalue_thresholds,
            np.array(recalls)[-1,:],
            label=[
                "ESM",
                "ProtTransT5XLU50",
                "ProtBERT",
                "NEAT-150",
                "MMseqs2", "LAST",
            ][idx],
        )
    plt.legend()
    plt.title("HMMER Max Recall by Evalue Threshold")
    #plt.savefig(f"ResNet1d/results/compared_recall.png")
    plt.xlabel("E-value thresholds")
    plt.ylabel("Recall")
    plt.xticks([0,1,2,3], labels = evalue_thresholds)
    plt.savefig(f"ResNet1d/results/compared_recall.png")
    plt.clf()

    esm = load_esm_inputs(all_hits_max, "normal", "esm")
    knn = load_knn_inputs(all_hits_max, "normal", "knn-for-homology")
    mmseqs = load_knn_inputs(all_hits_max, "normal", "mmseqs")
    protbert = load_knn_inputs(all_hits_max, "normal", "protbert")
    last = load_mmseqs_inputs(all_hits_max, "normal", "last")

    evalue_recalls = []
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
        #evalue_recalls.append(recalls[-1])

        plt.plot(
            evalue_thresholds,
            np.array(recalls)[-1,:],
            label=[
                "ESM",
                "ProtTransT5XLU50",
                "ProtBERT",
                "NEAT-150",
                "MMseqs2", "LAST",
            ][idx],
        )
    plt.legend()
    plt.xlabel("E- value thresholds")
    plt.ylabel("Recall")
    
    plt.title("HMMER Normal Recall by Evalue Threshold")
    plt.savefig(f"ResNet1d/results/compared_recall_normal.png")


def evaluate(
    models: list = ["align", "knn"],
    modes: list = ["normal", "max"],
    modelname: str = None,
):
    """Main function for evaluation"""

    print(f"Evaluating {modelname}")

    all_hits_max, all_hits_normal = load_hmmer_hits(4)

    if "align" in models:
        if "max" in modes:
            print("Parsing Alignment Model IVF Query 4 Max")

            align_ivf_max_inputs_4 = load_alignment_inputs(
                all_hits_max, "max", modelname
            )
            alignment_model_ivf_max = Results(**align_ivf_max_inputs_4)
        if "normal" in modes:
            align_ivf_normal_inputs_4 = load_alignment_inputs(
                all_hits_normal, "normal", modelname
            )

            print("Parsing Alignment Model IVF Query 4 Normal")
            _ = Results(**align_ivf_normal_inputs_4)

    if "knn" in models:
        if "max" in modes:
            kmer_inputs = load_knn_inputs(all_hits_max, "max", modelname)
            print("Parsing Alignment Model KNN Max")
            alignment_model_knn_max = Results(**kmer_inputs)
        if "normal" in modes:
            kmer_inputs_normal = load_knn_inputs(all_hits_normal, "normal", modelname)
            print("Parsing Alignment Model KNN Normal")
            _ = Results(**kmer_inputs_normal)

    if "esm" in models:
        if "max" in modes:
            esm_inputs = load_esm_inputs(all_hits_max, "max", modelname)
            print("Parsing Alignment Model ESM Max")
            alignment_model_knn_max = Results(**esm_inputs)
        if "normal" in modes:
            esm_inputs_normal = load_esm_inputs(all_hits_normal, "normal", modelname)
            print("Parsing Alignment Model ESM Normal")
            _ = Results(**esm_inputs_normal)

    if "mmseqs" in models:
        if "max" in modes:
            mmseqs_inputs = load_mmseqs_inputs(all_hits_max, "max", modelname)
            print("Parsing Alignment Model ESM Max")
            alignment_model_knn_max = Results(**mmseqs_inputs)
        if "normal" in modes:
            mmseqs_inputs_normal = load_mmseqs_inputs(
                all_hits_normal, "normal", modelname
            )
            print("Parsing Alignment Model ESM Normal")
            _ = Results(**mmseqs_inputs_normal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, default=4)
    parser.add_argument("--models", type=str, default=["ABK"])
    parser.add_argument("--modes", type=str, default=["MNF"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--impose", action="store_true")

    args = parser.parse_args()

    modelinitials = args.models
    modeinitials = args.modes
    modelname = args.modelname

    models = []
    modes = []
    if "A" in modelinitials:
        models.append("align")
    if "K" in modelinitials:
        models.append("knn")
    if "E" in modelinitials:
        models.append("esm")
    if "M" in modelinitials:
        models.append("mmseqs")
    if "M" in modeinitials:
        modes.append("max")
    if "N" in modeinitials:
        modes.append("normal")

    if args.compare:
        # compare_models()
        plot_recall_by_evalue_threshold()
    elif args.impose:
        compare_nprobe()
    else:
        evaluate(models, modes, modelname)
