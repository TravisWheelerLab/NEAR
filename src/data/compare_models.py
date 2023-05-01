"""Model comparison

You need to compare on data with query id 0 as well 

You need to increase max sequence length too"""

from src.data.benchmarking import (
    get_data,
    generate_roc,
    plot_mean_e_values,
    plot_lengths,
)
from src.data.eval_data_config import (
    load_alignment_inputs,
    load_knn_inputs,
    load_esm_inputs,
    all_hits_max_file_4,
    all_hits_normal_file_4,
)
import numpy as np
import pickle
import argparse


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


def get_overlap(all_hits_max: dict, all_hits_normal: dict):
    """Calculates the overlapping hits between hmmer normal
    and hmmer max outputs"""
    hmmer_normal = []
    hmmer_max_only = []
    num_missing = 0
    missingvalues = []
    missingpairs = []
    num_overlap = 0
    for q, v in all_hits_max.items():
        if q in all_hits_normal:
            for t, values in v.items():
                if t in all_hits_normal[q]:
                    hmmer_normal.append(values)
                else:
                    hmmer_max_only.append(values)

    for query, v in all_hits_normal.items():

        for t, values in v.items():
            if t not in all_hits_max[query]:
                num_missing += 1
                missingpairs.append([(query, t)])
                missingvalues.append(values)
            else:
                num_overlap += 1

    print(f"Num missing from HMMER Max that are in HMMER Normal: {num_missing}")
    print(f"Number overlapping in both HMMER Max and HMMER Normal: {num_overlap}")
    print(f"Number of hits in HMMER Max not in HMMER Normal: {len(hmmer_normal)}")

    return (
        hmmer_normal,
        hmmer_max_only,
        num_missing,
        num_overlap,
        missingvalues,
        missingpairs,
    )


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

        (
            self.hits_dict,
            self.similarities,
            self.e_values,
            self.biases,
            self.numhits,
        ) = get_data(model_results_path, hmmer_hits_dict, savedir=data_savedir)
        # pdb.set_trace()
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
            _plot_lengths=True,
            title=evaluemeanstitle,
        )
        if plot_roc:
            generate_roc(model_results_path, hmmer_hits_dict, roc_filepath, temp_file)


def evaluate(
    query_id=4,
    models: list = ["align", "knn"],
    modes: list = ["normal", "max"],
    modelname: str = None,
    lengths: bool = True,
):
    """Main function for evaluation"""
    if query_id == 4:
        all_hits_max, all_hits_normal = load_hmmer_hits(query_id)

        if "align" in models:
            if "max" in modes:
                print("Parsing Alignment Model IVF Query 4 Max")

                align_ivf_max_inputs_4 = load_alignment_inputs(all_hits_max, "max", modelname)
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
        if lengths:
            plot_lengths(alignment_model_ivf_max.similarities, alignment_model_knn_max.similarities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, default=4)
    parser.add_argument("--models", type=str, default=["ABK"])
    parser.add_argument("--modes", type=str, default=["MNF"])
    parser.add_argument("--lengths", action="store_true")
    parser.add_argument("--modelname", type=str)

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
    if "M" in modeinitials:
        modes.append("max")
    if "N" in modeinitials:
        modes.append("normal")

    evaluate(args.query_id, models, modes, modelname, lengths=args.lengths)
