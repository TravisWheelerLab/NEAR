"""Model comparison

You need to compare on data with query id 0 as well 

You need to increase max sequence length too"""

from src.data.benchmarking import (
    get_data,
    generate_roc_from_sorted_pairs,
    plot_mean_e_values,
    plot_lengths,
)
from src.data.eval_data_config import (
    load_alignment_inputs,
    load_blosum_inputs,
    load_kmer_inputs,
    all_hits_max_file_4,
    all_hits_normal_file_4,
    all_hits_max_file_0,
    all_hits_normal_file_0,
)
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--query_id", type=int, default=4)
parser.add_argument("--models",type = str, default=["ABK"])
parser.add_argument("--modes", type = str,default=["MNF"])
parser.add_argument("--lengths", action='store_true')

args = parser.parse_args()


def load_hmmer_hits(query_id: int = 4):
    """Loads pre-saved hmmer hits dictionaries for a given
    evaluation query id, currently can only be 4 or 0"""
    if query_id == 4:
        with open(all_hits_max_file_4 + ".pkl", "rb") as file:
            all_hits_max_4 = pickle.load(file)
        with open(all_hits_normal_file_4 + ".pkl", "rb") as file:
            all_hits_normal_4 = pickle.load(file)
        return all_hits_max_4, all_hits_normal_4
    elif query_id == 0:
        with open(all_hits_max_file_0, "rb") as file:
            all_hits_max_0 = pickle.load(file)
        with open(all_hits_normal_file_0, "rb") as file:
            all_hits_normal_0 = pickle.load(file)
        return all_hits_max_0, all_hits_normal_0
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

    print(
        f"Num missing from HMMER Max that are in HMMER Normal: {num_missing}"
    )
    print(
        f"Number overlapping in both HMMER Max and HMMER Normal: {num_overlap}"
    )
    print(
        f"Number of hits in HMMER Max not in HMMER Normal: {len(hmmer_normal)}"
    )

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
        sorted_alignment_pairs_path: str,
        temp_data_file: str,
        roc_filepath: str,
        plot_roc: bool = False,
        num_pos_per_evalue: list = None,
        num_hits: int = None,
        plot_evalue_means: bool = False,
    ):
        """evaluates a given model"""

        (
            self.hits_dict,
            self.similarities,
            self.e_values,
            self.biases,
            self.numhits,
        ) = get_data(model_results_path, hmmer_hits_dict, savedir=data_savedir)

        plot_mean_e_values(
            self.similarities,
            self.e_values,
            self.biases,
            min_threshold=0,
            max_threshold=np.max(self.similarities),
            outputfilename=evaluemeansfile,
            plot_stds=True,
            plot_lengths=True,
            title=evaluemeanstitle,
        )
        if plot_roc:
            generate_roc_from_sorted_pairs(
                model_results_path,
                sorted_alignment_pairs_path,
                temp_data_file,
                hmmer_hits_dict,
                roc_filepath,
                num_pos_per_evalue,
                num_hits,
            )


def evaluate(
    query_id=4,
    models: list = ["align", "blosum", "kmer"],
    modes: list = ["normal", "max", "flat","scann"],
    lengths: bool = True,
):
    """Main function for evaluation"""
    if query_id == 4:
        all_hits_max, all_hits_normal = load_hmmer_hits(query_id)

        if "align" in models:
            if "max" in modes:
                print("Parsing Alignment Model IVF Query 4 Max")

                align_ivf_max_inputs_4 = load_alignment_inputs(
                    all_hits_max, "max"
                )
                alignment_model_ivf_max = Results(**align_ivf_max_inputs_4)
            if "normal" in modes:
                align_ivf_normal_inputs_4 = load_alignment_inputs(
                    all_hits_normal, "normal"
                )

                print("Parsing Alignment Model IVF Query 4 Normal")
                _ = Results(**align_ivf_normal_inputs_4)
            if "scann" in modes:
                align_ivf_scann_inputs = load_alignment_inputs(
                    all_hits_max, "scann"
                )
                print("Parsing Alignment Model Scann Query 4 Normal")

                _ = Results(**align_ivf_scann_inputs)
	
        if "blosum" in models:
            if "normal" in modes:
                blosum_ivf_normal_inputs = load_blosum_inputs(
                    all_hits_normal, "normal"
                )
                print("Parsing Blosum Model IVF")
                _ = Results(**blosum_ivf_normal_inputs)
            if "max" in modes:
                blosum_ivf_max_inputs = load_blosum_inputs(all_hits_max, "max")

                print("Parsing Blosum Model IVF - Max")
                blosum_model_ivf_max = Results(**blosum_ivf_max_inputs)

            if "flat" in modes:
                blosum_flat_inputs = load_blosum_inputs(all_hits_max, "flat")
                print("Parsing Blosum Model Flat - Max")
                _ = Results(**blosum_flat_inputs)

        if "kmer" in models:
            if "max" in modes:
                kmer_inputs = load_kmer_inputs(all_hits_max, "max")
                print("Parsing Alignment Model KMER Max")
                alignment_model_kmer_max = Results(
                    **kmer_inputs
                )
            if "normal" in modes:
                kmer_inputs_normal = load_kmer_inputs(
                    all_hits_normal, "normal"
                )
                print("Parsing Alignment Model KMER Normal")
                _ = Results(**kmer_inputs_normal)

        if lengths:
            plot_lengths(
                alignment_model_ivf_max.similarities,
                blosum_model_ivf_max.similarities,
                alignment_model_kmer_max.similarities,
            )

    # elif query_id == 0:
    #     all_hits_max_0, all_hits_normal_0 = load_hmmer_hits(query_id)
    #     align_ivf_max_inputs_0 = {
    #         "model_results_path": ALIGNMENT_MODEL_RESULTS_PATH_IVF_0,
    #         "hmmer_hits_dict": all_hits_max_0,
    #         "data_savedir": "/xdisk/twheeler/daphnedemekas/alignment_model_ivf_0",
    #         "evaluemeansfile": "evaluemeans_align_ivf",
    #         "evaluemeanstitle": "Correlation in ALIGN IVF model - HMMER Max",
    #         "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_ivf_pairs_0.pkl",
    #         "temp_data_file": ALIGNMENT_MODEL_IVF_0_MAX_DATAFILE,
    #         "roc_filepath": "ResNet1d/eval_align_ivf_roc_0.png",
    #         "num_pos_per_evalue": [352617, 810152, 1105290, 2667980],
    #         "num_hits": 954905307,
    #         "plot_roc": True,
    #     }

    #     align_ivf_normal_inputs_0 = {
    #         "model_results_path": ALIGNMENT_MODEL_RESULTS_PATH_IVF_0,
    #         "hmmer_hits_dict": all_hits_normal_0,
    #         "data_savedir": "/xdisk/twheeler/daphnedemekas/alignment_model_ivf_0_normal",
    #         "evaluemeansfile": "evaluemeans_align_ivf_normal_0",
    #         "evaluemeanstitle": "Correlation in ALIGN IVF model - HMMER Normal",
    #         "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_ivf_pairs_0_normal.pkl",
    #         "temp_data_file": ALIGNMENT_MODEL_IVF_0_NORMAL_DATAFILE,
    #         "roc_filepath": "ResNet1d/eval_align_ivf_roc_normal_0.png",
    #         "num_pos_per_evalue": [352447, 763108, 794484, 798858],
    #         "num_hits": 954905307,
    #         "plot_roc": True,
    #     }

    #     kmer_inputs = {
    #         "model_results_path": KMER_MODEL_RESULTS_PATH,
    #         "hmmer_hits_dict": all_hits_max_0,
    #         "data_savedir": "/xdisk/twheeler/daphnedemekas/kmer_model_max",
    #         "evaluemeansfile": "evaluemeans_align_kmer_max",
    #         "evaluemeanstitle": "Correlation in ALIGN Kmer model - HMMER Max",
    #         "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_kmer_pairs_max.pkl",
    #         "temp_data_file": KMER_MODEL_DATAFILE,
    #         "roc_filepath": "ResNet1d/eval/align_kmer_roc.png",
    #     }

    #     kmer_inputs_normal = {
    #         "model_results_path": KMER_MODEL_RESULTS_PATH,
    #         "hmmer_hits_dict": all_hits_normal_0,
    #         "data_savedir": "/xdisk/twheeler/daphnedemekas/kmer_model_normal",
    #         "evaluemeansfile": "evaluemeans_align_kmer_normal",
    #         "evaluemeanstitle": "Correlation in ALIGN Kmer model - HMMER Normal",
    #         "sorted_alignment_pairs_path": "/xdisk/twheeler/daphnedemekas/sorted_alignment_kmer_pairs_normal.pkl",
    #         "temp_data_file": KMER_MODEL_DATAFILE_NORMAL,
    #         "roc_filepath": "ResNet1d/eval/align_kmer_roc_normal.png",
    #     }

    #     print("Parsing Alignment Model KMER Normal")
    #     _ = Results(**kmer_inputs_normal, plot_roc=False)

    #     print("Parsing Alignment Model KMER Max")
    #     _ = Results(**kmer_inputs, plot_roc=False)

    #     print("Parsing Alignment Model IVF Query 0")
    #     _ = Results(**align_ivf_normal_inputs_0)

    #     print("Parsing Alignment Model IVF Query 0 - Max")
    #     _ = Results(**align_ivf_max_inputs_0)

modelinitials = args.models
modeinitials = args.modes

models = []
modes = []
if 'A' in modelinitials:
    models.append("align")
if 'B' in modelinitials:
    models.append('blosum')
if 'K' in modelinitials:
    models.append('kmer')

if 'M' in modeinitials:
    modes.append('max')
if 'N' in modeinitials:
    modes.append('normal')
if 'F' in modeinitials:
    modes.append('flat')
if 'S' in modeinitials:
    modes.append('scann')
evaluate(args.query_id, models, modes, lengths = args.lengths)
