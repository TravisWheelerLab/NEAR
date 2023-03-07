""" Classes to interact with Fasta Files and Hmmer Hits files"""


import glob
import os
from typing import List, Tuple

import numpy as np


def dict_of_dicts(keys: list):
    _dict = {}
    for key in keys:
        _dict[key] = {}
    return _dict


class FastaFile:
    """Class representing a FASTA file.
    Class variables include lists of names and sequences,
    and a dictionary mapping names to sequences."""

    def __init__(self, filepath: str):

        self.filepath: str = filepath

        if not os.path.exists(filepath) or not filepath.endswith("fa"):
            raise f"The filepath is invalud: {filepath}"

        fastafile = open(filepath, "r", encoding="utf8")
        data: str = fastafile.readlines()

        self.data: dict = self.clean_data(data)

        self.uniref_names: List[str] = list(self.data.keys())
        self.sequences: List[str] = list(self.data.values())

    def clean_data(self, data: list) -> dict:
        """Cleans the fasta files and generates
        a dictionary mapping names to sequences"""
        data_dict = {}
        for i, line in enumerate(data):
            if i % 2 == 0:
                assert line[0] == ">", "This line does not begin with >"
                uniref_name = line.split()[0].strip(">")
                sequence_data = data[i + 1]
                sequence = sequence_data.strip("\n")
                data_dict[uniref_name] = sequence
        return data_dict


class HmmerHits:
    """Class for a HMMER hits file.
    Calling get_hits will return a dictionary of
    format {query:{target:data}}"""

    def __init__(self, dir_path: str = "uniref/phmmer_normal_results"):
        """Directory path should map to the directory
        in which the hits are stored
        The hits are stored in the following structure:
        phmmer_normal_results/
            target_dir1/
                query_file1
                query_file2...
            target_dir2/
            ..."""
        self.dir_path: str = dir_path
        self.root: str = os.path.dirname(dir_path)

        self.target_dirs: List[str] = glob.glob(f"{self.dir_path}/*")
        self.target_dirnums = os.listdir(self.dir_path)

    def get_targets_from_dirnum(self, dirnum: str) -> FastaFile:
        """Inputs a directory number for a target directory
        and returns a FastaFile object with the data
        for these target sequences"""
        target_fasta = FastaFile(
            os.path.join(self.root, "split_subset", "targets", f"targets_{dirnum}.fa")
        )
        return target_fasta

    def get_queries_from_dirnum(self, dirnum: str) -> FastaFile:
        """Inputs a directory number for a query directory
        and returns a FastaFile object with the data
        for these query sequences"""
        query_fasta = FastaFile(
            os.path.join(self.root, "split_subset", "queries", f"queries_{dirnum}.fa")
        )
        return query_fasta

    def parse_hits_file(self, hits_file: str, filtered_targets=None) -> Tuple[dict, np.array]:
        """parses a HMMER hits file
        Input: the path to hits file
        Returns: a dictionary of structure {query: {target:data}}
        and a numpy array of just the data"""
        hmmer_hits_file = open(hits_file, "r", encoding="utf8")
        hits = hmmer_hits_file.readlines()
        data_dict = {}
        for row in hits:
            if row[0] == "#":
                continue
            row_info = " ".join(row.split()).split(" ")
            if len(row_info) < 10:
                print(f"Found an oddly formatted row: {row}")
                continue
            target_name = row_info[0]
            assert "UniRef90" in target_name
            if filtered_targets is not None and target_name not in filtered_targets:
                continue

            query_name = row_info[2]
            assert "UniRef90" in query_name
            (e_value_full, score_full, bias_full, e_value_best, score_best, bias_best,) = (
                row_info[4],
                row_info[5],
                row_info[6],
                row_info[7],
                row_info[8],
                row_info[9],
            )
            data = np.array(
                [
                    e_value_full,
                    score_full,
                    bias_full,
                    e_value_best,
                    score_best,
                    bias_best,
                ]
            ).astype("float64")

            if query_name in data_dict.keys():
                data_dict[query_name][target_name] = data
            else:
                data_dict[query_name] = {}
                data_dict[query_name][target_name] = data

        return data_dict

    def get_hits(
        self, dir: str, target_num, query_num=None, filtered_targets=None
    ) -> Tuple[dict, np.array]:
        """
        args:
            target_dir: the directory where the targets are stored
            query_num: the query number (marking a directory)
            to get the hits for
        returns:
            target_query_hits: of format {query:{target:data}}
            hits_array: np array of just the data
        """
        hits_file = os.listdir(f"{dir}/{query_num}/{target_num}")[0]

        hits_dict = self.parse_hits_file(
            f"{dir}/{query_num}/{target_num}/{hits_file}", filtered_targets=filtered_targets
        )

        return hits_dict
