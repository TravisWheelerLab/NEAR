""" Classes to interact with Fasta Files and Hmmer Hits files"""


import glob
import os
from typing import List, Tuple
import tqdm
import numpy as np


class FastaFile:
    """Class representing a FASTA file.
    Class variables include lists of names and sequences,
    and a dictionary mapping names to sequences."""

    def __init__(self, filepath: str):
        """initialize fasta file from fasta path"""
        self.filepath: str = filepath

        if not os.path.exists(filepath) or not filepath.endswith("fa"):
            raise f"The filepath is invalud: {filepath}"

        with open(filepath, "r", encoding="utf8") as fastafile:
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

    def parse_hits_file(self, hits_file: SyntaxError) -> Tuple[dict, np.array]:
        """parses a HMMER hits file
        Input: the path to hits file
        Returns: a dictionary of structure {query: {target:data}}
        and a numpy array of just the data"""
        data_dict = {}
        hmmer_hits_file = open(hits_file, "r")
        for row in hmmer_hits_file:
            if row[0] == "#":
                continue
            row_info = " ".join(row.split()).split(" ")
            if len(row_info) < 10:
                print(f"Found an oddly formatted row: {row}")
                continue
            target_name = row_info[0]
            assert "UniRef90" in target_name

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

            if query_name in data_dict:
                data_dict[query_name].update({target_name: data})
            else:
                data_dict[query_name] = {}
                data_dict[query_name].update({target_name: data})
            # if idx == 275:
            #     print(data_dict['UniRef90_UPI001F15798D'])
        hmmer_hits_file.close()

        return data_dict

    def get_hits(self, directory: str) -> Tuple[dict, np.array]:

        assert os.path.exists(
            f"{directory}/hits.tblout"
        ), f"No HMMER hits at {directory}/hits.tblout"

        hits_dict = self.parse_hits_file(f"{directory}/hits.tblout")

        return hits_dict
