import glob
import os
import pdb

import numpy as np


def dict_of_dicts(keys: list):
    _dict = {}
    for key in keys:
        _dict[key] = {}
    return _dict


class FastaFile:
    def __init__(self, filepath):

        self.filepath = filepath

        if not os.path.exists(filepath) or not filepath.endswith("fa"):
            raise f"The filepath is invalud: {filepath}"

        f = open(filepath, "r")
        data = f.readlines()

        self.data: dict = self.clean_data(data)

        self.uniref_names: list = list(self.data.keys())
        self.sequences: list = list(self.data.values())

    def clean_data(self, data: list) -> dict:
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
    def __init__(self, dir_path="uniref/phmmer_normal_results"):
        self.dir_path = dir_path
        self.root = os.path.dirname(dir_path)

        self.target_dirs = glob.glob(f"{self.dir_path}/*")
        self.target_dirnums = os.listdir(self.dir_path)

    def get_targets_from_dirnum(self, dirnum: str):
        target_fasta = FastaFile(
            os.path.join(self.root, "split_subset", "targets", f"targets_{dirnum}.fa")
        )
        return target_fasta

    def get_queries_from_dirnum(self, dirnum: str):
        query_fasta = FastaFile(
            os.path.join(self.root, "split_subset", "queries", f"queries_{dirnum}.fa")
        )
        return query_fasta

    def parse_hits_file(self, hits_file: str):
        f = open(hits_file, "r")
        hits = f.readlines()
        data_dict = {}
        data_array = []
        for row in hits:
            if row[0] == "#":
                continue
            row_info = " ".join(row.split()).split(" ")
            if len(row_info) < 10:
                print(f"Found an oddly formatted row: {row}")
                continue
            target_name = row_info[0]
            query_name = row_info[2]
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
            data_array.append(data)

            if query_name in data_dict.keys():

                data_dict[query_name][target_name] = data
            else:
                data_dict[query_name] = {}
                data_dict[query_name][target_name] = data

        return data_dict, np.array(data_array)

    def get_hits(self, target_dir: str, query_num=None) -> dict:
        hits_files = glob.glob(target_dir + "/*")
        target_dirnum = target_dir.split("/")[-1]

        if query_num:
            hits_files = [f"{target_dir}/queries_{query_num}.fa.tblout"]

        query_dirnums = [h.split("_")[-1][0] for h in hits_files]

        target_query_hits = {target_dirnum: {}}

        for i, hits_file in enumerate(hits_files):
            query_dirnum = query_dirnums[i]
            hits_dict, hits_array = self.parse_hits_file(hits_file)
            target_query_hits[target_dirnum][query_dirnum] = hits_dict

        return target_query_hits, hits_array
