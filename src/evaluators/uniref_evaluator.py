"""Evaluator base class for any evaluator using 
Uniref database"""

import json
import logging
import os
import pdb
import time
from typing import List, Tuple

import numpy as np
import torch
import tqdm

# from src.evaluators.metrics import plot_roc_curve
from src.evaluators import Evaluator
from src.utils import encode_string_sequence
from src.utils.gen_utils import generate_string_sequence

logger = logging.getLogger("evaluate")
COLORS = ["r", "c", "g", "k"]


def filter_sequences_by_length(
        names: List[str],
        sequences: List[str],
        model_class,
        model_device = 'cpu',
        max_seq_length = 512, 
        minimum_seq_length = 0,
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Filters the sequences by length thresholding given the
        minimum and maximum length threshold variables"""
        embeddings = []
        lengths = []

        filtered_names = names.copy()
        num_removed = 0
        for name, sequence in zip(names, sequences):
            length = len(sequence)
            if max_seq_length >= length >= minimum_seq_length:
                embed = (model_class(encode_string_sequence(sequence).unsqueeze(0).to(model_device))
                        .squeeze()
                        .T)
                # return: seq_lenxembed_dim shape
                embeddings.append(embed.to("cpu"))
                lengths.append(length)
            else:
                num_removed += 1
                filtered_names.remove(name)
                # filtered_sequences.remove(sequence)
        return filtered_names, embeddings, lengths

@torch.no_grad()
def _calc_embeddings(data) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """Calculates the embeddings for the sequences by
    calling the model forward function. Filters the sequences by max/min
    sequence length and returns the filtered sequences/names and embeddings

    Returns [names], [sequences], [embeddings]"""
    
    sequence_data, model_class = data

    names = list(sequence_data.keys())
    sequences = list(sequence_data.values())

    filtered_names, embeddings, lengths = filter_sequences_by_length(
        names, sequences, model_class
    )

    return filtered_names, embeddings, lengths

def evaluate2(query_seqs, model_class, evaluator, target_names, target_embeddings, target_lengths) -> dict:
    """Evaluation pipeline.

    Calculates embeddings for query and targets
    If visactmaps is true, generates activation map plots given the target embeddings
        (probably want to remove this feature)
    Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
    """
    if hasattr(model_class, "initial_seq_len"):
        evaluator.tile_size = model_class.initial_seq_len

    print(f"Found {(len(evaluator.target_seqs))} targets")

    print("Embedding queries...")
    query_names, query_embeddings, _ = evaluator._calc_embeddings(
        sequence_data=query_seqs,
        model_class=model_class,
        apply_random_sequence=evaluator.add_random_sequence,
        max_seq_length=evaluator.max_seq_length,
    )
    del query_seqs

    del evaluator.target_seqs  # remove from memory

    evaluator._setup_targets_for_search(target_embeddings, target_names, target_lengths)

    evaluator.filter(query_embeddings, query_names)

class UniRefEvaluator(Evaluator):
    """The superclass for all evaluators that are based on the UniRef dataset.
    This version of this class takes in dictionaries of sequence data, target data
    and hmmer hits data that come from code in the src/data/ module."""

    def __init__(
        self,
        query_seqs: dict,
        target_seqs: dict,
        hmmer_hits_max: dict,
        encoding_func,
        model_device,
        figure_path,
        select_random_aminos=False,
        minimum_seq_length=256,
        max_seq_length=512,
        evalue_threshold=1,
        add_random_sequence=False,
        nprobe=5,
        distance_threshold=0,
        normalize_embeddings=False,
        index_string="Flat",
        index_device="cpu",
        output_path="",
        num_threads = 16
    ):
        """
        Args:
            query_seqs: dict
                A dictionary of {queryname: sequence, ...} for all query sequences
            target_seqs: dict
                A dictionary of {targetname :sequence, ...} for all target sequences
            hmmer_hits_max: dict
                A dictinary of {queryname: {targetname: hmmerdata}}, ... }
                for all query and target combinations
            encoding_func: func | None
                The function used to encode the sequences.
                If None, we use src.utils.encode_string_sequence
            model_device: str
                (cpu or cuda)
            figure_path: str
                where to save the ROC plot
            select_random_aminos: bool
            minimum_seq_length: int
                we will cut all sequences in the search space
                to have this minimum length
            max_seq_length: int
                we cut all sequences in the search space
                to have this maximum length
            evalue_threshold: int
                we remove from our search space sequences
                that don't have hits with e value above this threshold
            add_random_sequence: bool
                if True, some random sequence will
                be added to the beginning of the query sequences
            nprobe: int
                a parameter of the Faiss index
        """

        self.query_seqs: dict = query_seqs
        self.target_seqs: dict = target_seqs
        self.max_hmmer_hits: dict = hmmer_hits_max

        self.encoding_func = encode_string_sequence if encoding_func is None else encoding_func
        self.add_random_sequence: bool = add_random_sequence
        self.figure_path: str = figure_path
        self.nprobe: int = nprobe
        self.model_device: str = model_device
        self.minimum_seq_length: int = minimum_seq_length
        self.max_seq_length: int = max_seq_length
        self.normalize_embeddings: bool = normalize_embeddings
        self.select_random_aminos: bool = select_random_aminos
        self.index_string: str = index_string
        self.index_device: str = index_device
        self.denom = None
        self.tile_size = None
        self.num_queries = None
        self.distance_threshold = float(distance_threshold)
        self.evalue_threshold = evalue_threshold
        self.output_path = output_path
        self.num_threads = num_threads

        if self.normalize_embeddings:
            logger.info("Using comparison function >= threshold for filtration.")
            self.comp_func = np.greater_equal
        else:
            logger.info("Using comparison function <= threshold for filtration.")
            self.comp_func = np.less_equal

        # root = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/"
        # self.max_hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")

    def filter_sequences_by_length(
        self,
        model_class,
        names: List[str],
        sequences: List[str],
        max_seq_length=512,
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Filters the sequences by length thresholding given the
        minimum and maximum length threshold variables"""
        embeddings = []
        lengths = []

        filtered_names = names.copy()
        num_removed = 0
        for name, sequence in tqdm.tqdm(zip(names, sequences)):
            length = len(sequence)
            if max_seq_length >= length >= self.minimum_seq_length:
                # if apply_random_sequence:
                # add 100 aminos on to the beginning
                # random_seq = generate_string_sequence(100)
                # embed = self.compute_embedding(
                #     random_seq + sequence, model_class
                # )
                # else:
                embed = self.compute_embedding(sequence, model_class)
                # return: seq_lenxembed_dim shape
                embeddings.append(embed.to("cpu"))
                lengths.append(length)
            else:
                num_removed += 1
                filtered_names.remove(name)
                # filtered_sequences.remove(sequence)
                logger.debug(f"Removing sequence {sequence} with length {len(sequence)}")
        logger.info(f"removed {num_removed} sequences. ")
        return filtered_names, embeddings, lengths

    @torch.no_grad()
    def _calc_embeddings(
        self, sequence_data: dict, model_class, apply_random_sequence: bool, max_seq_length=512,
    ) -> Tuple[List[str], List[torch.Tensor],List[str]]:
        """Calculates the embeddings for the sequences by
        calling the model forward function. Filters the sequences by max/min
        sequence length and returns the filtered sequences/names and embeddings

        Returns [names], [sequences], [embeddings]"""

        names = list(sequence_data.keys())
        sequences = list(sequence_data.values())

        logger.info("Filtering sequences by length...")
        filtered_names, embeddings, lengths = self.filter_sequences_by_length(model_class,
            names, sequences, max_seq_length,
        )

        assert len(filtered_names) == len(embeddings)

        return filtered_names, embeddings, lengths

    def evaluate_multiprocessing(self, query_names, query_embeddings, target_names, target_embeddings, target_lengths) -> dict:
        """Evaluation pipeline.

        runs Faiss clustering and filtering and returns a dictionary of the model's hits.
        """

        print(f"Found {(len(self.target_seqs))} targets")

        self._setup_targets_for_search(target_embeddings, target_names, target_lengths)

        self.filter(query_embeddings, query_names)

    def evaluate2(self,query_seqs, model_class, target_names, target_embeddings, target_lengths) -> dict:
        """Evaluation pipeline.

        Calculates embeddings for query and targets
        If visactmaps is true, generates activation map plots given the target embeddings
            (probably want to remove this feature)
        Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
        """
        if hasattr(model_class, "initial_seq_len"):
            self.tile_size = model_class.initial_seq_len

        print(f"Found {(len(self.target_seqs))} targets")

        print("Embedding queries...")
        query_names, query_embeddings, _ = self._calc_embeddings(
            sequence_data=query_seqs,
            model_class=model_class,
            apply_random_sequence=self.add_random_sequence,
            max_seq_length=self.max_seq_length,
        )
        del query_seqs

        del self.target_seqs  # remove from memory

        self._setup_targets_for_search(target_embeddings, target_names, target_lengths)

        self.filter(query_embeddings, query_names)


    def evaluate(self, model_class) -> dict:
        """Evaluation pipeline.

        Calculates embeddings for query and targets
        If visactmaps is true, generates activation map plots given the target embeddings
            (probably want to remove this feature)
        Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
        """
        if hasattr(model_class, "initial_seq_len"):
            self.tile_size = model_class.initial_seq_len

        print(f"Found {(len(self.target_seqs))} targets")

        print("Embedding queries...")
        query_names, query_embeddings, _ = self._calc_embeddings(
            sequence_data=self.query_seqs,
            model_class=model_class,
            apply_random_sequence=self.add_random_sequence,
            max_seq_length=self.max_seq_length,
        )
        del self.query_seqs

        print("Embedding targets...")
        target_names, target_embeddings, target_lengths = self._calc_embeddings(
            sequence_data=self.target_seqs,
            model_class=model_class,
            apply_random_sequence=False,
            max_seq_length=self.max_seq_length,
        )


        del self.target_seqs  # remove from memory

        self._setup_targets_for_search(target_embeddings, target_names, target_lengths)

        self.filter(query_embeddings, query_names)

    def compute_embedding(self, sequence, model_class):
        """This gets called from within the calculate embedding function
        and will be specific for the model class

        :param sequence:
        :type sequence:
        :param model_class:
        :type model_class:
        :return: An embedding, seq_lenxembed dim.
        :rtype:
        """
        raise NotImplementedError()

    def _setup_targets_for_faiss(self, target_embeddings, target_names):
        """Base method"""
        raise NotImplementedError()

    def search(self, query_embedding):
        """
        :param query_embedding: seq_lenxembedding dimension query to search against the target database.
        :type query_embedding: torch.Tensor()
        :return:
        :rtype:
        """
        raise NotImplementedError()

    @torch.no_grad()
    def filter(
        self, queries, query_names
    ):
        """Filters our hits based on
        distance to the query in the Faiiss
        cluster space"""

        logger.info("Beginning search.")

        t_begin = time.time()

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        print(self.output_path)
        print(f"Number of queries: {len(queries)}")
        # search_time, filter_time, aggregate_time = 0, 0, 0
        #all_filtered_scores = {}
        for i in tqdm.tqdm(range(len(queries))):

            f = open(f"{self.output_path}/{query_names[i]}.txt", "w")
            f.write("Name     Distance" + "\n")

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(queries[i], dim=-1)
            else:
                qval = queries[i]

            filtered_scores = self.search(qval)#, search_time, filter_time, aggregate_time)
            #all_filtered_scores[i] = filtered_scores
            for name, distance in filtered_scores.items():
                f.write(f"{name}     {distance}" + "\n")
            f.close()


        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")
        # logger.info(f"Search time: {search_time}.")
        # logger.info(f"Filter time: {filter_time}.")
        # logger.info(f"Aggregate time: {aggregate_time}.")

        # print("Writing results to file...")
        # for query_idx, score in all_filtered_scores.items():
        #     f = open(f"{self.output_path}/{query_names[query_idx]}.txt", "w")
        #     f.write("Name     Distance" + "\n")
        #     for name, distance in score.items():
        #         f.write(f"{name}     {distance}" + "\n")
        #     f.close()
