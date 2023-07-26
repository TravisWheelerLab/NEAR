"""Evaluator base class for any evaluator using 
Uniref database"""

import logging
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from src.evaluators.contrastive_functional import search

from src.evaluators import Evaluator
from src.utils import encode_string_sequence
import pickle

logger = logging.getLogger("evaluate")
COLORS = ["r", "c", "g", "k"]
from ctypes import POINTER, c_char


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
        select_random_aminos=False,
        minimum_seq_length=256,
        max_seq_length=512,
        evalue_threshold=1,
        add_random_sequence=False,
        distance_threshold=0,
        normalize_embeddings=False,
        index_string="Flat",
        index_device="cpu",
        output_path="",
        num_threads=16,
        omp_num_threads=4,
        nprobe=100,
        target_embeddings_path=None,
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
            alpha: int
                a parameter of the Faiss index
        """

        self.query_seqs: dict = query_seqs
        self.target_seqs: dict = target_seqs
        self.max_hmmer_hits: dict = hmmer_hits_max
        self.nprobe = nprobe

        self.encoding_func = (
            encode_string_sequence if encoding_func is None else encoding_func
        )
        self.add_random_sequence: bool = add_random_sequence
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
        self.omp_num_threads = omp_num_threads
        self.target_embeddings_path = target_embeddings_path

        print(f"Target embeddings path: {target_embeddings_path}")
        if self.normalize_embeddings:
            logger.info("Using comparison function >= threshold for filtration.")
            self.comp_func = np.greater_equal
        else:
            logger.info("Using comparison function <= threshold for filtration.")
            self.comp_func = np.less_equal

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
                embed = self.compute_embedding(sequence, model_class)
                # return: seq_lenxembed_dim shape
                embeddings.append(
                    torch.nn.functional.normalize(embed, dim=-1).to("cpu")
                )
                lengths.append(length)
            else:
                num_removed += 1
                filtered_names.remove(name)
                logger.debug(
                    f"Removing sequence {sequence} with length {len(sequence)}"
                )
        logger.info(f"removed {num_removed} sequences. ")
        return filtered_names, embeddings, lengths

    @torch.no_grad()
    def _calc_embeddings(
        self,
        sequence_data: dict,
        model_class,
        apply_random_sequence: bool,
        max_seq_length=512,
    ) -> Tuple[List[str], List[torch.Tensor], List[str]]:
        """Calculates the embeddings for the sequences by
        calling the model forward function. Filters the sequences by max/min
        sequence length and returns the filtered sequences/names and embeddings

        Returns [names], [sequences], [embeddings]"""

        names = list(sequence_data.keys())
        sequences = list(sequence_data.values())

        logger.info("Filtering sequences by length...")
        filtered_names, embeddings, lengths = self.filter_sequences_by_length(
            model_class,
            names,
            sequences,
            max_seq_length,
        )

        assert len(filtered_names) == len(embeddings)

        return filtered_names, embeddings, lengths

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
        print(self.target_embeddings_path)
        print(os.path.exists(self.target_embeddings_path))
        if os.path.exists(self.target_embeddings_path):
            print("Loading saved target embeddings")
            target_embeddings = torch.load(self.target_embeddings_path)

            with open(f"target_names.txt", "r") as file_handle:
                target_names = [t.strip("\n") for t in file_handle.readlines()]

            with open(f"target_lengths.txt", "r") as file_handle:
                target_lengths = [int(t.strip("\n")) for t in file_handle.readlines()]

        else:
            print("Embedding targets...")
            target_names, target_embeddings, target_lengths = self._calc_embeddings(
                sequence_data=self.target_seqs,
                model_class=model_class,
                apply_random_sequence=False,
                max_seq_length=self.max_seq_length,
            )

            print(f"Saving target embeddings to: {self.target_embeddings_path}")

            torch.save(target_embeddings, self.target_embeddings_path)
            with open(f"{self.target_embeddings_path}_names.pickle", "wb") as handle:
                pickle.dump(target_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f"{self.target_embeddings_path}_lengths.pickle", "wb") as handle:
                pickle.dump(target_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Embedding queries...")
        query_names, query_embeddings, _ = self._calc_embeddings(
            sequence_data=self.query_seqs,
            model_class=model_class,
            apply_random_sequence=self.add_random_sequence,
            max_seq_length=self.max_seq_length,
        )
        del self.query_seqs

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

    @torch.no_grad()
    def filter(self, queries, query_names, write_results=False):
        """Filters our hits based on
        distance to the query in the Faiiss
        cluster space"""

        logger.info("Beginning search.")

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        print(self.output_path)
        print(f"Number of queries: {len(queries)}")

        t_begin = time.time()

        total_search_time = 0
        total_filtration_time = 0

        unrolled_names_ptr = self.unrolled_names.ctypes.data_as(POINTER(c_char))

        for i in tqdm.tqdm(range(len(queries))):
            filtered_scores, search_time, filtration_time = search(
                self.index, unrolled_names_ptr, queries[i]
            )
            total_search_time += search_time
            total_filtration_time += filtration_time

            if write_results:
                f = open(f"{self.output_path}/{query_names[i]}.txt", "w")
                f.write("Name     Distance" + "\n")
                for name, distance in filtered_scores.items():
                    f.write(f"{name}     {distance}" + "\n")
                f.close()

        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")

        print(f"Search time: {total_search_time}")
        print(f"Filtration time: {total_filtration_time}")
