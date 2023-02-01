from src.evaluators import Evaluator
from src.evaluators.metrics import plot_roc_curve
from src.data.utils import actmap_pipeline
from src.utils.gen_utils import generate_string_sequence
from typing import List, Tuple
from src.utils import encode_string_sequence
import pdb
import logging
import time
import numpy as np
import torch
import tqdm

logger = logging.getLogger("evaluate")
COLORS = ["r", "c", "g", "k"]


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
        nprobe=1,
        distance_threshold=0,
        normalize_embeddings=False,
        index_string="Flat",
        index_device="cpu",
    ):
        """
        Args:
            query_seqs: dict
                A dictionary of {queryname: sequence, ...} for all query sequences
            target_seqs: dict
                A dictionary of {targetname :sequence, ...} for all target sequences
            hmmer_hits_max: dict
                A dictinary of {queryname: {targetname: hmmerdata}}, ... } for all query and target combinations
            encoding_func: func | None
                The function used to encode the sequences. If None, we use src.utils.encode_string_sequence
            model_device: str
                (cpu or cuda)
            figure_path: str
                where to save the ROC plot
            select_random_aminos: bool
            minimum_seq_length: int
                we will cut all sequences in the search space to have this minimum length
            max_seq_length: int
                we cut all sequences in the search space to have this maximum length
            evalue_threshold: int
                we remove from our search space sequences that don't have hits with e value above this threshold
            add_random_sequence: bool
                if True, some random sequence will be added to the beginning of the query sequences
            nprobe: int
                a parameter of the Faiss index
        """

        # self.query_file = deepcopy(query_file)
        # self.target_file = deepcopy(target_file)

        self.query_seqs: dict = query_seqs
        self.target_seqs: dict = target_seqs
        self.max_hmmer_hits: dict = hmmer_hits_max

        self.encoding_func = (
            encode_string_sequence if encoding_func is None else encoding_func
        )
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
        self.distance_threshold = float(distance_threshold)
        self.evalue_threshold = evalue_threshold

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
        names: List[str],
        sequences: List[str],
        model_class,
        apply_random_sequence: bool,
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Filters the sequences by length thresholding given the
        minimum and maximum length threshold variables"""
        embeddings = []

        filtered_sequences, filtered_names = sequences.copy(), names.copy()

        for name, sequence in tqdm.tqdm(zip(names, sequences)):
            if self.max_seq_length >= len(sequence) >= self.minimum_seq_length:
                if apply_random_sequence:
                    # add 100 aminos on to the beginning
                    random_seq = generate_string_sequence(100)
                    embed = self.compute_embedding(random_seq + sequence, model_class)
                else:
                    embed = self.compute_embedding(sequence, model_class)
                # return: seq_lenxembed_dim shape
                embeddings.append(embed.to("cpu"))
            else:
                filtered_names.remove(name)
                filtered_sequences.remove(sequence)
                logger.debug(f"Removing sequence {sequence} with length {len(sequence)}")
        return filtered_names, filtered_sequences, embeddings

    def filter_hmmer_hits(self, target_names: List[str]):
        """Filters the max hmmer hits dict to include
        only the targets that passed length thresolding
        This is in order to keep our benchmarking consistent."""

        init_len = sum(map(lambda x: len(x), self.max_hmmer_hits.values()))
        filtered_hmmer_hits = {}

        for query in self.max_hmmer_hits:
            query_dict = {}
            for hit in self.max_hmmer_hits[query]:
                if hit in target_names:
                    query_dict[hit] = self.max_hmmer_hits[query][hit]
            filtered_hmmer_hits[query] = query_dict

        self.max_hmmer_hits = filtered_hmmer_hits

        new_len = sum(map(lambda x: len(x), self.max_hmmer_hits.values()))
        logger.info(
            f"Removed {init_len - new_len} entries from the target hit dictionary"
            f" since they didn't pass length thresholding."
        )

    @torch.no_grad()
    def _calc_embeddings(
        self, sequence_data: dict, model_class, apply_random_sequence: bool
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Calculates the embeddings for the sequences by
        calling the model forward function. Filters the sequences by max/min
        sequence length and returns the filtered sequences/names and embeddings

        Returns [names], [sequences], [embeddings]"""

        names = list(sequence_data.keys())
        sequences = list(sequence_data.values())

        logger.info("Filtering sequences by length...")
        filtered_sequences, filtered_names, embeddings = self.filter_sequences_by_length(
            names, sequences, model_class, apply_random_sequence
        )

        assert len(filtered_names) == len(filtered_sequences) == len(embeddings)

        return filtered_names, filtered_sequences, embeddings

    def evaluate(self, model_class, visactmaps: bool = False) -> dict:
        """Evaluation pipeline.

        Calculates embeddings for query and targets
        If visactmaps is true, generates activation map plots given the target embeddings
            (probably want to remove this feature)
        Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
        """
        if hasattr(model_class, "initial_seq_len"):
            self.tile_size = model_class.initial_seq_len

        target_names, _, target_embeddings = self._calc_embeddings(
            sequence_data=self.target_seqs,
            model_class=model_class,
            apply_random_sequence=False,
        )
        del self.target_seqs  # remove from memory

        self.filter_hmmer_hits(target_names)

        if visactmaps:
            actmap_pipeline(target_names, target_embeddings, self.max_hmmer_hits)

        query_names, _, query_embeddings = self._calc_embeddings(
            sequence_data=self.query_seqs,
            model_class=model_class,
            apply_random_sequence=self.add_random_sequence,
        )
        del self.query_seqs

        self._setup_target_and_query_dbs(
            target_embeddings, query_embeddings, target_names, query_names
        )

        model_hits, avg_it, total_t = self.filter(query_embeddings, query_names)

        self.denom = len(query_names) * len(target_names)
        self.num_queries = len(query_names)

        plot_roc_curve(
            model_hits,
            self.max_hmmer_hits,
            self.normalize_embeddings,
            self.distance_threshold,
            self.denom,
            self.figure_path,
            self.comp_func,
            evalue_thresholds=[1e-10],
        )
        pdb.set_trace()
        return model_hits

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

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):
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
        self,
        queries,
        query_names,
    ):
        qdict = dict()

        logger.info("Beginning search.")
        # for query in queries
        t_tot = 0
        t_begin = time.time()

        for i in tqdm.tqdm(range(len(queries))):
            loop_begin = time.time()
            logger.debug(f"{i / (len(queries)):.3f}")

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(queries[i], dim=-1)
            else:
                qval = queries[i]

            filtered_hits = self.search(qval)
            names = np.array([f[0] for f in filtered_hits])
            distances = np.array([f[1] for f in filtered_hits])
            sorted_idx = np.argsort(distances)[::-1]

            names = names[sorted_idx]
            distances = distances[sorted_idx]
            logger.debug(f"len names: {len(names)}")
            names, name_idx = np.unique(names, return_index=True)

            filtered_hits = {}
            for name, distance in zip(names, distances[name_idx]):
                filtered_hits[name] = distance

            logger.debug(f"len unique names: {len(filtered_hits)}")
            qdict[query_names[i]] = filtered_hits
            time_taken = time.time() - loop_begin
            t_tot += time_taken

            logger.debug(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")

        return qdict, loop_time / i, loop_time
