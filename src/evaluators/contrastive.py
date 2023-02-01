import logging
import torch
import faiss
from src.utils import create_faiss_index, encode_string_sequence
import numpy as np
import os
import pdb
from typing import List, Tuple
import time
import tqdm
from src.evaluators import Evaluator
from src.utils.gen_utils import generate_string_sequence
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import cv2

logger = logging.getLogger("evaluate")


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

    @torch.no_grad()
    def _calc_embeddings(
        self, sequence_data: dict, model_class, apply_random_sequence: bool
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Calculates the embeddings for the sequences by
        calling the model forward function

        Does so for one sequence at a time

        Returns [names], [sequences], [embeddings]"""

        # names, sequences = fasta_from_file(fasta_file=fasta_or_pickle_file)
        # names = list(map(lambda x: x.split(" ")[0], names))

        names = list(sequence_data.keys())
        sequences = list(sequence_data.values())

        embeddings = []
        for j, sequence in enumerate(tqdm.tqdm(sequences)):
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
                logger.debug(f"Removing sequence {sequence} with length {len(sequence)}")
                names.remove(names[j])
                sequences.remove(sequences[j])

        assert len(names) == len(sequences) == len(embeddings)

        return names, sequences, embeddings

    def evaluate(self, model_class, visactmaps: bool = False) -> dict:
        """Evaluation pipeline.

        Calculates embeddings for query and targets
        If visactmaps is true, generates activation map plots given the target embeddings
            (probably want to remove this feature)
        Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
        """
        if hasattr(model_class, "initial_seq_len"):
            self.tile_size = model_class.initial_seq_len

        (target_names, target_sequences, target_embeddings,) = self._calc_embeddings(
            sequence_data=self.target_seqs,
            model_class=model_class,
            apply_random_sequence=False,
        )
        del self.target_seqs  # remove from memory

        if visactmaps:
            self.actmap_pipeline(target_names, target_embeddings)

        query_names, query_sequences, query_embeddings = self._calc_embeddings(
            sequence_data=self.query_seqs,
            model_class=model_class,
            apply_random_sequence=self.add_random_sequence,
        )
        del self.query_seqs

        # now, remove elements from hmmer --max if the target name is not in
        # target names.
        # TODO: this should be its own function
        init_len = sum(map(lambda x: len(x), self.max_hmmer_hits.values()))

        for query in self.max_hmmer_hits:
            for hit in self.max_hmmer_hits[query]:
                if hit not in target_names:
                    del self.max_hmmer_hits[query][hit]
                    print(f"{hit} from hmmer hits is not in target name")

        new_len = sum(map(lambda x: len(x), self.max_hmmer_hits.values()))
        logger.info(
            f"Removed {init_len - new_len} entries from the target hit dictionary"
            f" since they didn't pass length thresholding."
        )

        self._setup_target_and_query_dbs(
            target_embeddings, query_embeddings, target_names, query_names
        )

        hits, avg_it, total_t = self.filter(query_embeddings, query_names)
        our_hits = defaultdict(dict)
        # stop writing and stuff to file.
        # just process the dang dictionary.
        for query in hits:
            for hit in hits[query]:
                our_hits[query][hit[0]] = hit[1]

        self.denom = len(query_names) * len(target_names)
        self.num_queries = len(query_names)

        self._roc_plot(our_hits, self.max_hmmer_hits)

        return our_hits

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

    def _roc_plot(self, our_hits, max_hits):

        if self.normalize_embeddings:
            distances = np.linspace(self.distance_threshold, 0.999, num=10)
        else:
            distances = np.linspace(0.001, self.distance_threshold, num=10)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"{os.path.splitext(os.path.basename(self.figure_path))[0]}")

        # for color, evalue_threshold in zip(["r", "c", "g", "k"], [1e-10, 1e-1, 1, 10]):
        for color, evalue_threshold in zip(["c"], [1e-1]):

            filtrations = []
            recalls = []
            for threshold in tqdm.tqdm(distances):
                recall, total_hits = recall_and_filtration(
                    our_hits,
                    max_hits,
                    threshold,
                    self.comp_func,
                    evalue_threshold,
                )

                filtration = 100 * (1.0 - (total_hits / self.denom))
                filtrations.append(filtration)
                recalls.append(recall)
                print(f"{recall:.3f}, {filtration:.3f}, {threshold:.3f}")

            ax.scatter(filtrations, recalls, c=color, marker="o")
            ax.plot(filtrations, recalls, f"{color}--", linewidth=2)

        ax.plot([0, 100], [100, 0], "k--", linewidth=2)
        ax.set_ylim([-1, 101])
        ax.set_xlim([-1, 101])
        ax.set_xlabel("filtration")
        ax.set_ylabel("recall")
        plt.savefig(f"{self.figure_path}", bbox_inches="tight")
        plt.close()

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

            filtered_hits = []
            for name, distance in zip(names, distances[name_idx]):
                filtered_hits.append((name, distance))

            logger.debug(f"len unique names: {len(filtered_hits)}")
            qdict[query_names[i]] = filtered_hits
            time_taken = time.time() - loop_begin
            t_tot += time_taken

            logger.debug(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")

        return qdict, loop_time / i, loop_time


class ContrastiveEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_embedding(self, sequence: str, model_class) -> torch.Tensor:
        return (
            model_class(
                encode_string_sequence(sequence).unsqueeze(0).to(self.model_device)
            )
            .squeeze()
            .T
        )

    def get_actmaps(self, embeddings: list, function="sum", p=2, show=True, title=None):
        num_embeds = 500
        if len(embeddings) > num_embeds:
            embeddings = random.sample(embeddings, num_embeds)

        else:
            print(f"Have {len(embeddings)} embeddings ")
            num_embeds = len(embeddings)

        seq_dim = np.max([len(s) for s in embeddings])

        all_actmaps = np.zeros((num_embeds, seq_dim, 1, 3))

        # embeddings shape (num_embeddings, seq_len, 256)
        for idx in range(num_embeds):
            sample = embeddings[idx]
            if function == "sum":
                outputs = (sample**p).sum(1)
            elif function == "max":
                outputs = (sample**p).max(1)
            try:
                outputs_n = outputs.reshape(1, outputs.shape[0])
                outputs_n = outputs_n / outputs_n.sum(axis=1)
            except AttributeError:
                pdb.set_trace()
            am = outputs_n[0, ...]
            am = am.numpy()
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
            seqlen = am.shape[0]
            all_actmaps[idx, :seqlen, :, :] = am
        if show:
            figname1 = f"activationmaps_{np.random.randint(0,5000)}.png"

            np.save(figname1, all_actmaps, allow_pickle=True)

            all_actmaps = all_actmaps.reshape(num_embeds, seq_dim, 3)
            plt.imshow(all_actmaps, aspect=4)
            if title:
                plt.title(title)

            print(f"Saving figure as {figname1}")
            plt.savefig(figname1)
            # sliced_actmaps = all_actmaps[:,:512, :]
            # plt.clf()
            # plt.imshow(sliced_actmaps, aspect = 3)
            # if title:
            #     plt.title(title + ', sliced to 512')

            # figname2 = f'activationmaps_sliced_{np.random.randint(0,5000)}.png'
            # print(f'Saving figure as {figname2}')
            # plt.savefig(figname2)
        return all_actmaps

    def actmap_pipeline(self, names: List[str], embeddings: List[torch.Tensor]):

        pos_samples, neg_samples = get_subsets(self.max_hmmer_hits)  # names of sequences

        similar_embeddings = []

        diff_embeddings = []

        for seq_name in pos_samples:
            try:
                idx = names.index(seq_name)
                emb = embeddings[idx]

                similar_embeddings.append(emb)
            except ValueError as e:
                print(e)

        print(f"Got {len(similar_embeddings)}  similar embeddings")

        for seq_name in neg_samples:
            idx = names.index(seq_name)
            emb = embeddings[idx]
            diff_embeddings.append(emb)

        print(f"Got {len(diff_embeddings)}  dissimilar embeddings")

        self.get_actmaps(embeddings, title="Amino activation maps")
        self.get_actmaps(similar_embeddings, title="Similar embedding activation map")
        self.get_actmaps(diff_embeddings, title="Different embedding activation map")

    def _setup_target_and_query_dbs(
        self,
        target_embeddings: List[torch.Tensor],
        query_embeddings: List[torch.Tensor],
        target_names: List[str],
        query_names: List[str],
    ):

        # no queries?
        lengths: List[int] = list(map(lambda s: s.shape[0], target_embeddings))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(
            zip(lengths, target_names, target_embeddings)
        ):
            # sample every N amino.
            aminos = torch.cat([target[j].unsqueeze(0) for j in range(length)], dim=0)

            self.unrolled_names.extend(
                [name] * length
            )  # record keeping (num targets x amino per target) - every given amino in a sequence has the same name
            unrolled_targets.append(aminos)

        unrolled_targets = torch.cat(
            unrolled_targets, dim=0
        )  # 128 x (num targets x amino per target)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index: faiss.Index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,
            nprobe=self.nprobe,
            device=self.index_device,
        )

        logger.info("Adding targets to index.")
        if self.index_device == "cpu":
            self.index.add(unrolled_targets.to("cpu"))
        else:
            self.index.add(unrolled_targets)

        self.unrolled_names = np.asarray(self.unrolled_names)

        faiss.omp_set_num_threads(int(os.environ.get("NUM_THREADS")))

    def search(self, query_embedding):
        filtered_list = []

        D, I = self.index.search(query_embedding.contiguous(), k=1000)  # top 2048 hits
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))
        # TODO: use a torch.cat instead
        # see line 313 in uniref_evaluator
        return filtered_list
