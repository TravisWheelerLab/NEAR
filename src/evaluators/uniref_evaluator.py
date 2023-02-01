import logging
import os
import pdb
import re
import time
from collections import defaultdict
from copy import deepcopy

import faiss
import matplotlib.pyplot as plt
import numpy as np

# import scann
import torch
import tqdm
from sequence_models.pretrained import load_model_and_alphabet
from torchaudio.transforms import MelSpectrogram

from src.evaluators import Evaluator
from src.utils import (
    amino_char_to_index,
    create_faiss_index,
    encode_string_sequence,
    fasta_from_file,
)
from src.utils.gen_utils import generate_string_sequence

logger = logging.getLogger("evaluate")

# load hits from the hmmer file.
def get_hmmer_hits(file_path):
    queries = defaultdict(dict)
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == "#":
                continue
            line = re.split(" +", line)
            target = line[0]
            query = line[2]
            e_value = float(line[4])
            score = float(line[6])

            queries[query][target] = (e_value, score)

        return queries


def recall_and_filtration(
    our_hits, hmmer_hits, distance_threshold, comp_func, evalue_threshold
):
    match_count = 0
    our_total_hits = 0
    hmmer_hits_for_our_queries = 0
    # since we sometimes don't have
    # all queries, iterate over the DB in this fashion.
    for query in our_hits:
        if query not in hmmer_hits:
            # we've set an e-value threshold, meaning
            # that this query was never picked up
            pdb.set_trace()

        matches = hmmer_hits[query]
        filtered = {}
        for match, evalue in matches.items():
            evalue = float(evalue[0])
            if evalue <= evalue_threshold:
                filtered[match] = evalue

        true_matches = filtered
        hmmer_hits_for_our_queries += len(true_matches)
        our_matches = our_hits[query]
        for match in our_matches:
            if comp_func(our_matches[match], distance_threshold):
                if match in true_matches:
                    # count the matches for each query.
                    # pdb.set_trace()
                    match_count += 1
                our_total_hits += 1

    # total hmmer hits
    denom = hmmer_hits_for_our_queries
    return 100 * (match_count / denom), our_total_hits


class UniRefEvaluator(Evaluator):
    def __init__(
        self,
        query_file,
        target_file,
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

        self.query_file = deepcopy(query_file)
        self.target_file = deepcopy(target_file)

        self.encoding_func = (
            encode_string_sequence if encoding_func is None else encoding_func
        )
        self.add_random_sequence = add_random_sequence
        self.figure_path = figure_path
        self.nprobe = nprobe
        self.model_device = model_device
        self.minimum_seq_length = minimum_seq_length
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.select_random_aminos = select_random_aminos
        self.index_string = index_string
        self.index_device = index_device
        self.denom = None
        self.distance_threshold = float(distance_threshold)
        self.evalue_threshold = evalue_threshold

        if self.normalize_embeddings:
            logger.info("Using comparison function >= threshold for filtration.")
            self.comp_func = np.greater_equal
        else:
            logger.info("Using comparison function <= threshold for filtration.")
            self.comp_func = np.less_equal

        root = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/"
        self.max_hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")

    @torch.no_grad()
    def _calc_embeddings(
        self, fasta_or_pickle_file: str, model_class, apply_random_sequence: bool
    ):
        """Returns [names], [sequences], [embeddings]"""

        names, sequences = fasta_from_file(fasta_file=fasta_or_pickle_file)
        names = list(map(lambda x: x.split(" ")[0], names))

        idx_to_keep = []

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
                idx_to_keep.append(j)
            else:
                logger.debug(f"Removing sequence {sequence} with length {len(sequence)}")

        _names = []
        _sequences = []
        for idx in idx_to_keep:
            _names.append(names[idx])
            _sequences.append(sequences[idx])

        names = _names
        sequences = _sequences
        assert len(names) == len(sequences)
        assert len(names) == len(embeddings)

        return names, sequences, embeddings

    def evaluate(self, model_class):
        if hasattr(model_class, "initial_seq_len"):
            self.tile_size = model_class.initial_seq_len

        (target_names, target_sequences, target_embeddings,) = self._calc_embeddings(
            self.target_file,
            model_class=model_class,
            apply_random_sequence=False,
        )
        query_names, query_sequences, query_embeddings = self._calc_embeddings(
            self.query_file,
            model_class=model_class,
            apply_random_sequence=self.add_random_sequence,
        )
        # now, remove elements from hmmer --max if the target name is not in
        # target names.
        init_len = sum(map(lambda x: len(x), self.max_hmmer_hits.values()))
        new_max = {}

        for query in self.max_hmmer_hits:
            new_dict = {}
            for hit in self.max_hmmer_hits[query]:
                if hit in target_names:
                    new_dict[hit] = self.max_hmmer_hits[query][hit]
            new_max[query] = new_dict

        self.max_hmmer_hits = new_max
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

    def compute_embedding(self, sequence, model_class):
        """
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


class ContrastiveEvaluatorFile(UniRefEvaluator):
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

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):
        # no queries?
        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(zip(lengths, target_names, targets)):
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


class UniRefVAEEvaluator(UniRefEvaluator):
    def __init__(self, overwrite, n_vae_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overwrite = overwrite
        self.n_vae_samples = n_vae_samples

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):
        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(zip(lengths, target_names, targets)):
            # sample every N amino.
            aminos = torch.cat([target[j].unsqueeze(0) for j in range(length)], dim=0)
            self.unrolled_names.extend([name] * length)
            unrolled_targets.append(aminos)

        unrolled_targets = torch.cat(unrolled_targets, dim=0)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
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

        D, I = self.index.search(query_embedding.contiguous(), k=1000)
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))

        return filtered_list

    def compute_embedding(self, sequence, model_class):
        slices = []
        for _ in range(self.n_vae_samples):
            for i in range(
                0,
                len(sequence) - model_class.initial_seq_len,
                model_class.initial_seq_len,
            ):
                embedding, _ = model_class(
                    encode_string_sequence(sequence[i : i + model_class.initial_seq_len])
                    .to(self.model_device)
                    .unsqueeze(0)
                )
                slices.append(embedding)

            if len(sequence) % model_class.initial_seq_len != 0:
                embedding, _ = model_class(
                    encode_string_sequence(sequence[-model_class.initial_seq_len :])
                    .to(self.model_device)
                    .unsqueeze(0)
                )
                slices.append(embedding)

            if len(sequence) == model_class.initial_seq_len:
                embedding, _ = model_class(
                    encode_string_sequence(sequence).to(self.model_device).unsqueeze(0)
                )
                slices.append(embedding)

        return torch.cat(slices, dim=-1).squeeze()


class UniRefUngappedVAEEvaluator(UniRefVAEEvaluator):
    """
    Data for true hits is different.
    """

    def __init__(self, hit_file, *args, **kwargs):
        super().__init__(*args, **kwargs)

        queries = defaultdict(dict)
        with open(hit_file, "r") as src:
            for line in src.readlines():
                query, target = line.strip().split()
                queries[query][target] = (0.0, 0.0)

        self.max_hmmer_hits = queries


class UniRefTiledVAEEvaluator(UniRefVAEEvaluator):
    def __init__(self, tile_size, tile_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.tile_step = tile_step

    @torch.no_grad()
    def compute_embedding(self, sequence, model_class):
        slices = []
        encoded = encode_string_sequence(sequence).to(self.model_device).unsqueeze(0)
        for _ in range(self.n_vae_samples):
            if len(sequence) == model_class.initial_seq_len:
                encoding, _ = model_class(encoded)
                encoding = encoding.reshape(1, -1)
                slices.append(encoding)

            else:
                # no padding, just overlapping evaluation steps.
                # i think this makes more sense than padding with stuff.
                # just pad the end with reflected sequence.
                # i need to get how much to pad
                # this will be len(sequence) - self.tile_size * (len(sequence)//self.tile_size)
                for begin_idx in range(0, len(sequence), self.tile_step):
                    if begin_idx + self.tile_size >= len(sequence):
                        # we can't go over the end, so break
                        break
                    encoded_slice = encoded[:, :, begin_idx : begin_idx + self.tile_size]
                    encoding, _ = model_class(encoded_slice)
                    encoding = encoding.reshape(1, -1)
                    slices.append(encoding)

                # now take care of the end
                # this might result in duplication but since we're VAE'ing it
                # i think it's OK
                encoded_slice = encoded[:, :, -model_class.initial_seq_len :]
                encoding, _ = model_class(encoded_slice)
                slices.append(encoding)
        slices = [i.reshape(1, -1) for i in slices]
        return torch.cat(slices, dim=0)


class UniRefRandomUngappedVAEEvaluator(UniRefTiledVAEEvaluator):
    """
    Only inject random sequences into the target database.
    """

    def __init__(self, hit_file, random_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # embed my target sequences within random sequence.

        self.random_length = random_length

        queries = defaultdict(dict)
        with open(hit_file, "r") as src:
            for line in src.readlines():
                query, target = line.strip().split()
                queries[query][target] = (0.0, 0.0)

        self.max_hmmer_hits = queries

    @torch.no_grad()
    def _calc_embeddings(self, fasta_file, model_class):

        names, sequences = fasta_from_file(fasta_file=fasta_file)
        names = list(map(lambda x: x.split(" ")[0], names))
        is_target_db = "target" in os.path.basename(fasta_file)

        if is_target_db:
            logger.info(
                "Only injecting random sequence onto start/end of target sequences."
            )

        idx_to_keep = []

        embeddings = []
        for j, sequence in enumerate(tqdm.tqdm(sequences)):
            if len(sequence) >= model_class.initial_seq_len:
                if is_target_db and self.random_length != 0:
                    generated = generate_string_sequence(2 * self.random_length)
                    init_len = len(sequence)
                    sequence = (
                        generated[: self.random_length]
                        + sequence
                        + generated[self.random_length :]
                    )
                    logger.debug(
                        f"Added {len(sequence) - init_len} characters onto sequence."
                    )
                    # add bits on to the beginning and end of sequence.
                embed = self.compute_embedding(sequence, model_class)
                embeddings.append(embed.to("cpu"))
                idx_to_keep.append(j)
            else:
                logger.debug(f"Removing sequence {sequence} with length {len(sequence)}")

        _names = []
        _sequences = []
        for idx in idx_to_keep:
            _names.append(names[idx])
            _sequences.append(sequences[idx])

        names = _names
        sequences = _sequences
        assert len(names) == len(sequences)
        assert len(names) == len(embeddings)

        return names, sequences, embeddings


class UniRefMeanPoolEvaluator(UniRefEvaluator):
    def __init__(self, hmmer_hit_file, *args, **kwargs):
        super().__init__(*args, **kwargs)

        queries = defaultdict(dict)
        if hmmer_hit_file is not None:
            with open(hmmer_hit_file, "r") as src:
                for line in src.readlines():
                    query, target = line.strip().split()
                    queries[query][target] = (0.0, 0.0)
            self.max_hmmer_hits = queries

    def compute_embedding(self, sequence, model_class):
        encoded = encode_string_sequence(sequence).to(self.model_device).unsqueeze(0)
        embedding = model_class(encoded).squeeze().mean(dim=1).unsqueeze(0)
        return embedding

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):
        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(zip(lengths, target_names, targets)):
            # sample every N amino.
            self.unrolled_names.extend([name] * length)
            unrolled_targets.append(target)

        unrolled_targets = torch.cat(unrolled_targets, dim=0)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
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

        D, I = self.index.search(query_embedding.contiguous(), k=2048)
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))

        return filtered_list


class UniRefTiledMeanPoolEvaluator(UniRefMeanPoolEvaluator):
    def __init__(self, tile_size, tile_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.tile_step = tile_step

    def helper(self, encoded, model_class):
        embedding = model_class(encoded).squeeze().mean(dim=1).unsqueeze(0)
        return embedding

    @torch.no_grad()
    def compute_embedding(self, sequence, model_class):
        slices = []
        encoded = encode_string_sequence(sequence).to(self.model_device).unsqueeze(0)
        if len(sequence) <= self.tile_size:
            encoding = self.helper(encoded, model_class)
            slices.append(encoding)
        else:

            for begin_idx in range(0, len(sequence), self.tile_step):
                if begin_idx + self.tile_size >= len(sequence):
                    # we can't go over the end, so break
                    break

                encoded_slice = encoded[:, :, begin_idx : begin_idx + self.tile_size]
                encoding = self.helper(encoded_slice, model_class)
                slices.append(encoding)
            encoded_slice = encoded[:, :, -self.tile_size :]
            encoding = self.helper(encoded_slice, model_class)
            slices.append(encoding)

        return torch.cat(slices, dim=0)


class UniRefTiledKmerEvaluator(UniRefTiledMeanPoolEvaluator):
    def helper(self, encoded, model_class):
        embedding = model_class(encoded).mean(dim=2)
        embedding = torch.cat(torch.unbind(embedding, dim=-1), dim=-1)
        return embedding


class UniRefSpectrogramEvaluator(UniRefMeanPoolEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mel = MelSpectrogram(win_length=32, hop_length=16, n_mels=50)

    @torch.no_grad()
    def compute_embedding(self, sequence, model_class):
        encoded = torch.as_tensor([amino_char_to_index[c] for c in sequence])
        encoded = self.mel(encoded.float()).unsqueeze(0).to(self.model_device)
        encoded = encoded / torch.max(encoded)
        encoding = model_class(encoded).mean(dim=-1)
        return encoding


class UniRefCARPEvaluator(UniRefEvaluator):
    def __init__(self, hmmer_hit_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, self.collater = load_model_and_alphabet("carp_600k")

        if hmmer_hit_file is not None:
            queries = defaultdict(dict)
            with open(hmmer_hit_file, "r") as src:
                for line in src.readlines():
                    query, target = line.strip().split()
                    queries[query][target] = (0.0, 0.0)
            self.max_hmmer_hits = queries

    def compute_embedding(self, sequence, model_class):
        encoded = self.collater([[sequence]])[0].to(self.model_device)
        embedding = model_class(encoded)["representations"][32]
        return embedding.mean(dim=1)

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):
        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(zip(lengths, target_names, targets)):
            # sample every N amino.
            aminos = torch.cat([target[j].unsqueeze(0) for j in range(length)], dim=0)
            self.unrolled_names.extend([name] * length)
            unrolled_targets.append(aminos)

        unrolled_targets = torch.cat(unrolled_targets, dim=0)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
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

        D, I = self.index.search(query_embedding.contiguous(), k=2048)
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))

        return filtered_list


class UniRefESMEvaluator(UniRefEvaluator):
    def __init__(self, hmmer_hit_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import esm

        _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()

        if hmmer_hit_file is not None:
            queries = defaultdict(dict)
            with open(hmmer_hit_file, "r") as src:
                for line in src.readlines():
                    query, target = line.strip().split()
                    queries[query][target] = (0.0, 0.0)
            self.max_hmmer_hits = queries

    def compute_embedding(self, sequence, model_class):
        _, _, encoded = self.batch_converter([("sequence", sequence)])
        embedding = model_class(
            encoded.to(self.model_device),
            repr_layers=[33],
            return_contacts=True,
        )
        embedding = embedding["representations"][33].mean(dim=1)
        return embedding

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):
        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(zip(lengths, target_names, targets)):
            # sample every N amino.
            aminos = torch.cat([target[j].unsqueeze(0) for j in range(length)], dim=0)
            self.unrolled_names.extend([name] * length)
            unrolled_targets.append(aminos)

        unrolled_targets = torch.cat(unrolled_targets, dim=0)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
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

        D, I = self.index.search(query_embedding.contiguous(), k=2048)
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))

        return filtered_list
