from src.evaluator.contrastive import _calc_embeddings, filter_sequences_by_mask
import torch
import itertools
from multiprocessing.pool import ThreadPool as Pool
import time
import pickle
import numpy as np
import os
from src.data.hmmerhits import FastaFile
from src.utils.loaders import load_model_class


def save_target_embeddings(target_data, model, max_seq_length, device):
    targets, lengths, indices = _calc_embeddings(
        list(target_data.values()), model, device, max_seq_length
    )
    names = np.array(list(target_data.keys()))[indices]
    return names, targets, lengths


def save_off_targets(
    target_sequences,
    target_names_file,
    target_lengths_file,
    unrolled_names_file,
    num_threads,
    model,
    max_seq_length,
    device,
    savedir,
):
    t_chunk_size = len(target_sequences) // num_threads

    print("Embedding targets...")

    start_time = time.time()

    target_names = []
    target_embeddings = []
    target_lengths = []

    target_names, target_embeddings, target_lengths = save_target_embeddings(
        target_sequences, model, max_seq_length, device
    )

    print(f"Number of target embeddings: {len(target_embeddings)}")

    torch.save(target_embeddings, savedir)
    with open(target_names_file, "w") as handle:
        for name in target_names:
            handle.write(f"{name}\n")

    with open(target_lengths_file, "w") as handle:
        for length in target_lengths:
            handle.write(f"{length}\n")

    unrolled_names = []
    for name, length in zip(target_names, target_lengths):
        unrolled_names.append([name] * length)
    with open(unrolled_names_file, "w") as handle:
        for name in unrolled_names:
            handle.write(f"{name}\n")

    loop_time = time.time() - start_time
    print(f"Embedding took: {loop_time}.")

    return target_names, target_lengths, target_embeddings, unrolled_names


def load_targets(
    target_embeddings_file,
    target_names_file,
    target_lengths_file,
    unrolled_names_file,
    masked_target_file,
    target_file,
    num_threads,
    model,
    max_seq_length,
    device,
    mask_repetetive_sequences=True,
):
    maskedtargetfasta = FastaFile(masked_target_file)
    masked_sequences = maskedtargetfasta.data
    targetfasta = FastaFile(target_file)
    target_sequences = targetfasta.data
    assert list(target_sequences.keys()) == list(masked_sequences.keys())

    # get target embeddings
    if not os.path.exists(target_embeddings_file):
        print(f"No saved target embeddings. Calculating them now from {target_file}")

        print(f"Number of target sequences: {len(target_sequences)}")

        (
            target_names,
            target_lengths,
            target_embeddings,
            unrolled_names,
        ) = save_off_targets(
            target_sequences,
            target_names_file,
            target_lengths_file,
            unrolled_names_file,
            num_threads,
            model,
            max_seq_length,
            device,
            target_embeddings_file,
        )
    else:
        target_embeddings = torch.load(target_embeddings)
        print(f"Number of target embeddings: {len(target_embeddings)}")

        with open(target_names_file, "r") as f:
            target_names = f.readlines()
            target_names = [t.strip("\n") for t in target_names]
        with open(target_lengths_file, "r") as f:
            target_lengths = f.readlines()
            target_lengths = [int(t.strip("\n")) for t in target_lengths]
        with open(unrolled_names_file, "r") as f:
            unrolled_names = f.readlines()
            unrolled_names = [t.strip("\n") for t in unrolled_names]

    if mask_repetetive_sequences:
        print("Filtering out masked regions of targets")
        target_embeddings, target_lengths = filter_sequences_by_mask(
            list(masked_sequences.values()), target_embeddings
        )

        print(f"Saving masked targets to ")
        masked_target_lengths_file = target_lengths_file.strip(".txt") + "-masked.txt"
        if not os.path.exists(masked_target_lengths_file):
            print(f"Saving masked target lengths to {masked_target_lengths_file}")

            with open(masked_target_lengths_file, "w") as handle:
                for length in target_lengths:
                    handle.write(f"{length}\n")
        else:
            print(f"Loading masked target lengths from {masked_target_lengths_file}")

            with open(masked_target_lengths_file, "r") as f:
                target_lengths = f.readlines()
                target_lengths = [int(t.strip("\n")) for t in target_lengths]

        unrolled_names_masked = unrolled_names_file.strip(".txt") + "-masked.txt"

        if not os.path.exists(unrolled_names_masked):
            print(
                f"Loading and saving masked unrolled_names to {unrolled_names_masked}"
            )
            unrolled_names = []
            for name, length in zip(target_names, target_lengths):
                unrolled_names.append([name] * length)
            with open(unrolled_names_masked, "w") as handle:
                for name in unrolled_names:
                    handle.write(f"{name}\n")
    return target_embeddings, target_names, target_lengths, unrolled_names


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def load_model(checkpoint_path, model_name, device="cpu"):
    print(f"Loading from checkpoint in {checkpoint_path}")

    model_class = load_model_class(model_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=torch.device(device),
    ).to(device)

    return model
