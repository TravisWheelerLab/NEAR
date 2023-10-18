import logging
import faiss
import torch
import h5py
import time
import os

from src.utils.eval_utils import load_targets

log = logging.getLogger("__name__")


def create_faiss_index(
    embeddings,
    embed_dim,
    index_string,
    device="cpu",
    distance_metric="cosine",
    gpu_num=0,
):
    log.info(f"using index {index_string} with {distance_metric} metric.")

    embeddings = embeddings[torch.randperm(embeddings.shape[0])]
    log.info(f"Sampling {embeddings.shape[0]} embeddings.")

    if distance_metric == "cosine":
        log.info("Normalizing embeddings for use with cosine metric.")
        index = faiss.index_factory(embed_dim, index_string, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.index_factory(embed_dim, index_string)

    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_num, index)
        index.train(embeddings.numpy())  # .to("cuda"))
    else:
        index.train(embeddings.numpy())
    
    return index


def save_FAISS_results(
    query_names,
    all_scores,
    all_indices,
    scores_path,
    indices_path,
    query_names_path,
):
    print(f"Saving FAISS results to {scores_path} and {indices_path}")

    with h5py.File(scores_path, "w") as hf:
        for i, arr in enumerate(all_scores):
            hf.create_dataset(f"array_{i}", data=arr)

    with h5py.File(indices_path, "w") as hf:
        for i, arr in enumerate(all_indices):
            hf.create_dataset(f"array_{i}", data=arr)

    with open(query_names_path, "w") as f:
        for name in query_names:
            f.write(name + "\n")


def _setup_targets_for_search(
    target_embeddings,
    index_string,
    nprobe,
    num_threads=1,
    normalize_embeddings=True,
    index_device="cpu",
    index_path="/xdisk/twheeler/daphnedemekas/faiss-index-targets.index",
):
    """Creates the Faiss Index object using the unrolled
    target embddings"""
    start = time.time()
    if not os.path.exists(index_path):
        print(f"Creating index: {index_string} and saving to {index_path}")
        unrolled_targets = torch.cat(target_embeddings, dim=0)
        unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)
        index: faiss.Index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if normalize_embeddings else "l2",
            index_string=index_string,  # f"IVF{K},PQ8", #self.index_string, #f"IVF100,PQ8", #"IndexIVFFlat", #self.index_string,
            device=index_device,
        )
        log.info("Adding targets to index.")
        if index_device == "cpu":
            index.add(unrolled_targets.to("cpu").numpy())
            faiss.write_index(index, index_path)
        else:
            index.add(unrolled_targets.numpy())
            faiss.write_index(index, index_path)
    else:    
        print(f"Reading index from {index_path}")
        index = faiss.read_index(index_path)

    index.nprobe = nprobe
    loop_time = time.time() - start

    print(f"Index Creation took: {loop_time}.")

    return index


def load_index(params, model):
    if os.path.exists(params.index_path):
        index = faiss.read_index(params.index_path)
        index.nprobe = params.nprobe
        if params.device == "cuda":
            num = 0
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, int(num), index)
    else:
        target_embeddings, target_names, target_lengths = load_targets(
            params.target_embeddings,
            params.target_names,
            params.target_lengths,
            params.target_file,
            params.num_threads,
            model,
            params.max_seq_length,
            params.device,
        )
        assert (
            len(target_lengths) == len(target_names) == len(target_embeddings)
        ), "Target lengths, names and embeddings are not all the same length"

        # if params.device == 'cuda':
        # target_embeddings = target_embeddings.to("cpu").numpy()
        index = _setup_targets_for_search(
            target_embeddings,
            params.index_string,
            params.nprobe,
            params.omp_num_threads,
            index_path=params.index_path,
            index_device = params.index_device,
        )
    faiss.omp_set_num_threads(params.omp_num_threads)

    return index
