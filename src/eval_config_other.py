# @evaluation_ex.config
def contrastive_scann():
    device = "cuda"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluatorScaNN"
    description = "ContrastiveEvaluatorScaNN"
    task_id = None
    testing = False
    print(task_id)
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/23"
    checkpoint_path = f"{root}/checkpoints/best_epoch.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    # val_target_file = open("/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r")
    # val_targets = val_target_file.read().splitlines()
    save_dir = (
        "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-scann"
    )

    querysequences, targetsequences, all_hits = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results", query_id=4, save_dir=save_dir,
    )
    print("Loaded all data")
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 10
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )

    if testing:
        querysequences = dict(itertools.islice(querysequences.items(), 0, 1000))
        targetsequences = dict(itertools.islice(targetsequences.items(), 0, 10000))

    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "Flat",
        "figure_path": f"{ROOT}/AlignmentEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 15,
        "max_seq_length": 512,
        "output_path": save_dir,
        "nprobe": 1,
    }


# evaluation_ex.config
def contrastive_blosum_flat():
    device = "cpu"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Use CPU with optimization"
    task_id = None
    print(task_id)
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/4"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    val_target_file = open("/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r")
    val_targets = val_target_file.read().splitlines()
    save_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities"

    querysequences, targetsequences, all_hits = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
        query_id=4,
        val_targets=val_targets,
        save_dir=save_dir,
    )
    print("Loaded all data")
    del val_targets
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 100
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )
    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "Flat",
        "figure_path": f"{ROOT}/BlosumEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 15,
        "max_seq_length": 512,
        "output_path": save_dir,
        "nprobe": 1,
    }


# @evaluation_ex.config
def contrastive_blosum_quantized():
    device = "cuda"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Use CPU with optimization"
    task_id = None
    query_id = 0
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/4"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    val_target_file = open("/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r")
    val_targets = val_target_file.read().splitlines()
    save_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities-IVF"

    querysequences, targetsequences, all_hits = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
        query_id=0,
        val_targets=val_targets,
        save_dir=save_dir,
    )
    print("Loaded all data")
    del val_targets
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 100
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )
    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "IVF256,PQ16",
        "figure_path": f"{ROOT}/BlosumEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 0,
        "max_seq_length": 512,
        "output_path": save_dir,
    }
