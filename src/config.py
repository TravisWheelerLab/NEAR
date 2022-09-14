import logging

from sacred import Experiment

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


@train_ex.config
def config():

    description = "A contrastive model for testing."
    gpus = 1
    num_nodes = 1
    num_workers = 0
    check_val_every_n_epoch = 1
    log_dir = "model_data/sept6/"
    batch_size = 32
    epochs = 30
    learning_rate = 1e-4
    seq_len = 32
    log_interval = 100

    model_name = "SequenceVAE"
    dataset_name = "SwissProtGeneratorDanielSequenceEncode"

    embed_msas = False
    root = "/xdisk/twheeler/colligan/"

    cnn_model_state_dict = f"{root}/data/prefilter/model_16.sdic"

    model_args = {
        "learning_rate": learning_rate,
        "log_interval": log_interval,
        "cnn_model_state_dict": cnn_model_state_dict,
        "initial_seq_len": seq_len,
    }

    train_dataset_args = {
        "fa_file": f"{root}/data/prefilter/uniprot_sprot.fasta",
        "minlen": seq_len,
    }

    val_dataset_args = {
        "fa_file": f"{root}/data/prefilter/uniprot_sprot.fasta",
        "minlen": seq_len,
        "training": False,
    }

    dataloader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": True,
    }

    trainer_args = {
        "gpus": gpus,
        "num_nodes": num_nodes,
        "max_epochs": epochs,
        "check_val_every_n_epoch": check_val_every_n_epoch,
    }
