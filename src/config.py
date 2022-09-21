import logging

from sacred import Experiment

from src.utils import to_dict

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


@train_ex.config
def config():

    description = "Add in the softmax on the reconstructed embeddings."

    gpus = 1
    num_nodes = 1
    num_workers = 12
    check_val_every_n_epoch = 1
    log_dir = "/xdisk/twheeler/colligan/model_data/"
    batch_size = 32
    epochs = 100
    learning_rate = 1e-4
    seq_len = 256
    log_interval = 100
    downsample_steps = 5
    apply_cnn_loss = True
    apply_contrastive_loss = True
    backprop_on_near_aminos = False

    pool_type = "mean"

    model_name = "ResNetSparseAttention"
    dataset_name = "SwissProtGeneratorDanielSequenceEncode"

    log_verbosity = logging.DEBUG
    embed_msas = False
    root = "/xdisk/twheeler/colligan/"

    cnn_model_state_dict = f"{root}/data/prefilter/model_16.sdic"

    model_args = {
        "learning_rate": learning_rate,
        "log_interval": log_interval,
        "cnn_model_state_dict": cnn_model_state_dict,
        "apply_contrastive_loss": apply_contrastive_loss,
        "backprop_on_near_aminos": backprop_on_near_aminos,
        "initial_seq_len": seq_len,
        "apply_cnn_loss": apply_cnn_loss,
        "downsample_steps": downsample_steps,
        "pool_type": pool_type,
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
