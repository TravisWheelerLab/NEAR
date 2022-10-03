import logging

from sacred import Experiment

from src.utils import to_dict

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


@train_ex.config
def config():

    description = (
        "Seq length 128, no cnn loss, 4 downsample steps for a fina"
        "l embedding length of 8. This model is also quite heavy in parameters on the upsampling"
    )

    gpus = 1
    num_nodes = 1
    num_workers = 12
    check_val_every_n_epoch = 1
    log_dir = "/xdisk/twheeler/colligan/indel_model_data/smaller_field_generated/"
    batch_size = 32
    epochs = 100
    learning_rate = 1e-4
    seq_len = 32
    log_interval = 100
    downsample_steps = 2
    apply_cnn_loss = False
    apply_contrastive_loss = False
    backprop_on_near_aminos = False
    log_verbosity = logging.INFO

    pool_type = "mean"

    model_name = "SequenceVAE"
    dataset_name = "SwissProtGeneratorDanielSequenceEncode"

    embed_msas = False
    root = "/xdisk/twheeler/colligan/"
    # root = "/home/tc229954/prefilter_data/"

    cnn_model_state_dict = f"{root}/data/prefilter/model_16.sdic"

    model_args = {
        "learning_rate": learning_rate,
        "log_interval": log_interval,
        "cnn_model_state_dict": cnn_model_state_dict,
        "initial_seq_len": seq_len,
        "apply_cnn_loss": apply_cnn_loss,
        "downsample_steps": downsample_steps,
        "pool_type": pool_type,
        "apply_contrastive_loss": apply_contrastive_loss,
        "backprop_on_near_aminos": backprop_on_near_aminos,
    }

    train_dataset_args = {
        "fa_file": f"{root}/uniprot_sprot.fasta",
        "minlen": seq_len,
    }

    val_dataset_args = {
        "fa_file": f"{root}/uniprot_sprot.fasta",
        "training": False,
        "minlen": seq_len,
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
