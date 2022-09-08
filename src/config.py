from sacred import Experiment

train_ex = Experiment()


@train_ex.config
def config():

    description = "A test of the contrastive dot product model (with normalization)."
    gpus = 1
    num_nodes = 1
    num_workers = 12
    check_val_every_n_epoch = 1
    log_dir = "model_data/sept6/"
    batch_size = 32
    epochs = 100
    learning_rate = 1e-4
    seq_len = 150
    log_interval = 1000

    model_name = "ResNet1d"
    dataset_name = "SwissProtGenerator"

    embed_msas = False

    model_args = {
        "learning_rate": learning_rate,
        "log_interval": log_interval,
    }

    train_dataset_args = {
        "fa_file": "/home/u4/colligan/data/prefilter/uniprot_sprot.fasta",
        "minlen": seq_len,
    }

    val_dataset_args = {
        "fa_file": "/home/u4/colligan/data/prefilter/uniprot_sprot.fasta",
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
