from sacred import Experiment

ex = Experiment()


@ex.config
def config():

    gpus = 1
    num_nodes = 1
    num_workers = 0
    check_val_every_n_epoch = 1
    log_dir = "model_data/aug18/testing"
    batch_size = 32
    epochs = 20
    learning_rate = 1e-4
    seq_len = 150

    # description = input("Describe your experiment.\n")

    model_name = "ResNet1d"
    dataset_name = "SwissProtGenerator"

    embed_msas = False

    model_args = {
        "learning_rate": learning_rate,
        "log_interval": 10,
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
