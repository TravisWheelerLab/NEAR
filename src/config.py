from sacred import Experiment

ex = Experiment()


@ex.config
def config():
    # TODO: wrap this function with sensible defaults,
    # so I don't have to specify all of them in the config.
    gpus = 0
    num_nodes = 1
    num_workers = 0
    check_val_every_n_epoch = 1
    log_dir = "models/may27/18_layer_resnet_with_attn_no_pos_enc"
    batch_size = 16
    epochs = 20
    learning_rate = 1e-4
    seq_len = 400

    model_name = "ResNet1d"
    dataset_name = "SwissProtGenerator"
    embed_msas = False

    model_args = {
        "learning_rate": learning_rate,
        "embed_msas": embed_msas,
        "apply_attention": False,
    }

    train_dataset_args = {
        "fa_file": "/Users/mac/data/uniprot_sprot.fasta",
        "minlen": seq_len,
    }

    val_dataset_args = {
        "fa_file": "/Users/mac/data/uniprot_sprot.fasta",
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
