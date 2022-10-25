import logging

from sacred import Experiment

from src.utils import to_dict

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


@train_ex.config
def config():

    description = "None"

    log_dir = "/xdisk/twheeler/colligan/model_data/contrastive_2_layers"
    log_verbosity = logging.INFO

    model_name = "ResNet1d"
    dataset_name = "SwissProtGeneratorDanielSequenceEncode"

    global root
    root = "/xdisk/twheeler/colligan/"
    global seq_len
    seq_len = 256
    global batch_size
    batch_size = 64

    @to_dict
    class model_args:
        learning_rate = 1e-3
        log_interval = 300

    @to_dict
    class train_dataset_args:
        fa_file = "/xdisk/twheeler/colligan/uniprot_sprot.fasta"
        minlen = seq_len

    @to_dict
    class val_dataset_args:
        fa_file = "/xdisk/twheeler/colligan/uniprot_sprot.fasta"
        training = False
        minlen = seq_len

    @to_dict
    class dataloader_args:
        batch_size = batch_size
        num_workers = 12
        drop_last = True

    @to_dict
    class trainer_args:
        accelerator = "gpu"
        devices = 1
        num_nodes = 1
        max_epochs = 15
        check_val_every_n_epoch = 1
        log_every_n_steps = 10
