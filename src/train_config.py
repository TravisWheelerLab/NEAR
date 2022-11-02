import logging

from sacred import Experiment

from src.utils import to_dict

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


@train_ex.config
def config():

    description = "None"

    # log_dir = "/home/tc229954/model_data"
    log_dir = "/xdisk/twheeler/colligan/model_data/spectrogram_real_data"
    log_verbosity = logging.INFO

    model_name = "ResNet1dKmerSampler"
    dataset_name = "UniProtSpectDataset"

    global seq_len
    seq_len = 256
    global kmer_length
    kmer_length = 32
    global batch_size
    batch_size = 512
    global num_workers
    num_workers = 12

    @to_dict
    class model_args:
        learning_rate = 1e-5
        log_interval = 100
        n_res_blocks = 5
        res_block_kernel_size = 3
        res_block_n_filters = 256
        in_channels = 50

    @to_dict
    class train_dataset_args:
        minlen = seq_len
        training = True
        fa_file = "/xdisk/twheeler/colligan/uniprot_sprot.fasta"

    @to_dict
    class val_dataset_args:
        minlen = seq_len
        training = False
        fa_file = "/xdisk/twheeler/colligan/uniprot_sprot.fasta"

    @to_dict
    class dataloader_args:
        batch_size = batch_size
        num_workers = num_workers
        drop_last = True

    @to_dict
    class trainer_args:
        accelerator = "gpu"
        devices = 1
        num_nodes = 1
        max_epochs = 500
        check_val_every_n_epoch = 1
        log_every_n_steps = 10
