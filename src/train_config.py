import logging
from glob import glob

from sacred import Experiment

from src.utils import to_dict

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


@train_ex.config
def config():

    description = "None"

    log_dir = "/xdisk/twheeler/colligan/model_data/resnet_meanpool"
    log_verbosity = logging.INFO

    model_name = "ResNet1dSequencePool"
    dataset_name = "PfamDataset"

    global root
    root = "/xdisk/twheeler/colligan/"
    global seq_len
    seq_len = 256
    global batch_size
    batch_size = 512

    # root = "/home/tc229954/prefilter_data/"

    @to_dict
    class model_args:
        learning_rate = 1e-3
        log_interval = 100

    @to_dict
    class train_dataset_args:
        train_files = glob("/xdisk/twheeler/colligan/1000_file_subset/*train*")
        valid_files = glob("/xdisk/twheeler/colligan/1000_file_subset/*valid*")
        minlen = seq_len

    @to_dict
    class val_dataset_args:
        train_files = glob("/xdisk/twheeler/colligan/1000_file_subset/*train*")
        valid_files = glob("/xdisk/twheeler/colligan/1000_file_subset/*valid*")
        training = False
        minlen = seq_len

    @to_dict
    class dataloader_args:
        batch_size = batch_size
        num_workers = 12
        drop_last = True

    @to_dict
    class trainer_args:
        gpus = 1
        num_nodes = 1
        max_epochs = 1000
        check_val_every_n_epoch = 1
        log_every_n_steps = 10
