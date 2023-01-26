"""Config file for running training code
Training code is in src/__init__.py"""

import logging
import os

HOME = os.environ["HOME"]

from sacred import Experiment

from src.utils import to_dict

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


# @train_ex.config
def config():

    description = "None"
    model_name = "ResNet1d"
    dataset_name = "SwissProtGenerator"
    log_dir = f"{HOME}/prefilter"
    log_verbosity = logging.INFO

    @to_dict
    class model_args:
        learning_rate = 1e-3
        log_interval = 100
        in_channels = 20

    @to_dict
    class train_dataset_args:
        minlen = 256
        fa_file = f"{HOME}/T_benchmark2k30k.fa"

    @to_dict
    class dataloader_args:
        batch_size = 32
        num_workers = 6
        drop_last = True

    @to_dict
    class trainer_args:
        accelerator = "gpu"
        num_nodes = 1
        max_epochs = 100

    @to_dict
    class val_dataset_args:
        minlen = 256
        fa_file = f"{HOME}/Q_benchmark2k30k.fa"


@train_ex.config
def vae_config():

    description = "Turning up the number of downsampling steps"
    model_name = "VAEIndels"
    dataset_name = "AlignmentGenerator"
    log_dir = f"{HOME}/prefilter"
    log_verbosity = logging.INFO

    @to_dict
    class model_args:
        learning_rate = 1e-5
        log_interval = 100
        downsample_steps = 6

    @to_dict
    class train_dataset_args:
        ali_path = "/xdisk/twheeler/colligan/indel_training_set/alignments"
        seq_len = 256

    @to_dict
    class dataloader_args:
        batch_size = 32
        num_workers = 6
        drop_last = True

    @to_dict
    class trainer_args:
        accelerator = "gpu"
        num_nodes = 1
        max_epochs = 100

    @to_dict
    class val_dataset_args:
        ali_path = "/xdisk/twheeler/colligan/indel_training_set/alignments"
        seq_len = 256
        training = False
