import logging

import yaml
from sacred import Experiment

# from src.datasets.datasets import sanitize_sequence
from src.utils.helpers import to_dict

logger = logging.getLogger("evaluate")
logger.setLevel(logging.WARNING)

evaluation_ex = Experiment()
# convert a class to a dictionary with a decorator


@evaluation_ex.config
def config():

    device = "cuda"
    model_name = "ResNetParamFactory"
    evaluator_name = "TemporalBenchmark"
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = "/xdisk/twheeler/colligan/model_data/resnet_meanpool/ResNet1dSequencePool/1"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    @to_dict
    class evaluator_args:
        model_device = "cuda"
        index_device = "cuda"
        index_string = "Flat"
        n_neighbors = 100
