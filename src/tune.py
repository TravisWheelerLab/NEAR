import torch
import os
from pytorch_lightning import Trainer, seed_everything
from types import SimpleNamespace
from pytorch_lightning.loggers import TensorBoardLogger
from src.callbacks import CallbackSet
import time
import yaml
import argparse
from src.utils.util import (
    load_dataset_class,
    load_model_class,
)
import torch
from torch import multiprocessing
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

HOME = os.environ["HOME"]


def tune_model(train_config, trainer_args, train_dataloader, val_dataloader, model_class,logger):
    model = model_class(**train_config)

    metrics = {"loss":"val_loss"}
    callbacks = [TuneReportCallback(metrics, on = "validation_end")]
        
    trainer = Trainer(
        **trainer_args,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=0.2,
        devices=1,
        # strategy="ddp_find_unused_parameters_false",
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # ckpt_path=params.checkpoint,
    )   

def train(_config, train_config):
    seed_everything(_config["seed"])
    params = SimpleNamespace(**_config)
    model_class = load_model_class(params.model_name)
    dataset_class = load_dataset_class(params.dataset_name)

    train_dataset = dataset_class(**params.train_dataset_args)

    val_dataset = dataset_class(**params.val_dataset_args)

    print(f"Training model {params.model_name} with dataset {params.dataset_name}.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn(), **params.dataloader_args,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=val_dataset.collate_fn(), **params.dataloader_args,
    )

    logger = TensorBoardLogger(save_dir=params.log_dir, name=params.model_name,)

    logger.experiment.add_text(
        tag="description", text_string=params.description, walltime=time.time()
    )
    save_path = os.path.join(params.log_dir, params.model_name, f"version_{logger.version}")

    with open(f"{save_path}/config.yaml", "w") as file:
        yaml.dump(_config, file)

    trainable = tune.with_parameters(tune_model, trainer_args = params.trainer_args, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model_class=model_class, logger=logger)
    analysis = tune.run(trainable,resources_per_trial = {"cpu":os.cpu_count(), "gpu":torch.cuda.device_count()}, metric = "val_loss", mode = "min", config=train_config,num_samples=10)

    print("Best config")
    print(analysis.best_config)
    print("Best checkpoint")
    print(analysis.best_checkpoint)
    print("Best result")
    print(analysis.best_result)
    print("Best trial")
    print(analysis.best_trial)


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()
    configfile = args.config
    if 'yaml' in configfile:
        configfile = configfile[:-5]

    with open(f"src/configs/{configfile}.yaml", "r") as stream:
        _config = yaml.safe_load(stream)


    trainer_args = {"learning_rate": tune.loguniform(1e-7,1e-3),
    "log_interval": 10000,
    "in_channels": 20,
    "indels": True,
    "res_block_n_filters": tune.choice([512, 128, 256]),
    "res_block_kernel_size": tune.choice([3,5,7]),
    "n_res_blocks": tune.choice([6,8,10,12]),
    "padding": "same",
    "padding_mode": "zeros"}

    train(_config, trainer_args)