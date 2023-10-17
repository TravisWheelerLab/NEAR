import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from types import SimpleNamespace
from pytorch_lightning.loggers import TensorBoardLogger
from src.callbacks import CallbackSet
import time
import yaml
import argparse
from src.utils.loaders import (
    load_dataset_class,
    load_model_class,
)

HOME = os.environ["HOME"]


def train(_config):
    seed_everything(_config["seed"])
    params = SimpleNamespace(**_config)
    model_class = load_model_class(params.model_name)
    dataset_class = load_dataset_class(params.dataset_name)

    model = model_class(**params.model_args)
    train_dataset = dataset_class(**params.train_dataset_args)

    val_dataset = dataset_class(**params.val_dataset_args)

    print(f"Training model {params.model_name} with dataset {params.dataset_name}.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn(),
        **params.dataloader_args,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate_fn(),
        **params.dataloader_args,
    )

    logger = TensorBoardLogger(
        save_dir=params.log_dir,
        name=params.model_name,
    )

    logger.experiment.add_text(
        tag="description", text_string=params.description, walltime=time.time()
    )
    save_path = os.path.join(
        params.log_dir, params.model_name, f"version_{logger.version}"
    )

    with open(f"{save_path}/config.yaml", "w") as file:
        yaml.dump(_config, file)

    trainer = Trainer(
        **params.trainer_args,
        #        callbacks=[EarlyStopping(monitor='val_loss')],
        callbacks=CallbackSet.callbacks(),
        logger=logger,
        log_every_n_steps=10000,
        # val_check_interval=0.2,
        devices=1,
        # strategy="ddp_find_unused_parameters_false",
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=params.checkpoint,
    )


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()
    configfile = args.config
    if "yaml" in configfile:
        configfile = configfile[:-5]

    with open(f"src/configs/{configfile}.yaml", "r") as stream:
        _config = yaml.safe_load(stream)
    train(_config)
