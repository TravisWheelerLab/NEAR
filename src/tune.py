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
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import LightningTrainer, LightningConfigBuilder

HOME = os.environ["HOME"]


def run_tune(num_samples, trainer, lightning_config):
    scheduler = ASHAScheduler(max_t=2, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=air.RunConfig(
            storage_path="/xdisk/twheeler/daphnedemekas/ray_tune", name="tune"
        ),
    )

    print("Fitting tuner...")
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")

    print(best_result)
    print("Best config")
    print(results.best_config)
    print("Best checkpoint")
    print(results.best_checkpoint)
    print("Best result")
    print(results.best_result)
    print("Best trial")
    print(results.best_trial)


def tune_model(train_config, train_dataloader, val_dataloader, model_class, logger):

    config = {
        "learning_rate": tune.loguniform(1e-7, 1e-3),
        "res_block_n_filters": tune.choice([512, 128, 256]),
        "res_block_kernel_size": tune.choice([3, 5, 7]),
        "n_res_blocks": tune.choice([6, 8, 10, 12]),
    }

    train_config.update(config)

    print("Train config:")
    print(train_config)
    lightning_config = (
        LightningConfigBuilder()
        .module(cls=model_class, **train_config)
        .trainer(
            max_epochs=2, accelerator="gpu", logger=logger, precision=16, log_every_n_steps=10000
        )
        .fit_params(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        .checkpointing(monitor="val_loss", save_top_k=2, mode="min", every_n_epochs=1000)
        .build()
    )
    scaling_config = ScalingConfig(
        use_gpu=True, num_workers=1, resources_per_worker={"CPU": 20, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )

    trainer = LightningTrainer(
        scaling_config=scaling_config,
        run_config=run_config,
    )

    run_tune(10, trainer, lightning_config)


def tune_pipeline(_config):
    seed_everything(_config["seed"])
    params = SimpleNamespace(**_config)
    model_class = load_model_class(params.model_name)
    dataset_class = load_dataset_class(params.dataset_name)

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
    save_path = os.path.join(params.log_dir, params.model_name, f"version_{logger.version}")

    with open(f"{save_path}/config.yaml", "w") as file:
        yaml.dump(_config, file)
    tune_model(params.model_args, train_dataloader, val_dataloader, model_class, logger)
    # trainable = tune.with_parameters(tune_model, trainer_args = params.trainer_args, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model_class=model_class, logger=logger)
    # analysis = tune.run(trainable,resources_per_trial = {"cpu":os.cpu_count(), "gpu":torch.cuda.device_count()}, metric = "val_loss", mode = "min", config=train_config,num_samples=10)


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

    tune_pipeline(_config)
