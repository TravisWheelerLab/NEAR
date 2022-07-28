"""
Prefilter passes good candidates to hmmer.
"""

__version__ = "0.0.1"

from types import SimpleNamespace

from pytorch_lightning import seed_everything

from src.config import ex


@ex.main
def train(_config):

    seed_everything(_config["seed"])
    params = SimpleNamespace(**_config)
    model = params.model_class(**params.model_args)
    data_paths = download_dataset(True, False)
    train_dataset = params.dataset_class(data_paths, **params.train_dataset_args)

    if hasattr(params, "val_dataset_args"):
        val_dataset = params.dataset_class(data_paths, **params.val_dataset_args)
    else:
        val_dataset = None

    print(f"Training model {params.model_name} with dataset {params.dataset_name}.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn(), **params.dataloader_args
    )

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, collate_fn=val_dataset.collate_fn(), **params.dataloader_args
        )
    else:
        val_dataloader = None

    # TODO: Add shopty integration.
    # Below we do some path manipulation so TB logs to the same
    # directory as the FileObserver.

    logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=os.path.split(ex.observers[0].dir)[0],
        version=Path(ex.observers[0].dir).name,
        name="",
    )

    trainer = Trainer(
        **params.trainer_args, callbacks=CallbackSet.callbacks(), logger=logger
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


ex.run()
