from typing import Dict, Type

import pytorch_lightning as pl

from src import datasets, evaluator, model
from src.utils import pluginloader


def load_models() -> Dict[str, Type[pl.LightningModule]]:
    return {m.__name__: m for m in pluginloader.load_plugin_classes(model, pl.LightningModule)}


def load_datasets() -> Dict[str, Type[datasets.DataModule]]:
    return {m.__name__: m for m in pluginloader.load_plugin_classes(datasets, datasets.DataModule)}


def load_evaluators() -> Dict[str, Type[evaluator.Evaluator]]:
    return {m.__name__: m for m in pluginloader.load_plugin_classes(evaluator, evaluator.Evaluator)}


def _get_dataset(name: str) -> Type[datasets.DataModule]:
    dataset_dict = load_datasets()
    print(dataset_dict)

    if name not in dataset_dict:
        raise ValueError(f"Dataset {name} not found.")

    return dataset_dict[name]


def _get_evaluator(name: str) -> Type[evaluator.Evaluator]:
    evaluator = load_evaluators()

    if name not in evaluator:
        raise ValueError(f"Evaluator {name} not found.")

    return evaluator[name]


def _get_model(name: str) -> Type[pl.LightningModule]:
    model_dict = load_models()

    if name not in model_dict:
        raise ValueError(f"Model {name} not found.")

    return model_dict[name]


def load_dataset_class(dataset_name):
    dataset_cls = _get_dataset(dataset_name)
    return dataset_cls


def load_evaluator_class(evaluator_name):
    evaluator_cls = _get_evaluator(evaluator_name)
    return evaluator_cls


def load_model_class(model_name):
    model_cls = _get_model(model_name)
    return model_cls