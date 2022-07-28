import importlib.util
import inspect
import json
import os
import pdb
import pkgutil
import sys
import zipfile
from importlib import reload
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, TextIO, Tuple, Type

from sacred.observers import FileStorageObserver

from src import datasets, models
from src.config import ex
from src.utils import pluginloader

log_dir = "model_data"


def configurator():
    # load user-config
    pdb.set_trace()
    # file system stuff
    # unpack configs variables into our scope
    source_files = set()
    ex.observers.append(FileStorageObserver(f"{log_dir}/{model_name}/"))
    model_class = load_model_class(model_name)
    dataset_class = load_dataset_class(dataset_name)
    ex.add_source_file(inspect.getfile(model_class))
    ex.add_source_file(inspect.getfile(dataset_class))

    source_files.add(inspect.getfile(model_class))
    source_files.add(inspect.getfile(dataset_class))

    # get model class dependencies
    script_package = inspect.getmodule(model_class)
    base_package = importlib.import_module(script_package.__package__.split(".")[0])
    depends = set()
    depends.update(get_dependencies(script_package, base_package))

    # and dataset class dependencies
    script_package = inspect.getmodule(dataset_class)
    base_package = importlib.import_module(script_package.__package__.split(".")[0])
    depends.update(get_dependencies(script_package, base_package))

    # add them to our experiment configuration
    for dependency in depends:
        ex.add_source_file(dependency)
        source_files.add(dependency)


def load_models() -> Dict[str, Type[models.ModelBase]]:
    return {
        m.__name__: m
        for m in pluginloader.load_plugin_classes(models, models.ModelBase)
    }


def load_datasets() -> Dict[str, Type[datasets.DataModule]]:
    return {
        m.__name__: m
        for m in pluginloader.load_plugin_classes(datasets, datasets.DataModule)
    }


def cli():
    pass


def login():
    Login.login()


def list_models():
    for name, plugin in load_models().items():
        print("Name:", name)
        print("Description:", plugin.__doc__)
        print()


def list_datasets():
    for name, plugin in load_datasets().items():
        print("Name:", name)
        print("Description:", plugin.__doc__)
        print()


def _get_dataset(name: str) -> Type[datasets.DataModule]:
    dataset_dict = load_datasets()

    if name not in dataset_dict:
        raise ValueError(f"Dataset {name} not found.")

    return dataset_dict[name]


def _get_model(name: str) -> Type[models.ModelBase]:
    model_dict = load_models()

    if name not in model_dict:
        raise ValueError(f"Model {name} not found.")

    return model_dict[name]


def _resolve_model_working_directory(model_name: str):
    """
    Assumes directory structure
    model_data/model_name/lightning_logs/version_N/
    """
    model_folder = Path(__file__).resolve().parent.parent / "model_data" / model_name
    model_folder.parent.mkdir(exist_ok=True)
    model_folder.mkdir(exist_ok=True)
    return model_folder


def _load_snapshots(model_name: str) -> List[Tuple[int, str]]:
    # Verify the model exists...
    model_folder = _resolve_model_working_directory(model_name)

    model_snapshots = []
    lightning_logs = model_folder / "lightning_logs"

    if not lightning_logs.is_dir():
        return []

    for version in lightning_logs.iterdir():
        parts = version.name.split("_")
        if len(parts) != 2 or parts[0] != "version" or (not parts[1].isnumeric()):
            continue

        version_num = int(parts[1])
        checkpoints = version / "checkpoints"

        if not checkpoints.is_dir():
            continue

        for snapshot in checkpoints.iterdir():
            model_snapshots.append((version_num, snapshot.stem))

    return sorted(model_snapshots)


def _select_snapshot(
    model_name: str, version: int = None, name: str = None
) -> Optional[Path]:
    snapshots = _load_snapshots(model_name)

    # To support partial searches, we run a filters to remove snapshots not match above values...
    for i, val in enumerate([version, name]):
        if val is None:
            continue
        snapshots = [val2 for val2 in snapshots if (val2[i] == val)]

    if len(snapshots) == 0:
        return None
    # Grab the most recent value of snapshots matching the search...
    v, name = snapshots[-1]
    model_folder = _resolve_model_working_directory(model_name)

    return (
        model_folder
        / "lightning_logs"
        / f"version_{v}"
        / "checkpoints"
        / f"{name}.ckpt"
    )


class _WorkingDirectory:
    def __init__(self, path: os.PathLike):
        self._orig_dir = None
        self._path = Path(path).resolve()

    def __enter__(self):
        self._orig_dir = Path.cwd().resolve()
        os.chdir(self._path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._orig_dir)


def list_checkpoints(model_name: str):
    _get_model(model_name)
    model_snapshots = _load_snapshots(model_name)

    if len(model_snapshots) == 0:
        print(f"No snapshots for model {model_name}")
        return

    print(f"Snapshots for model {model_name}:")
    for version, snapshot in sorted(model_snapshots):
        print(f"\tModel Version: {version}, snapshot: {snapshot}")

    print()


def load_dataset_class(dataset_name):
    dataset_cls = _get_dataset(dataset_name)
    return dataset_cls


def load_model_class(model_name):
    model_cls = _get_model(model_name)
    return model_cls


def get_dependencies(script_package, base_package):
    depends = set()
    depends.add(inspect.getfile(base_package))
    for attr in dir(script_package):
        try:
            file_of_item = inspect.getfile(getattr(script_package, attr))
            if str(Path(inspect.getfile(base_package)).resolve().parent) in str(
                Path(file_of_item).resolve()
            ):
                depends.add(file_of_item)

        except TypeError as e:
            pass

    if len(depends) == 0:
        print(f"package {script_package} has no dependencies!")

    return depends


def load_source_snapshot(experiment_directory):

    config = experiment_directory / "config.json"

    with config.open() as src:
        config = SimpleNamespace(**json.load(src))

    archive = experiment_directory / "sources.zip"
    sys.path.insert(0, archive)

    for importer, mod_name, ispkg in pkgutil.iter_modules([sys.path[0]]):
        sub_module = importer.find_module(mod_name).load_module(mod_name)
        sys.modules["_hidden_source"] = sub_module

    from _hidden_source import datasets as datasets
    from _hidden_source import models as models

    reload(models)
    reload(datasets)

    model_class = load_model_class(config.model_name)
    dataset_class = load_dataset_class(config.dataset_name)

    return model_class, dataset_class, config


if __name__ == "__main__":
    experiment_directory = Path("/Users/mac/share/mabe/model_data/TransformerAE/1")
    load_source_snapshot(experiment_directory)
