"""Initialize dataloaders"""
from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data.dataset import Dataset


class DataModule(Dataset, ABC):
    """abstract DataModule superclass"""
    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def collate_fn(self):
        """
        Return None for default collate function.
        Otherwise, return a function that operates on batch.

        :return:
        :rtype:
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass
