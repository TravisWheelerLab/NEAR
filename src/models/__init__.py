from abc import ABC, abstractmethod

from pytorch_lightning import LightningModule


class ModelBase(ABC, LightningModule):
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()
