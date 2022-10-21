from typing import IO, Callable, Dict, Optional, Union

import esm
import pytorch_lightning as pl
import torch
from sequence_models.pretrained import load_model_and_alphabet


class CARP(pl.LightningModule):
    def __init__(self):
        super(CARP, self).__init__()
        self.carp_model, self.collater = load_model_and_alphabet("carp_76M")

    def forward(self, x):
        """Ingests already-collated string sequences."""
        return self.carp_model(x)

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def load_from_checkpoint(
        self,
        checkpoint_path: Union[str, IO],
        map_location: Optional[
            Union[Dict[str, str], str, torch.device, int, Callable]
        ] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        return self


class ESM(pl.LightningModule):
    def __init__(self):
        super(ESM, self).__init__()
        self.esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
        self.esm_model.eval()  # disables dropout for deterministic results

    def forward(self, x, **kwargs):
        """Does something based on attention to select a subset of amino acids to return at this step."""
        return self.esm_model(x, **kwargs)

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def load_from_checkpoint(
        self,
        checkpoint_path: Union[str, IO],
        map_location: Optional[
            Union[Dict[str, str], str, torch.device, int, Callable]
        ] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        return self
