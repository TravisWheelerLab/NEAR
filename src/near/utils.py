import torch
import yaml
from pathlib import Path
from .models import NEARResNet

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(config, model_path):
    model = NEARResNet(
        embedding_dim=config['embedding_dim'],
        num_layers=config.get('num_layers', 5),
        kernel_size=config.get('kernel_size', 5),
        h_kernel_size=config.get('h_kernel_size', 1),
        in_symbols=config.get('in_symbols', 25)
    )
    model.load_state_dict(torch.load(model_path))
    return model