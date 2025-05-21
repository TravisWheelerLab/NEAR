import argparse
import sys
import torch
import yaml
from pathlib import Path
from .models import NEARResNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="NEAR: Neural Efficient Accurate Retrieval for sequence similarity search"
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    search_parser = subparsers.add_parser('search', help='Search using embeddings')
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for sequences')
    index_parser = subparsers.add_parser('index', help='Build search index')
    train_parser = subparsers.add_parser('train', help='Train a NEAR model')

    return parser.parse_args()


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


def main():
    try:
        args = parse_args()

        if args.command == 'embed':
            pass

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
