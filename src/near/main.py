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

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for sequences')
    embed_parser.add_argument('config', type=str, help='Path to model configuration YAML file')
    embed_parser.add_argument('model', type=str, help='Path to trained model weights')
    embed_parser.add_argument('input', type=str, help='Input FASTA file')
    embed_parser.add_argument('output', type=str, help='Output NPZ file for embeddings')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search using embeddings')
    search_parser.add_argument('-q', '--query', required=True, help='Query embeddings NPZ file')
    search_parser.add_argument('-t', '--target', required=True, help='Target embeddings NPZ file')
    search_parser.add_argument('-o', '--output', required=True, help='Output CSV file for hits')
    search_parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for search')
    search_parser.add_argument('--query_sequence', help='Query sequence FASTA file (for softmasking)')
    search_parser.add_argument('--target_sequence', help='Target sequence FASTA file (for softmasking)')

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


def embed_sequences(args):
    config = load_config(args.config)
    model = load_model(config, args.model)

    # TODO: Implement sequence embedding logic
    # 1. Load sequences from args.input
    # 2. Process sequences through model
    # 3. Save embeddings to args.output
    pass


def search_embeddings(args):
    # TODO: Implement search logic
    # 1. Load query and target embeddings
    # 2. Set up FAISS index
    # 3. Perform search
    # 4. Process results with process_near_results
    # 5. Save to output file
    pass


def main():
    try:
        args = parse_args()

        if args.command == 'embed':
            embed_sequences(args)
        elif args.command == 'search':
            search_embeddings(args)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
