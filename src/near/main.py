import argparse
import sys
import torch
import yaml
from pathlib import Path
from .models import NEARResNet
from .index_program import run_index_program

def parse_args():
    parser = argparse.ArgumentParser(
        description="NEAR: Neural Embeddings for Amino acid Relationships"
                    "A tool to quickly find similar sequence pairs in large datasets"
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    search_parser = subparsers.add_parser('search', help='Search using embeddings')
    search_parser.add_argument('-o', '--output', type=str, default=None, required=True, help='Output file path')

    search_parser.add_argument('-q', '--query', type=str, default=None,
                               help='Path to the query fasta file')
    search_parser.add_argument('--query_emb', type=str, default=None,
                               help='Path to the query embedding file')

    search_parser.add_argument('-t', '--target', type=str, default=None,
                               help='Path to the target fasta file')
    search_parser.add_argument('--target_index', type=str, default=None,
                               help='Path to the target index file')
    search_parser.add_argument('--target_emb', type=str, default=None,
                               help='Path to the target embedding file')

    search_parser.add_argument("--index_type", choices=["Default", "GPUCagra", "GPUCagraNN"],
        default="Default",
        help="Which backend to use (Default=GPUCagraNN, Cagra, or GPUCagraNN)")
    search_parser.add_argument("--index_degree", type=int,
                               default=64,
                               help="The degree of graph used by the index")
    search_parser.add_argument("--save_path", type=str,
                               default=None,
                               help="Where to save the index")
    search_parser.add_argument("--discard_freq", type=int,
                               default=16,
                               help="The discard frequency for the index")
    search_parser.add_argument("-k", "--top_k", type=int, default=64,
                               help='Number of nearest neighbors to return for each query embedding')
    search_parser.add_argument('--model_path', type=str, default=None,
                               help='Path to the model json file. '
                                    'Searches for local installation if unspecified.'
                                    'Searches online if no local installation is found')
    search_parser.add_argument('-d', '--device', type=str, default="cuda", help='Device to use for embedding')

    index_parser = subparsers.add_parser('index', help='Build search index')
    index_parser.add_argument('-o', '--out_path', type=str, default=None, required=True, help='Output file path')
    index_parser.add_argument('-i', '--input_path', type=str, default=None, required=True, help='Path to the input fasta file')
    index_parser.add_argument('-m', '--model_path', type=str, default=None, help='Path to the model json file')
    index_parser.add_argument('-d', '--device', type=str, default="cuda", help='Device to use for embedding')
    index_parser.add_argument("--index_build_algo", choices=["Default", "GPU_CAGRA_NN_DESCENT", "GPU_CAGRA"],
        default="Default",
        help="Which backend to use (Default=GPU_CAGRA_NN_DESCENT, GPU_CAGRA_NN_DESCENT, or GPU_CAGRA)")

    index_parser.add_argument("--graph_degree", type=int,
                               default=128,
                               help="The degree of graph used by the index")

    index_parser.add_argument("--intermediate_graph_degree", type=int,
                              default=256,
                              help="The intermediate degree of graph used by the index")

    index_parser.add_argument("--nn_descent_niter", type=int,
                              default=100,
                              help="The number of NN descent iterations when using GPU_CAGRA_NN_DESCENT")

    index_parser.add_argument("--itopk_size",
                              default=64,
                              help="Internal topk size list used by CAGRA")

    index_parser.add_argument("--stride", type=int,
                               default=8,
                               help="The stride used by the target index")

    return parser.parse_args()

def main():
    try:
        args = parse_args()

        if args.command == 'index':
            run_index_program(args)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
