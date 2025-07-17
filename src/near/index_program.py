import os
import torch
from pathlib import Path
import yaml

from .models import NEARResNet
from .fasta_data import FASTAData
from .index import NEARIndex

def run_index_program(args):
    """
    Build a search index from sequences in a fasta file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments for the index command.
    """

    print(f"Using device: {args.device}")
    device = torch.device(args.device)


    # Find model path if not specified
    model_path = args.model_path
    if model_path is None:
        # Look for default model in installation directory
        package_dir = Path(__file__).parent.absolute()
        model_path = package_dir / "models" / "resnet_877_256.pt"
        config_path = package_dir / "models" / "resnet_877_256.yaml"

        if not model_path.exists():
            raise FileNotFoundError(f"Could not find model at {model_path}. Please specify a model path.")
    else:
        model_dir = Path(model_path).parent
        config_path = model_dir / "config.yaml"

    # Load configuration and model
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = NEARResNet( **config['model_args'])

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model = model.to(device)
    model.half()
    model.eval()

    # Read sequences from fasta file
    if args.verbose:
        print(f"Reading sequences from {args.input_path}")
    fasta_data =  FASTAData(args.input_path)
    if args.verbose:
        print(f"{len(fasta_data.seqid_to_name)} sequences kept for index")

    # Generate embeddings
    if args.verbose:
        print("Creating index")

    index = NEARIndex(fasta_data)
    index.create_index(model,
                       stride                       = args.stride,
                       index_build_algo             = args.index_build_algo,
                       graph_degree                 = args.graph_degree,
                       intermediate_graph_degree    = args.intermediate_graph_degree,
                       nn_descent_niter             = args.nn_descent_niter,
                       itopk_size                   = args.itopk_size,
                       model_dims                   = config['model_args']['embedding_dim'],
                       verbose                      = args.verbose,
                       device                       = device)

    if args.verbose:
        print(f"Saving index to {args.out_path}")
    index.save_to_path(args.out_path)