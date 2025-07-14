import os
import torch
from pathlib import Path
from .models import NEARResNet
import yaml


def run_index_program(args):
    """
    Build a search index from sequences in a fasta file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments for the index command.
    """
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Find model path if not specified
    model_path = args.model_path
    if model_path is None:
        # Look for default model in installation directory
        package_dir = Path(__file__).parent.absolute()
        model_path = package_dir / "models" / "near_model.pt"
        config_path = package_dir / "models" / "config.yaml"

        if not model_path.exists():
            raise FileNotFoundError(f"Could not find model at {model_path}. Please specify a model path.")
    else:
        model_dir = Path(model_path).parent
        config_path = model_dir / "config.yaml"

    # Load configuration and model
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = NEARResNet(
        embedding_dim=config['embedding_dim'],
        num_layers=config.get('num_layers', 5),
        kernel_size=config.get('kernel_size', 5),
        h_kernel_size=config.get('h_kernel_size', 1),
        in_symbols=config.get('in_symbols', 25)
    )

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Read sequences from fasta file
    print(f"Reading sequences from {args.input_path}")
    ids, sequences = read_fasta(args.input_path)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embed_sequences(model, sequences, device=device)

    # Build the index
    print(f"Building index with {args.index_build_algo} algorithm...")

    # Import needed cagra components
    try:
        from cagra import (
            CagraIndex,
            GPU_CAGRA_NN_DESCENT,
            GPU_CAGRA,
        )
    except ImportError:
        raise ImportError("Could not import cagra. Please install it using pip install cagra")

    # Determine the index build algorithm
    if args.index_build_algo == "Default" or args.index_build_algo == "GPU_CAGRA_NN_DESCENT":
        build_algo = GPU_CAGRA_NN_DESCENT
    elif args.index_build_algo == "GPU_CAGRA":
        build_algo = GPU_CAGRA
    else:
        raise ValueError(f"Unknown index build algorithm: {args.index_build_algo}")

    # Create the index
    index = CagraIndex(
        dim=config['embedding_dim'],
        graph_degree=args.graph_degree,
        intermediate_graph_degree=args.intermediate_graph_degree,
        build_algo=build_algo
    )

    # Set nn_descent parameters if using that algorithm
    if build_algo == GPU_CAGRA_NN_DESCENT:
        index.nn_descent_niter = args.nn_descent_niter

    # Build the index
    index.build(embeddings)

    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save the index, metadata, and sequence info
    print(f"Saving index to {args.out_path}")

    # Create a metadata dictionary
    metadata = {
        "num_sequences": len(ids),
        "embedding_dim": config['embedding_dim'],
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
        "build_algorithm": args.index_build_algo,
        "stride": args.stride
    }

    # Save everything in a dictionary format
    index_data = {
        "index": index,
        "metadata": metadata,
        "ids": ids,
        "embeddings": embeddings,
    }

    torch.save(index_data, args.out_path)
    print(f"Index successfully saved to {args.out_path}")