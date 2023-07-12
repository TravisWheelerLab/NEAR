
import itertools
import yaml
from src.train import train
import argparse
# Define the hyperparameter options
lr_values = [1e-7, 1e-5, 1e-3]
n_filters_values = [512, 128, 256]
kernel_size_values = [3,5,7]
n_blocks_values = [ 6,8,10]

# Generate all possible combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(lr_values, n_filters_values, kernel_size_values, n_blocks_values))

if __name__ == '__main__':

 # multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("param_index")

    args = parser.parse_args()

    numpossibilities = len(hyperparameter_combinations)
    print(f"Possibility {args.param_index} / {numpossibilities}")
    lr, n_filters, kernel_size, n_blocks = hyperparameter_combinations[int(args.param_index)]



    configfile = args.config
    if 'yaml' in configfile:
        configfile = configfile[:-5]

    with open(f"src/configs/{configfile}.yaml", "r") as stream:
        _config = yaml.safe_load(stream)

    _config["model_args"]["learning_rate"] = lr
    _config["model_args"]["res_block_n_filters"] = n_filters
    _config["model_args"]["res_block_kernel_size"] = kernel_size
    _config["model_args"]["n_res_blocks"] = n_blocks

    train(_config)

